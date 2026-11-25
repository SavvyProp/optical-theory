"""Compare dense forward reads at 20 ps and 50 GHz for 64 and 128 channels.

Run OPENBLAS_NUM_THREADS=1 python3 -B notes/hankel_size_scaling.py.
All gains are spectrally flat. The added 128-channel delay can have scalar
propagation loss. The two pump powers are in-guide. The electrical budget
retains the 64-channel assumptions and extrapolates output pulse drive.
"""
from dataclasses import dataclass

import numpy as np
from scipy.constants import c, h

from hankel_amplifier_sweep import GATE, MESH, REJECTION, TARGET, added_photons
from hankel_fwm_stress import ETA, PUMP_W, phase_matched_output
from hankel_power_budget import DC_EFFICIENCY, budget
from hankel_single_read_energy import waveguide_field_coefficient
from hankel_temporal_combiner import masked_weights


BASE_INTERVAL_NS = 3.493
TRAVERSAL = 2 * 10**(-0.5 / 10)  # Timed selector recovery relative to a fixed output port.
COUPLING = waveguide_field_coefficient(4, .2) / waveguide_field_coefficient(2.12, .8)
PHOTON_J = h * c / 1.55e-6
BASE_BUDGET = budget(BASE_INTERVAL_NS, gated_booster=True, pulsed_reference=True,
                     extra_booster_pj=50)[0]
PUMP_BALANCE = np.sqrt(BASE_BUDGET['weight loop'] / BASE_BUDGET['reference'])
BASE_REF = .29 * PUMP_W * PUMP_BALANCE
BASE_WEIGHT = .29 * PUMP_W / PUMP_BALANCE


@dataclass
class Program:
    target: np.ndarray
    weights: np.ndarray
    vectors: np.ndarray


def workload(n):
    """Match the 64-channel workload and repeat the same sampling rule at 128."""
    rng = np.random.default_rng(12023)
    index = np.arange(n)
    chirp = np.exp(1j * np.pi * index**2 / (2 * n))
    masks = np.exp(2j * np.pi * np.outer(n-1-index, index) / n) / np.sqrt(n)
    programs = []
    for _ in range(16):
        matrix = rng.uniform(-1, 1, (n, n))
        matrix *= np.sqrt(n) / np.linalg.norm(matrix)
        vectors = rng.uniform(-1, 1, (4, n))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        weights = masked_weights((chirp[:, None] * matrix * chirp).T, masks)
        programs.append(Program(matrix, weights, vectors))
    return programs, chirp


def weak_signal_transfer(n, scale, mix, delay_transmission):
    return (mix * MESH**2 * REJECTION * GATE * TRAVERSAL**2 * delay_transmission
            / (16 * n * scale**2))


def errors(programs, chirp, scale, input_pj, ref_w, weight_w, samples, steps,
           extra_delay_loss_db=0.):
    n = len(chirp)
    # Use the original calibrated short-guide RK4 solver with all input field
    # amplitudes multiplied by sqrt(COUPLING) to model the longer guide.
    input_amp = np.sqrt(input_pj / 20 * TRAVERSAL * MESH * COUPLING / (2*n))
    weight_amp = np.sqrt(weight_w * COUPLING) / scale
    reference_amp = np.sqrt(ref_w * COUPLING)
    normalization = np.sqrt(n*ETA) * reference_amp * weight_amp * input_amp / PUMP_W
    # In the added binary stage, the first half takes its long path on return,
    # the second half on launch. Every desired signal traverses it exactly once.
    long_transmission = 10**(-extra_delay_loss_db/10)
    launch = np.ones(n)
    launch[n//2:] = np.sqrt(long_transmission)
    collection = launch[::-1]
    normalization *= np.sqrt(long_transmission)
    shifted = (np.arange(samples) - (n-1-np.arange(n))[:, None] * (samples//n)) % samples

    # Read.signal, Read.raw and Read.spontaneous of hankel_amplifier_sweep
    # generalized to n exactly equal-amplitude branches. The last factor of
    # 1/2 in raw accounts for two incoherent ASE windows in the B copier.
    mix = ETA * COUPLING**2 * ref_w * weight_w / PUMP_W**2
    signal = weak_signal_transfer(n, scale, mix, long_transmission)
    raw = MESH * GATE * TRAVERSAL * np.mean(collection**2) / 4
    early = 10.
    late = TARGET / (early * signal)
    tally = np.zeros(3)
    for program in programs:
        weight = weight_amp * samples * np.fft.ifft(program.weights, n=samples, axis=1)
        slot_energies = np.sum(abs(program.weights)**2, axis=1)
        spontaneous = (mix * REJECTION * MESH * GATE * TRAVERSAL
                       * np.dot(collection**2, slot_energies) / (8*n*scale**2))
        photons = (spontaneous*early*late + added_photons(early,6)*raw*late
                   + added_photons(late,5))
        noise = n * photons * PHOTON_J / (TARGET * input_pj * 1e-12)
        for vector in program.vectors:
            data = (input_amp * samples * np.fft.ifft(vector * chirp, n=samples)
                    * launch[:, None])
            weak = np.sqrt(ETA) / PUMP_W * weight * reference_amp * data.conj()
            actual = np.empty_like(weak)
            for start in range(0, n, 4):
                block = slice(start, start+4)
                actual[block], residual = phase_matched_output(weight[block], data[block], reference_amp, steps)
                assert residual < 1e-5

            def collect(raw_wave):
                field = collection @ np.take_along_axis(raw_wave, shifted, axis=1)
                return np.fft.fft(field)[:n] / (samples * normalization * chirp)

            target = program.target @ vector
            np.testing.assert_allclose(collect(weak), target, atol=1e-11, rtol=1e-10)
            error = collect(actual) - target
            tally += (np.vdot(target,target).real, np.vdot(error,error).real, noise)
    target_energy, bias, noise = tally
    return 10*np.log10(late), 100 * np.sqrt(np.array([bias, noise, bias+noise])/target_energy)


def electrical(n, input_pj, ref_w, weight_w):
    """Return electrical pJ/read, with the original 64-channel current allocations.

    Extend the previous G2 driver reserves of 281.25 pJ at 2 pJ input and
    331.25 pJ at 4 pJ input linearly. The additional 25 pJ drive per pJ of
    input assumes ~5% incremental conversion to useful output under load.
    """
    interval = BASE_INTERVAL_NS + (n-64)*.045
    ref = BASE_BUDGET['reference'] * BASE_INTERVAL_NS * (n/64) * ref_w / PUMP_W
    # Fixed slot pitch; N and weight frame duration both grow, keeping the
    # *average* weight-loop replenishment power per mW of pump setting fixed.
    loop = BASE_BUDGET['weight loop'] * interval * weight_w / PUMP_W
    continuous = interval * (BASE_BUDGET['raw idler gain'] + BASE_BUDGET['clock, holding and trims'])
    gate = BASE_INTERVAL_NS * BASE_BUDGET['output gate']
    booster = BASE_INTERVAL_NS*BASE_BUDGET['output gain'] + 25*(input_pj-4)
    return (ref + loop + continuous + gate + booster + 10) / DC_EFFICIENCY


def main():
    banks = {n: workload(n) for n in (64,128)}
    peaks = {n: max(np.linalg.norm(p.weights, axis=1).max() for p in programs)
             for n, (programs, _) in banks.items()}
    scale = max(peaks.values())
    print(f'Peak slot norms {peaks}, shared programmed normalization {scale:.6f}.', flush=True)
    # Pump ratios below apply to *both* pump powers. sqrt(2) offsets the
    # inherent 3 dB loss at 128; additional path loss d costs 10**(d/20)
    # in each pump to retain the original final gain. The 0.432 m added path
    # gives approximately 1.04, 3.45 or 4.32 dB at 2.4, 8 or 10 dB/m.
    cases = [(64, 0., 4, 1.), (128, 0., 4, 1.), (128, 0., 4, np.sqrt(2)),
             (128, 0., 8, np.sqrt(2)), (128, 0., 4, 2.)]
    cases.extend((128, loss_db, 8, np.sqrt(2) * 10**(loss_db/20))
                 for loss_db in (1.04, 3.45, 4.32))
    # Brighter-packet sensitivities. The second also needs a higher loaded G2.
    base_mix = ETA * COUPLING**2 * BASE_REF * BASE_WEIGHT / PUMP_W**2
    base_signal = weak_signal_transfer(64, scale, base_mix, 1.)
    base_gain = 10*np.log10(TARGET / (10 * base_signal))
    for loss_db, input_pj, gain_db in ((3.45, 16, 30), (4.32, 20, 31)):
        multiplier = np.sqrt(2) * 10**((loss_db + base_gain - gain_db)/20)
        cases.append((128, loss_db, input_pj, multiplier))
    print('N extra loss(dB) input(pJ) reference/weight(mW) G2(dB) '
          'deterministic/noise/total RMS(%) pJ/read fJ/MAC')
    for n, loss_db, input_pj, multiplier in cases:
        programs, chirp = banks[n]
        ref_w, weight_w = BASE_REF*multiplier, BASE_WEIGHT*multiplier
        args = (chirp, scale, input_pj, ref_w, weight_w)
        late, rms = errors(programs, *args, samples=8*n, steps=32, extra_delay_loss_db=loss_db)
        _, coarse = errors(programs[:1], *args, samples=8*n, steps=32, extra_delay_loss_db=loss_db)
        _, fine = errors(programs[:1], *args, samples=16*n, steps=64, extra_delay_loss_db=loss_db)
        np.testing.assert_allclose(coarse, fine, rtol=.001, atol=.002)
        energy = electrical(n, input_pj, ref_w, weight_w)
        print(f'{n} {loss_db:.2f} {input_pj} {1000*ref_w:.2f}/{1000*weight_w:.2f} {late:.2f} '
              f'{rms.round(2)} {energy:.1f} {1000*energy/n**2:.2f}', flush=True)
        if n == 64:
            np.testing.assert_allclose(rms, [4.82,9.66,10.79], atol=.01)
            np.testing.assert_allclose(1000*energy/n**2, 198.86, atol=.02)


if __name__ == '__main__':
    main()
