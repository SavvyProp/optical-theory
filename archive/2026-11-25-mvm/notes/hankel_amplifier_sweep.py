"""Optimize directional read amplification for the 64-slot shared interleaver.

Run python3 notes/hankel_amplifier_sweep.py. Powers are linear internally.
The covariance model is phase insensitive and applies to the weak, linear MVM.
The separate hankel_fwm_stress.py tests that approximation in the time domain.
"""

import numpy as np
from scipy.constants import c, h

from hankel_comb_codes import N, SLOTS, code, masked_weights, passive_taps


ETA = 0.468
INPUT_PHOTONS = 2e-12 / (h*c / 1.55e-6) / N
MESH = 10**(-2.2 / 10)  # Excess loss per traversal, beyond ideal port splitting
REJECTION = 10**(-0.5 / 10)
GATE = 10**(-0.5 / 10)
TARGET = 10**0.1  # +1 dB useful power, relative to the programmed unitary MVM
SITES = ('launch', 'replicas', 'raw idler', 'after mesh', 'after gate')


def db(power):
    return 10 * np.log10(power)


def added_photons(gain, noise_figure_db):
    # A bypass has neither optical gain nor ASE. Use a constant spontaneous
    # emission factor calibrated to the specified high-gain noise figure.
    return 0.5 * 10**(noise_figure_db / 10) * (gain - 1)


class Read:
    def __init__(self, matrix, pump_scale, adjoint=False):
        taps = passive_taps()
        masks = code(-(N - 1 - SLOTS))
        program = masked_weights(matrix.T, masks.conj()) / pump_scale
        self.signal = ETA / (16*N*pump_scale**2) * MESH**2 * REJECTION * GATE
        self.after_mesh = (1 if adjoint else 0.5) * GATE
        # For continuous white raw ASE, both B copier taps admit independent
        # T-long noise windows. Only one cyclic packet of signal is retained.
        self.raw = np.sum(abs(taps)**2) * MESH * self.after_mesh
        return_taps = taps[::-1] * (1 if adjoint else 0.5)
        self.partners = (ETA * abs(return_taps)**2 * REJECTION * MESH * GATE)
        # Sum vacuum over every input-band partner j=s-i of every pumped h[s].
        # Full-family upper bandwidth case, one polarization, equal photon energies.
        self.spontaneous = self.partners @ np.sum(abs(program)**2, axis=1)
        valid = SLOTS[:, None] + SLOTS
        self.in_band = self.partners @ np.sum(abs(program[:, valid])**2, axis=(1, 2)) / N

    def noise(self, site, early_db, *, broad_nf=6, final_nf=5):
        early = 10**(early_db / 10)
        late = TARGET / (self.signal * early)
        early_nf = broad_nf if site in ('raw idler', 'after mesh') else final_nf
        transfer = {'launch': self.signal, 'replicas': self.in_band,
                    'raw idler': self.raw, 'after mesh': self.after_mesh,
                    'after gate': 1}[site]
        # Launch assumes a cleanup gate: the noise on that packet is copied
        # coherently with x. The replica booster adds independent slot noise.
        post_fwm_gain = 1 if site in ('launch', 'replicas') else early
        terms = np.array([self.spontaneous * post_fwm_gain * late,
                          added_photons(early, early_nf) * transfer * late,
                          added_photons(late, final_nf)])
        rms = np.sqrt(np.sum(terms) / (INPUT_PHOTONS * TARGET))
        return db(late), 100 * rms, terms


def noise_crosscheck(read, adjoint):
    """Sample the selected raw ASE mode via all delayed independent windows."""
    rng = np.random.default_rng(121)
    taps = passive_taps()
    # Each input window is a normalized square tooth mode. In the wideband
    # white limit the windows are disjoint (Q > 2T), hence independent.
    if not adjoint:
        taps = np.tile(taps / 2, 2)
    field_taps = taps * np.sqrt(MESH * GATE)
    total = 0.0
    samples = 0
    for _ in range(64):
        draws = rng.standard_normal((2, 512, len(taps)))
        fields = (draws[0] + 1j * draws[1]) @ field_taps / np.sqrt(2)
        total += np.vdot(fields, fields).real
        samples += len(fields)
    variance = total / samples
    print(f'raw ASE transfer: analytic {read.raw:.5f}, sampled {variance:.5f}')
    np.testing.assert_allclose(variance, read.raw, rtol=0.025)


def main():
    identity = np.eye(N)
    forward, backward = (Read(identity, 1, flag) for flag in (False, True))
    for mode, read in [('forward', forward), ('adjoint', backward)]:
        noise_crosscheck(read, mode == 'adjoint')
        print(f'{mode}: passive {db(read.signal):.3f} dB; accepted-64 / full vacuum: '
              f'{read.in_band:.5f} / {read.spontaneous:.5f} photons before amplification')
    # One early stage plus a gated output amplifier. Sweep to 10 dB flat
    # broadband gain on raw traffic; same upper bound for the input comparison.
    # The after-gate comparator uses two conventional amplifiers at one location.
    print('site                 first dB  final dB  forward RMS %  adjoint RMS %')
    for site in SITES:
        sweep = []
        for early_db in np.arange(0, 10.001, 0.25):
            final_db, fe, _ = forward.noise(site, early_db)
            _, be, _ = backward.noise(site, early_db)
            if final_db <= 35:
                sweep.append((max(fe, be), early_db, final_db, fe, be))
        # Equal-NF adjacent output amplifiers have a degenerate gain split.
        _, early_db, final_db, fe, be = min(sweep, key=lambda row: (round(row[0], 10), row[1]))
        print(f'{site:20} {early_db:8.2f} {final_db:9.3f} {fe:14.3f} {be:14.3f}')
    for broad_gain in (5, 8, 10, 15, 20):
        print(f'raw gain {broad_gain:d} dB, forward / adjoint RMS %:',
              *(round(r.noise('raw idler', broad_gain)[1], 3) for r in (forward, backward)))
    for broad_nf in (4, 6, 8):
        print(f'raw NF {broad_nf:d} dB, forward / adjoint RMS %:',
              *(round(r.noise('raw idler', 10, broad_nf=broad_nf)[1], 3) for r in (forward, backward)))

    # Freeze one pump normalization over the entire bank; re-normalizing after
    # each optical delta write would change the logical memory and learning rate.
    rng = np.random.default_rng(734)
    matrices = [identity]
    for _ in range(100):
        matrix, triangular = np.linalg.qr(rng.normal(size=(N, N)))
        matrices.append(matrix * np.sign(np.diag(triangular)))
    masks = code(-(N - 1 - SLOTS))
    scale = max(np.linalg.norm(masked_weights(w.T, masks.conj()), axis=1).max() for w in matrices)
    for adjoint in (False, True):
        results = [Read(w, scale, adjoint).noise('raw idler', 10) for w in matrices]
        name = 'adjoint' if adjoint else 'forward'
        final = results[0][0]
        errors = np.array([row[1] for row in results])
        print(f'fixed scale {scale:.5f}, {name}, final {final:.3f} dB, '
              f'RMS min/median/max % {np.percentile(errors, [0, 50, 100]).round(3)}')
    for read in (forward, backward):
        final, rms, photons = read.noise('raw idler', 10)
        print(f'selected identity: G2={final:.3f}, net={db(read.signal)+10+final:.3f} dB, '
              f'RMS={rms:.3f}%, FWM/G1/G2 noise photons per tooth {photons.round(1)}')


if __name__ == '__main__':
    main()
