"""Spectral-gain stress test for amplification of the shared cell's raw idlers.

Run python3 notes/hankel_idler_bandwidth.py [--size 128 --selector --samples 4096]. The gate
compares full complex waveforms. The source is one isolated 2 pJ packet,
the occupied-window mean is 100 mW, each pump is capped at 50 mW, and FWM
uses the repo's flat small-signal efficiency 0.468. Power peaks predicted
by this linear model can exceed the available pump photon flux.

The idler amplifier has zero phase ripple, ideal linear gain, and unity
field transmission outside its gain band. No brick-wall idler filter, ASE,
compression, coherent pump leakage or restorative final gain is included.
"""

import argparse

import numpy as np
from scipy.fft import fft, fftfreq, ifft, next_fast_len

from hankel_comb_codes import code, masked_weights, passive_taps


SPACING_THZ = 0.050
PEAK_PACKET_MEAN_MW = 100.0
CONVERSION = 0.468
INPUT_LOSS_DB = 2.2
PUMP_REJECTION_INSERTION_DB = 0.5
RAW_GAIN_DB = 10.0
TRANSITION_THZ = 0.25  # On each edge, in addition to the specified flat width.


def amplifier(frequency, width, ripple_pp_db, n=64):
    """Field transfer. Ripple is sinusoidal in dB with a 0.70 THz period."""
    offset = frequency - (n - 1) * SPACING_THZ / 2
    if width is None:
        return np.full(frequency.shape, 10**(RAW_GAIN_DB / 20))
    edge = np.clip((np.abs(offset) - width / 2) / TRANSITION_THZ, 0, 1)
    weight = (1 + np.cos(np.pi * edge)) / 2
    ripple = ripple_pp_db / 2 * np.sin(2 * np.pi * offset / 0.70)
    return 1 + weight * (10**((RAW_GAIN_DB + ripple) / 20) - 1)


def run(samples, n=64, selector=False):
    if samples < 1024 or samples % n:
        raise ValueError('Use at least 1024 samples per 20 ps, in multiples of the matrix size.')
    pitch = 9 * samples // 4   # Every FWM packet starts on the 45 ps grid.
    extra = samples // n
    guard = 2 * samples if selector else 0
    window = np.arange(samples)
    time = (window + 0.5) / samples
    basis = np.exp(2j * np.pi * np.arange(2 * n - 1)[:, None] * time)
    indices = np.arange(n)
    lower = indices >= n // 2
    starts = indices * pitch + lower * guard
    slots = starts[:, None] + window
    taps = passive_taps(n) * (np.sqrt(2) if selector else 1)
    signs = np.sign(taps * taps[::-1])
    masks = code(-(n - 1 - indices), n)
    scale = np.sqrt(PEAK_PACKET_MEAN_MW * CONVERSION *
                    10**(-(INPUT_LOSS_DB + PUMP_REJECTION_INSERTION_DB) / 10))
    # All signal fields downstream of the pump-rejection filter are in sqrt(mW).
    # The impulse response of the passive collector is one tap per path,
    # duplicated at a one-comb-period offset by the external B interferometer.
    delays = indices * (pitch + extra) + lower * guard
    gate = starts[-1] + samples + window
    times = gate - delays[:, None, None] - np.array([0, samples])[:, None]
    if selector:
        # Route the early half of the raw train into the long return paths.
        b_starts = starts - (n - 1 - indices) * extra
        switch = (b_starts[lower][0] + b_starts[~lower][-1] + 2 * samples) // 2
        accepted = (times < switch) == lower[:, None, None]
    else:
        accepted = 1
    raw_length = slots[-1, -1] + 1
    length = next_fast_len(raw_length + delays[-1] + samples + 1 + 128 * samples)
    frequencies = fftfreq(length, d=20 / samples)  # cycles/ps = THz
    profiles = [('flat', None, 0), ('4 THz', 4, 0), ('10 THz', 10, 0),
                ('12 THz', 12, 0), ('17 THz', 17, 0), ('4 THz, 0.5 dB p-p', 4, 0.5)]
    if n == 128:
        profiles = [('flat', None, 0), ('8 THz', 8, 0), ('12 THz', 12, 0),
                    ('17 THz', 17, 0), ('20 THz', 20, 0)]
    transfers = [amplifier(frequencies, width, ripple, n) for _, width, ripple in profiles]
    rng = np.random.default_rng(527)
    edge = np.eye(n)[0]
    aligned = np.ones(n) / np.sqrt(n)
    phases = np.exp(2j * np.pi * rng.random(n)) / np.sqrt(n)
    orthogonal, _ = np.linalg.qr(rng.normal(size=(n, n)))
    cases = [('I / edge tooth', np.eye(n), edge),
             ('I / aligned teeth', np.eye(n), aligned),
             ('I / random phases', np.eye(n), phases),
             ('dense / aligned teeth', orthogonal, aligned),
             ('dense / random phases', orthogonal, phases)]
    if selector:
        # First programmed matrix and input of the normalized dense-read workload.
        workload_rng = np.random.default_rng(12023)
        matrix = workload_rng.uniform(-1, 1, (n, n))
        matrix *= np.sqrt(n) / np.linalg.norm(matrix)
        vector = workload_rng.uniform(-1, 1, n)
        chirp = np.exp(1j * np.pi * indices**2 / (2 * n))
        cases.append(('dense / coded read', chirp[:, None]*matrix*chirp,
                      chirp.conj()*vector/np.linalg.norm(vector)))

    print(f'{samples} samples/T, 10 dB at the raw idler port. Amplifier power peaks are conditional on linear FWM.')
    print('input                   gain profile         field error (%)  gain error (dB)  amplifier output peak (mW)  nominal tooth RMS (%)')
    for name, target_matrix, vector in cases:
        weights = masked_weights(target_matrix.T, masks.conj())
        peak_weight_norm = np.linalg.norm(weights, axis=1).max()
        source = vector.conj() @ basis[:n]
        data_slots = taps[:, None] * source
        raw = np.zeros(raw_length, dtype=complex)
        raw[slots] = (scale / peak_weight_norm * signs[:, None] *
                      (weights @ basis) * data_slots.conj())
        raw_fft = fft(raw, length)
        target = ((target_matrix @ vector) @ basis[:n] * scale / peak_weight_norm *
                  taps[0]**2 * np.sqrt(n) / 2 * 10**(RAW_GAIN_DB / 20))
        for (profile, width, _), transfer in zip(profiles, transfers, strict=True):
            boosted = ifft(raw_fft * transfer)
            result = (taps[:, None, None] * accepted * boosted[times]).sum(axis=(0, 1)) / 2
            fit = np.vdot(target, result) / np.vdot(target, target)
            distortion = np.linalg.norm(result / fit - target) / np.linalg.norm(target)
            # The nonlinear read's accuracy metric keeps the programmed gain and
            # projects onto the useful teeth. The waveform metric fits a scalar.
            tooth_rms = np.linalg.norm(fft(result - target)[:n]) / (np.sqrt(samples)*np.linalg.norm(target))
            if width is None:
                np.testing.assert_allclose(result, target, rtol=1e-10, atol=1e-11)
            peak = np.max(np.abs(boosted)**2)
            print(f'{name:22}  {profile:18}  {100 * distortion:15.3f}  '
                  f'{20 * np.log10(abs(fit)):15.3f}  {peak:26.3f}  {100*tooth_rms:20.3f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--samples', type=int, default=2048, help='Samples per 20 ps comb period')
    parser.add_argument('--size', type=int, choices=(64,128), default=64)
    parser.add_argument('--selector', action='store_true', help='use the timed terminal port and 40 ps midframe guard')
    args = parser.parse_args()
    run(args.samples, args.size, args.selector)
