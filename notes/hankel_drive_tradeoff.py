"""Drive and gain trade-offs for the canonical 64-tooth Hankel cell.

Run from the repo root with OPENBLAS_NUM_THREADS=1 python3 -B notes/hankel_drive_tradeoff.py.
FWM and downstream losses use the shared interleaver stress model. A single
gain is calibrated on the one-tooth identity read and held for every input.
"""
import numpy as np

from hankel_amplifier_sweep import INPUT_PHOTONS, TARGET, Read, added_photons
from hankel_fwm_stress import (
    INPUT_LOSS_DB, N, OUTPUT_LOSS_DB, SLOTS, TAPS, cases, field_gain,
    optical_program, phase_matched_output, projected_gain,
)


def propagate(weight, modes, vector, energy_pj, reference_w, length_ratio, steps):
    samples = modes.shape[1]
    data = np.sqrt(energy_pj / 20) * TAPS * field_gain(-INPUT_LOSS_DB) * (vector.conj() @ modes)
    # FWM's derivative is cubic in its fields. Scaling all four field amplitudes
    # by sqrt(L/L0) simulates a change in effective length with the same solver.
    scale = np.sqrt(length_ratio)
    raw = np.empty_like(weight)
    for start in range(0, N, 8):
        sl = slice(start, start + 8)
        raw[sl], error = phase_matched_output(
            scale * weight[sl], scale * data, scale * np.sqrt(reference_w), steps)
        assert error < 5e-5
    offsets = (N-1-SLOTS)[:, None] * (samples // N)
    summed = np.take_along_axis(raw, (np.arange(samples) - offsets) % samples, axis=1).sum(axis=0)
    return summed / scale * TAPS / 2 * field_gain(-OUTPUT_LOSS_DB)


def drive_test(programs, energy_pj, reference_mw, weight_mw, length_ratio, steps):
    actual, targets = [], []
    for matrix, vector, weights, modes, normalization in programs:
        actual.append(propagate(weights*np.sqrt(weight_mw/50), modes, vector,
                                energy_pj, reference_mw/1000, length_ratio, steps))
        # The dense program is divided by its precomputed maximum slot norm.
        # Express each target in that physical normalization for the common gain test.
        targets.append(np.sqrt(energy_pj/20) * ((matrix @ vector) @ modes) / normalization)
    calibration = projected_gain(actual[0], targets[0])
    total_gain_db = 1 - 20*np.log10(abs(calibration))
    errors = np.array([np.linalg.norm(y-calibration*t) / np.linalg.norm(calibration*t)
                       for y, t in zip(actual, targets)])
    # Fit individual gains only as a diagnostic for shape versus scale error.
    fitted = []
    for y, t in zip(actual, targets):
        fit = projected_gain(y, t)
        fitted.append(np.linalg.norm(y - fit*t) / np.linalg.norm(fit*t))
    # Lower bound for additive noise at the one-tooth calibration: two ideal
    # flat unsaturated gain stages, omit spontaneous FWM and other noise.
    first = 10
    final = 10**((total_gain_db - 10)/10)
    receiver = Read(np.eye(N), 1)
    photons = (added_photons(first, 6)*receiver.raw*final + added_photons(final, 5))
    ase = np.sqrt(photons / (INPUT_PHOTONS*(energy_pj/2)*TARGET))
    return np.r_[total_gain_db, 100*ase, 100*errors, 100*np.array(fitted)]


def run(samples=2048, steps=64):
    programs = [(m, v, *optical_program(m, samples)) for _, m, v in cases()]
    drives = [
        ('nominal', 2, 50, 50, 1),
        ('data only / 100', 0.02, 50, 50, 1),
        ('0.4 length', 2, 50, 50, 0.4),
        ('0.23 length', 2, 50, 50, 0.23),
        ('0.16 length', 2, 50, 50, 0.16),
        ('0.1 length', 2, 50, 50, 0.1),
        ('weight only / 50', 2, 50, 1, 1),
        ('pump split 95/5', 2, 95, 5, 1),
    ]
    rows = {}
    for name, *drive in drives:
        rows[name] = drive_test(programs, *drive, steps)
    return rows


if __name__ == '__main__':
    print('Columns within each error group: identity one tooth / equal phases / random signed; dense orthogonal random signed.')
    print('drive                 dB to +1  ASE lower %    fixed one-tooth calibration errors %    individually fitted waveform errors %')
    coarse, fine = run(1024, 32), run(2048, 64)
    for name, row in fine.items():
        print(f'{name:<21} {row[0]:8.2f} {row[1]:12.1f}   {np.round(row[2:6],1)}    {np.round(row[6:],1)}')
        # The baseline aligned peaks require finer sampling; the established
        # stress script resolves those separately to 8192 samples per period.
        if name != 'nominal' and name != 'data only / 100':
            np.testing.assert_allclose(row, coarse[name], rtol=0.015, atol=0.2)
