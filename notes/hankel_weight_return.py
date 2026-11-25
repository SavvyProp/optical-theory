"""Coherently return the surviving weight tap in one 64 × 64 cell.

Run with OPENBLAS_NUM_THREADS=1 python3 notes/hankel_weight_return.py.
Each of two full-rank matrices is read 128 times with independent real queries.
The return-only phase screen is a sensitivity test. The FWM solver omits Kerr
phase shifts, competing idlers, dispersion and amplifier noise. The through arm
has lumped propagation loss. The retained 127 teeth have an ideal flat filter.
"""

import numpy as np

from hankel_fwm_stress import (
    INPUT_LOSS_DB, N, PUMP_W, SLOTS, TAPS, field_gain, optical_program, phase_matched_output,
)
from hankel_power_budget import DC_EFFICIENCY, DENSE_BANK_SCALE, FRAME_NS, budget

TAP = 0.25
DELIVERY = 10 ** (-0.5 / 10)       # From tap to guide or from guide to combiner
GUIDE_LENGTH_M = 0.0212
GUIDE_LOSS_DB = 0.8 * 100 * GUIDE_LENGTH_M
MAIN_POWER = (1 - TAP) * 10 ** (-0.2 / 10)
THROUGH_POWER = TAP * DELIVERY**2 * 10 ** (-GUIDE_LOSS_DB / 10)
ADDED_LIGHT_EFFICIENCY = 0.121
INTERVAL_NS = 3.16
CHIRP = np.exp(1j * np.pi * SLOTS**2 / (2 * N))
READ_CODE = np.exp(-2j * np.pi * np.outer(N - 1 - SLOTS, SLOTS) / N) / np.sqrt(N)
TOOTH_PAIRS = SLOTS[:, None] + SLOTS
LOSS_PER_M = 0.8 * 100 * np.log(10) / 10
GAMMA_LEFF = 437 * (-np.expm1(-LOSS_PER_M * GUIDE_LENGTH_M)) / LOSS_PER_M


def program_matrix(matrix, samples, pump_ratio):
    physical = CHIRP[:, None] * matrix * CHIRP
    weight, modes, peak = optical_program(physical, samples)
    assert peak <= DENSE_BANK_SCALE
    weight *= peak / (DENSE_BANK_SCALE * np.sqrt(pump_ratio))
    return weight, modes


def decode(weight, pump_ratio):
    coefficients = np.fft.fft(weight, axis=1) / weight.shape[1]
    coefficients *= DENSE_BANK_SCALE * np.sqrt(pump_ratio / PUMP_W)
    physical = np.einsum('mr,mrj->rj', READ_CODE, coefficients[:, TOOTH_PAIRS])
    return CHIRP.conj()[:, None] * physical * CHIRP.conj()


def propagate_tap(weight, modes, query, pump_ratio, steps, phase_strength):
    # 4 pJ launched in 20 ps; the parent read model sends one small replica into each slot.
    data = TAPS * field_gain(-INPUT_LOSS_DB) * np.sqrt(0.2) * ((CHIRP * query) @ modes)
    through = np.empty_like(weight)
    idler_energy = 0.0
    for start in range(0, N, 4):
        block = slice(start, start + 4)
        idler, invariant_error, through[block] = phase_matched_output(
            weight[block], data, np.sqrt(PUMP_W * pump_ratio), steps, return_weight=True)
        assert invariant_error < 1e-5
        idler_energy += np.vdot(idler, idler).real
    if phase_strength:
        # Approximate SPM and data XPM on the return only. The static CW XPM
        # phase can be absorbed by one phase trim. No per-tooth compensation.
        through *= np.exp(1j * phase_strength * (abs(weight)**2 + 2 * abs(data)**2))
    depletion = idler_energy / np.vdot(weight, weight).real
    return through, depletion


def recombine(weight, through, trim):
    # Fixed lossless coupler matched to the two undepleted arm powers. The
    # orthogonal, dark output and all Fourier components outside the weight
    # band are discarded. Keep absolute amplitude for the amplifier calculation.
    mixed = (MAIN_POWER * weight + THROUGH_POWER * trim * through) / np.sqrt(
        MAIN_POWER + THROUGH_POWER)
    band = np.fft.fft(mixed, axis=1)
    band[:, 2 * N - 1:] = 0
    return np.fft.ifft(band, axis=1)


def run(matrix, queries, pump_ratio=1.0, *, corrected=False, phase_strength=0.0,
        samples=512, steps=16):
    weight, modes = program_matrix(matrix, samples, pump_ratio)
    original_norm = np.linalg.norm(weight)
    np.testing.assert_allclose(decode(weight, pump_ratio), matrix, atol=1e-13)
    through, _ = propagate_tap(weight, modes, queries[0], pump_ratio, steps, phase_strength)
    # Use an independent calibration query to set one phase and one field gain.
    trim = np.exp(-1j * np.angle(np.vdot(weight, through)))
    fixed_gain = original_norm / np.linalg.norm(recombine(weight, through, trim))
    guide_pj = 20 * original_norm**2 / samples
    circulating_pj = guide_pj / (TAP * DELIVERY)
    added_pj = 0.0
    checkpoints = {}
    for n, query in enumerate(queries[1:], 1):
        through, depletion = propagate_tap(weight, modes, query, pump_ratio, steps, phase_strength)
        returned = recombine(weight, through, trim)
        norm = np.linalg.norm(returned)
        gain = original_norm / norm if corrected else fixed_gain
        added_pj += circulating_pj * (gain**2 - 1) * (norm / original_norm)**2
        weight = gain * returned
        if n in (1, 32, 128):
            decoded = decode(weight, pump_ratio)
            drift = np.linalg.norm(decoded - matrix) / np.linalg.norm(matrix)
            fit = np.vdot(decoded, matrix) / np.vdot(decoded, decoded)
            shape_drift = np.linalg.norm(fit * decoded - matrix) / np.linalg.norm(matrix)
            checkpoints[n] = (drift, shape_drift, depletion, added_pj / n)
    return checkpoints


def energy(mean_added_pj, pump_ratio, buildup):
    """Whole-cell fJ/MAC counting loss replacement each 2.88 ns as in original budget."""
    components, *_ = budget(INTERVAL_NS, gated_booster=True, extra_booster_pj=50)
    components['reference'] *= pump_ratio / buildup
    components['weight loop'] = mean_added_pj / (FRAME_NS * ADDED_LIGHT_EFFICIENCY)
    return sum(components.values()) * INTERVAL_NS * 1000 / (DC_EFFICIENCY * N**2)


def main():
    rng = np.random.default_rng(937)
    matrices = rng.normal(size=(2, N, N))
    matrices *= np.sqrt(N) / np.linalg.norm(matrices, axis=(1, 2))[:, None, None]
    assert all(np.linalg.matrix_rank(w) == N for w in matrices)
    queries = np.random.default_rng(1020).uniform(-1, 1, size=(2, 129, N))
    queries /= np.linalg.norm(queries, axis=2, keepdims=True)
    # Buildup reduces external reference power; loading remains set by the
    # separate, internal pump ratio. The 1.70 dB guide loss caps a
    # full-guide, single-pass-per-round-trip cavity at ~3.1x before other
    # losses. Twofold buildup and its control overhead need a device budget.
    print('case, matrix, pump ratio, buildup, read count, matrix drift %, '
          'drift after fitting one complex scalar %, weight depletion %, '
          'mean added light pJ/lap, projected cell fJ/MAC')
    for case, ratio, buildup, corrected, phase in [
        ('fixed gain', 1, 1, False, 0),
        ('norm servo', 1, 1, True, 0),
        ('return phase screen', 1, 1, False, GAMMA_LEFF),
        ('reference buildup', 1.25, 2, False, 0),
    ]:
        for m, (matrix, inputs) in enumerate(zip(matrices, queries)):
            checkpoints = run(matrix, inputs, ratio, corrected=corrected, phase_strength=phase)
            for n, (drift, shape, loss, added) in checkpoints.items():
                print(f'{case}, {m}, {ratio}, {buildup}, {n}, {100*drift:.4f}, '
                      f'{100*shape:.4f}, {100*loss:.3f}, '
                      f'{added:.3f}, {energy(added, ratio, buildup):.2f}', flush=True)
            if m == 0:
                fine = run(matrix, inputs[:33], ratio, corrected=corrected, phase_strength=phase,
                           samples=1024, steps=32)
                for n in fine:
                    np.testing.assert_allclose(checkpoints[n], fine[n], rtol=.003, atol=2e-6)


if __name__ == '__main__':
    main()
