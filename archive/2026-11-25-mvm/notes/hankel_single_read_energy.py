"""Electrical energy and field accuracy of one fully dense, real 64 x 64 read.

Run OPENBLAS_NUM_THREADS=1 python3 notes/hankel_single_read_energy.py.
All scenarios use one 4 pJ vector, one stored matrix, and one restored output.
The source and converter changes below are design targets, not measured devices.
"""
import numpy as np

from hankel_fwm_stress import N, optical_program
from hankel_power_budget import DC_EFFICIENCY, budget
from hankel_sparse_read import read_error_powers


def workload():
    rng = np.random.default_rng(12023)
    phase = np.exp(1j * np.pi * np.arange(N)**2 / (2 * N))
    bank = []
    for _ in range(16):
        matrix = rng.uniform(-1, 1, (N, N))
        matrix *= np.sqrt(N) / np.linalg.norm(matrix)
        assert np.linalg.matrix_rank(matrix) == N
        vectors = rng.uniform(-1, 1, (4, N))
        vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
        bank.append((matrix, phase[:, None] * matrix * phase, vectors))
    return bank, phase


def error_powers(case, phase, scale, pump_ratio, interaction, samples, steps):
    # At fixed small-signal transfer, multiplying the effective coupling by m
    # and dividing both pump powers by m gives the original four-wave equations
    # in fields scaled by sqrt(m). The solver therefore sees m times the seed
    # energy. Physical seed energy stays at 4 pJ, so restore the weak-model
    # noise variance computed for 4 pJ when combining it with depletion error.
    values, _ = read_error_powers(*case, phase, scale, 4 * interaction, samples, steps,
                                  reference_power_ratio=pump_ratio)
    values[-1] *= interaction
    return values


def waveguide_field_coefficient(length_cm, loss_db_per_cm):
    # Phase-matched, equal distributed loss at all four frequencies. Overall
    # amplitude conversion includes effective length and output attenuation.
    attenuation = loss_db_per_cm * np.log(10) / 10
    return np.exp(-attenuation * length_cm / 2) * (-np.expm1(-attenuation * length_cm)) / attenuation


def main():
    bank, phase = workload()
    scale = max(optical_program(encoded, N)[2] for _, encoded, _ in bank)
    long_guide = waveguide_field_coefficient(4, 0.2) / waveguide_field_coefficient(2.12, 0.8)
    added_flight_ns = 0.0188 * 4.04 / 299792458 * 1e9
    # Keep the original, slightly more conservative electrical loop allocation
    # in every row so different test matrices receive no artificial energy credit.
    settings = dict(gated_booster=True, extra_booster_pj=50)
    print(f'16 dense, full-rank matrices; 64 single-vector reads. Fixed bank scale {scale:.5f}.')
    for name, pulsed, buildup, interaction, rebalance, interval_ns in (
            ('Existing cell', False, 1, 1, False, 3.16),
            ('Pulsed source, same in-guide powers', True, 1, 1, False, 3.16),
            ('Pulsed source, rebalanced pumps', True, 1, 1, True, 3.16),
            ('CW, twofold reference buildup', False, 2, 1, True, 3.16),
            ('Pulsed, 4 cm traveling guide at 0.2 dB/cm', True, 1, long_guide, True,
             3.16 + added_flight_ns)):
        power, _, _, _ = budget(interval_ns, **settings, pulsed_reference=pulsed)
        power['reference'] /= buildup
        ratio = np.sqrt(power['weight loop'] / power['reference']) if rebalance else 1.0
        power['reference'] *= ratio / interaction
        power['weight loop'] /= ratio * interaction
        energy_pj = sum(power.values()) * interval_ns / DC_EFFICIENCY
        totals = np.zeros(4)
        for index, case in enumerate(bank):
            args = case, phase, scale, ratio, interaction
            result = error_powers(*args, 1024, 32)
            if index == 0:
                np.testing.assert_allclose(result, error_powers(*args, 2048, 64),
                                           rtol=2e-5, atol=1e-12)
            totals += result
        signal, err, _, noise = totals
        rms = 100 * np.sqrt((err + noise) / signal)
        print(f'{name}: interval {interval_ns:.3f} ns, effective coupling {interaction:.3f}x,'
              f' guide reference {50 * ratio / interaction:.1f} mW,'
              f' nominal weight {50 / (ratio * interaction):.1f} mW,'
              f' {energy_pj:.1f} pJ/read, {1000 * energy_pj / N**2:.1f} fJ/MAC,'
              f' pooled complex-field RMS {rms:.2f}%')


if __name__ == '__main__':
    main()
