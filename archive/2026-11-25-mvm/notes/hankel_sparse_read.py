"""Forward field error for 30%-sparse uniform weights and 10%-sparse uniform inputs.

Run with OPENBLAS_NUM_THREADS=1 python3 notes/hankel_sparse_read.py.
"""
import numpy as np

from hankel_amplifier_sweep import Read
from hankel_fwm_stress import (
    ETA, INPUT_LOSS_DB, N, OUTPUT_LOSS_DB, PACKET_POWER_W, PUMP_W, SLOTS,
    field_gain, optical_program, read,
)


def sparse_workload():
    rng = np.random.default_rng(20261123)
    chirp = np.exp(1j * np.pi * SLOTS**2 / (2 * N))
    programs = []
    for _ in range(16):
        matrix = rng.uniform(-1, 1, (N, N))
        matrix[rng.random(matrix.shape) < 0.30] = 0
        matrix *= np.sqrt(N) / np.linalg.norm(matrix)
        inputs = rng.uniform(-1, 1, (4, N))
        inputs[rng.random(inputs.shape) < 0.10] = 0
        inputs /= np.linalg.norm(inputs, axis=1, keepdims=True)
        programs.append((matrix, chirp[:, None] * matrix * chirp, inputs))
    return programs, chirp


def read_error_powers(matrix, encoded_matrix, inputs, chirp, bank_scale, energy_pj, samples, steps,
                      *, reference_power_ratio=1.0):
    weight, modes, peak = optical_program(encoded_matrix, samples)
    # Reciprocal changes in the two pump powers preserve their weak FWM product.
    weight *= peak / (bank_scale * np.sqrt(reference_power_ratio))
    weak_gain = np.sqrt(ETA / N) * field_gain(-INPUT_LOSS_DB - OUTPUT_LOSS_DB) / (4 * bank_scale)
    seed_scale = np.sqrt(energy_pj / (20 * PACKET_POWER_W))  # 20 ps input packet.
    decoder = samples * np.sqrt(PACKET_POWER_W) * seed_scale * weak_gain * chirp
    final_gain_db, noise_percent, _ = Read(encoded_matrix, bank_scale).noise('raw idler', 10)

    # For unit-norm x, Read.noise uses a unit-norm output as its reference.
    # Track absolute error powers and normalize by the observed sum of ||Wx||².
    signal_power = complex_error_power = real_error_power = 0.0
    for vector in inputs:
        target = matrix @ vector
        weak, depleted, *_ = read(weight, modes, seed_scale * chirp.conj() * vector, 0, steps,
                                 reference_power_w=PUMP_W * reference_power_ratio)
        np.testing.assert_allclose(np.fft.fft(weak)[:N] / decoder, target, rtol=1e-11, atol=1e-12)
        error = np.fft.fft(depleted)[:N] / decoder - target
        signal_power += np.vdot(target, target).real
        complex_error_power += np.vdot(error, error).real
        real_error_power += error.real @ error.real
    # At fixed pump powers and gains, additive weak-model photon counts are constant.
    noise_power = len(inputs) * (noise_percent / (100 * seed_scale))**2
    return np.array([signal_power, complex_error_power, real_error_power, noise_power]), final_gain_db


def main():
    programs, chirp = sparse_workload()
    bank_scale = max(optical_program(encoded, N)[2] for _, encoded, _ in programs)
    weight_energy_pj = PUMP_W * N * 20 / bank_scale**2  # W × ps gives pJ, ||W||F²=N.
    print(f'64 reads per energy, fixed bank scale {bank_scale:.6f}')
    print(f'Weight delivery {weight_energy_pj:.2f} pJ/read ({weight_energy_pj/2.88:.2f} mW average)')
    norms_sq = np.concatenate([np.sum((inputs @ matrix.T)**2, axis=1)
                               for matrix, _, inputs in programs])
    print(f'Target ||Wx||² range {norms_sq.min():.4f} to {norms_sq.max():.4f}')
    print('Fixed analytic gain, pooled RMS relative to ideal Wx (%).')
    print('Input (pJ)   G2 (dB)   Output at ||Wx||=1 (pJ)   Complex (bias/noise/total)    Real (bias/noise/total)')
    for energy_pj in (2, 3, 3.5, 4, 5):
        total = np.zeros(4)
        for matrix, encoded, inputs in programs:
            powers, final_gain_db = read_error_powers(
                matrix, encoded, inputs, chirp, bank_scale, energy_pj, 1024, 32)
            total += powers

        # Check one read at doubled time resolution and propagation steps for each energy.
        matrix, encoded, inputs = programs[0]
        args = matrix, encoded, inputs[:1], chirp, bank_scale, energy_pj
        coarse, _ = read_error_powers(*args, 1024, 32)
        fine, _ = read_error_powers(*args, 2048, 64)
        np.testing.assert_allclose(coarse, fine, rtol=2e-5, atol=1e-12)

        signal, complex_error, real_error, noise = total
        components = np.array([complex_error, noise, complex_error + noise,
                               real_error, noise / 2, real_error + noise / 2])
        errors = 100 * np.sqrt(components / signal)
        output_pj = energy_pj * 10**0.1  # +1 dB nominal energy gain at unit target norm.
        print(f'{energy_pj:10} {final_gain_db:9.3f} {output_pj:25.3f}'
              + ''.join(f'{error:8.2f}' for error in errors))


if __name__ == '__main__':
    main()
