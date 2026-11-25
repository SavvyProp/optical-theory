"""Check the adjoint gate with saved x, old h, R present and residual e = 0.

Run OPENBLAS_NUM_THREADS=1 python3 notes/hankel_simultaneous_read.py.
An ordinary adjoint read with a separate unit-norm residual defines full scale.
It traverses the B two-copy interferometer and returns through the A gate.
The zero-residual test reads the same slots, with a 2 pJ saved input on A.
Both signals are compared in the 64 intended output teeth before common linear
read amplification. This uses the noiseless four-band, common-velocity model.
"""

import numpy as np

from hankel_comb_codes import passive_taps
from hankel_fwm_stress import (
    ETA, INPUT_LOSS_DB, N, OUTPUT_LOSS_DB, PACKET_POWER_W, SLOTS, TAPS,
    field_gain, optical_program, raw_idlers, read,
)


def teeth(waveform):
    return np.fft.fft(waveform)[:N] / waveform.size


def zero_residual_spur(matrix, key, unit_residual, samples, steps):
    q = np.exp(1j * np.pi * SLOTS**2 / (2*N))
    weights, modes, scale = optical_program(q[:, None] * matrix * q, samples)
    taps = passive_taps()
    # A reaches slot m by tap m and returns to the selected gate by tap 63-m.
    np.testing.assert_allclose(abs(taps), TAPS)
    roundtrip_sign = np.sign(taps * taps[::-1])
    a = TAPS * np.sqrt(PACKET_POWER_W) * field_gain(-INPUT_LOSS_DB) * ((q * key) @ modes)
    gain_out = TAPS * field_gain(-OUTPUT_LOSS_DB)
    flat_return = gain_out * roundtrip_sign @ np.broadcast_to(a, weights.shape)
    np.testing.assert_allclose(flat_return, 0, atol=1e-15)

    # B starts dark. The repo solver returns the generated B. With SPM and
    # XPM omitted, A keeps its input phase and |A_out|² = |A_in|² + |B_out|².
    # Changes of sign in either the reference or the A tap affect B's phase
    # but leave this magnitude identity unchanged.
    _, generated_b, invariant_error = raw_idlers(weights, a, steps)
    assert invariant_error < 3e-6
    leaked_power = abs(generated_b)**2
    abs_a = abs(a)
    denom = np.sqrt(abs_a**2 + leaked_power) + abs_a
    amplitude_excess = np.divide(leaked_power, denom, out=np.zeros_like(leaked_power), where=denom > 0)
    seed_phase = np.exp(1j * np.angle(a))
    spur = flat_return + gain_out * (roundtrip_sign @ amplitude_excess) * seed_phase

    # Independently propagate the weak reference through the actual adjoint
    # route, then compare it tooth by tooth to the known weak analytical read.
    nominal, *_ = read(weights, modes, q * unit_residual, 0, steps, reverse=True)
    weak_gain = np.sqrt(ETA / N) * field_gain(-INPUT_LOSS_DB-OUTPUT_LOSS_DB) / (4*scale)
    target = np.sqrt(PACKET_POWER_W) * weak_gain * q * (matrix.T @ unit_residual)
    np.testing.assert_allclose(teeth(nominal), target, rtol=1e-10, atol=1e-13)
    return np.linalg.norm(teeth(spur)) / np.linalg.norm(teeth(nominal))


def main():
    rng = np.random.default_rng(310)
    dense, triangular = np.linalg.qr(rng.standard_normal((N, N)))
    dense *= np.sign(np.diag(triangular))
    key, residual = rng.standard_normal((2, N))
    key /= np.linalg.norm(key)
    residual /= np.linalg.norm(residual)
    q = np.exp(1j * np.pi * SLOTS**2 / (2*N))
    signed = np.diag(np.sign((q**2 * np.exp(-0.17j)).real))
    print('e = 0; spurious A-gate field as % of weak adjoint field for a unit residual')
    print('Matrix                      512 samples/32 steps   1024 samples/64 steps')
    for name, matrix in [('identity', np.eye(N)), ('dense orthogonal', dense),
                         ('signed diagonal', signed)]:
        np.testing.assert_allclose(np.linalg.norm(matrix.T @ residual), 1)
        coarse = zero_residual_spur(matrix, key, residual, 512, 32)
        fine = zero_residual_spur(matrix, key, residual, 1024, 64)
        np.testing.assert_allclose(coarse, fine, rtol=2e-4, atol=1e-7)
        print(f'{name:<26} {100*coarse:14.4f} {100*fine:24.4f}')


if __name__ == '__main__':
    main()
