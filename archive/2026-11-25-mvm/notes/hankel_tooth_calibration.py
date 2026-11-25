"""Fit one complex gain per output tooth and test on independent 2 pJ inputs.

Use the same phase-coded, depleted forward-read model as hankel_phase_recovery.
All errors are coherent field errors on the 64 retained teeth, before noise.
"""

import numpy as np

from hankel_fwm_stress import (
    ETA, INPUT_LOSS_DB, N, OUTPUT_LOSS_DB, PACKET_POWER_W, SLOTS,
    field_gain, optical_program, read,
)


def measure(matrix, keys, q, samples=1024, steps=48):
    weight, modes, scale = optical_program(q[:, None] * matrix * q, samples)
    weak_gain = np.sqrt(ETA / N) * field_gain(-INPUT_LOSS_DB - OUTPUT_LOSS_DB) / (4 * scale)
    decode = samples * np.sqrt(PACKET_POWER_W) * weak_gain * q
    outputs = []
    for key in keys:
        weak, actual, _, _, residual = read(weight, modes, q.conj() * key, 0, steps)
        np.testing.assert_allclose(np.fft.fft(weak)[:N] / decode, matrix @ key, atol=1e-12)
        assert residual < 1e-5
        outputs.append(np.fft.fft(actual)[:N] / decode)
    return np.array(outputs)


def correction(actual, target, axis):
    return np.sum(actual.conj() * target, axis=axis) / np.sum(abs(actual)**2, axis=axis)


def error(actual, target):
    return 100 * np.linalg.norm(actual - target) / np.linalg.norm(target)


def main():
    q = np.exp(1j * np.pi * SLOTS**2 / (2 * N))
    diagonal = np.diag(np.sign((q**2 * np.exp(-0.17j)).real))
    rng = np.random.default_rng(8011)
    dense, r = np.linalg.qr(rng.standard_normal((N, N)))
    dense *= np.sign(np.diag(r))
    pilots, tests = 128, 64
    keys = rng.standard_normal((pilots + tests, N))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    # The fitting rule must recover a known, fixed amplitude and phase response.
    static_gain = np.linspace(0.5, 1.5, N) * q
    fitted = correction(static_gain * keys[:pilots], keys[:pilots], axis=0)
    np.testing.assert_allclose(fitted * static_gain, 1, atol=1e-13)

    print(f'Pooled held-out in-band relative RMS field error (%), {pilots} pilots and {tests} tests')
    print('matrix              weak normalization    one complex gain    64 complex gains')
    for name, matrix in [('identity', np.eye(N)), ('dense orthogonal', dense),
                         ('signed diagonal', diagonal)]:
        actual = measure(matrix, keys, q)
        target = keys @ matrix.T
        held, desired = actual[pilots:], target[pilots:]
        scalar = correction(actual[:pilots], target[:pilots], axis=None)
        gains = correction(actual[:pilots], target[:pilots], axis=0)
        print(f'{name:<18} {error(held, desired):12.3f} {error(scalar*held, desired):19.3f}'
              f' {error(gains*held, desired):19.3f}')
        refined = measure(matrix, keys[pilots:pilots+1], q, samples=2048, steps=96)
        np.testing.assert_allclose(refined[0], held[0], atol=1e-5)

    # The diagonal map should leave output 0 unchanged when only other signs change.
    equal = np.ones(N) / np.sqrt(N)
    reinforced = np.sign((q * np.exp(-0.17j)).real) / np.sqrt(N)
    dark = equal.copy()
    dark[0] = 0
    dark /= np.linalg.norm(dark)
    probes = np.stack((equal, reinforced, dark, 0.5 * reinforced))
    actual = measure(diagonal, probes, q, samples=1024, steps=48)
    refined = measure(diagonal, probes, q, samples=2048, steps=96)
    np.testing.assert_allclose(actual, refined, atol=1e-6)
    target = probes @ diagonal.T
    gain = actual[:2, 0] / target[:2, 0]
    print('Same target on tooth 0, measured field gain and phase (degrees):',
          abs(gain), np.angle(gain, deg=True))
    best = correction(actual[:2, 0], target[:2, 0], axis=0)
    print('Best tooth-0 correction even with both answers known, relative RMS error (%):',
          error(best * actual[:2, 0], target[:2, 0]))
    print('Dark tooth 0, fraction of unit target vector norm:', abs(actual[2, 0]))
    print('Homogeneity defect, norm(F(x) - 2F(x/2)) / norm(Wx):',
          np.linalg.norm(actual[1] - 2 * actual[3]) / np.linalg.norm(target[1]))


if __name__ == '__main__':
    main()
