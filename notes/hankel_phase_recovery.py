"""Test fixed spectral phase conventions with the depleted shared-interleaver model.

The physical A/B fields carry qa*conj(x) and qb*y, respectively. The optical
matrix is diag(qb) W diag(qa) because the guide conjugates the A field. An
optical outer-product write uses the same phases on the saved input and error.
The ordinary logical coefficients and squared packet energies are unchanged.
"""

import numpy as np

from hankel_fwm_stress import (
    ETA, INPUT_LOSS_DB, N, OUTPUT_LOSS_DB, PACKET_POWER_W, PUMP_W, SLOTS,
    KAPPA_L, TAPS, cases, field_gain, optical_program, phase_matched_output, read,
)
from hankel_comb_codes import code


READ_GAIN_DB = 39.8


def golay_signs(bit_order):
    # Bipolar Golay sequence from a path on the six binary digits.
    bits = ((SLOTS[:, None] >> np.asarray(bit_order)) & 1)
    parity = np.sum(bits[:, :-1] * bits[:, 1:], axis=1) % 2
    return 1 - 2 * parity


def response(matrix, key, qa, qb, samples=2048, steps=48, derivative=False, bank_scale=None):
    physical_matrix = qb[:, None] * matrix * qa
    encoded = qa.conj() * key  # read() conjugates this argument on the A port.
    desired = qb * (matrix @ key)
    weight, modes, peak_norm = optical_program(physical_matrix, samples)
    if bank_scale is not None:
        assert bank_scale >= peak_norm
        weight *= peak_norm / bank_scale
        peak_norm = bank_scale
    weak, actual, raw_weak, _, _ = read(weight, modes, encoded, 0, steps)
    target = np.sqrt(PACKET_POWER_W) * (desired @ modes)
    weak_scale = np.sqrt(ETA/N) * field_gain(-INPUT_LOSS_DB-OUTPUT_LOSS_DB) / (4*peak_norm)
    np.testing.assert_allclose(weak, weak_scale*target, rtol=1e-11, atol=1e-11)
    scale = np.vdot(target, actual) / np.vdot(target, target)
    residual = actual - scale*target
    full = np.linalg.norm(residual) / np.linalg.norm(scale*target)
    in_band = np.linalg.norm(np.fft.fft(residual)[:N]) / np.linalg.norm(np.fft.fft(scale*target)[:N])
    # Fixed field gain from the linear model; no individual amplitude correction.
    uncalibrated = np.linalg.norm(actual - weak) / np.linalg.norm(weak)
    result = [20*np.log10(abs(scale)) + READ_GAIN_DB, 100*full, 100*in_band,
              100*uncalibrated, 1000*np.max(abs(raw_weak))**2, peak_norm]
    if derivative:
        def coefficients(packet):
            return np.fft.fft(packet)[:N] / (samples*np.sqrt(PACKET_POWER_W)*weak_scale)

        # A radial test at the operating input, with e equal to the ideal
        # output. Real inner products accommodate the complex physical gauge.
        eps = 1e-4
        plus = read(weight, modes, (1+eps)*encoded, 0, steps)
        minus = read(weight, modes, (1-eps)*encoded, 0, steps)
        weak_back, back, *_ = read(weight, modes, desired, 0, steps, reverse=True)
        weak_fd = np.vdot(desired, coefficients(plus[0]-minus[0])).real / (2*eps)
        weak_reverse = np.vdot(coefficients(weak_back).conj(), encoded).real
        np.testing.assert_allclose([weak_fd, weak_reverse], [1, 1], rtol=0, atol=1e-9)
        fd = np.vdot(desired, coefficients(plus[1]-minus[1])).real / (2*eps)
        reverse = np.vdot(coefficients(back).conj(), encoded).real
        result.extend((fd, reverse))
    return np.array(result)


def check_phase_coded_writes(q):
    """Propagate a single optical outer-product write, prior to fiber injection."""
    samples = 1024
    modes = np.exp(2j*np.pi*np.outer(SLOTS, np.arange(samples))/samples)
    masks = code(-(N-1-SLOTS))
    tap = TAPS*np.sqrt(PACKET_POWER_W)*field_gain(-INPUT_LOSS_DB)
    reference = np.sqrt(PUMP_W)
    indices = SLOTS[:, None] + SLOTS
    write_scale = KAPPA_L*reference*tap**2*np.sqrt(N)/2
    rng = np.random.default_rng(23)
    pair = rng.normal(size=(2, N))
    pair /= np.linalg.norm(pair, axis=1)[:, None]
    equal = np.ones(N)/np.sqrt(N)
    crest = np.where(np.real(q*np.exp(-0.17j)) >= 0, 1., -1.)/np.sqrt(N)
    for name, key, error in [('equal/equal', equal, equal), ('two signed', *pair),
                             ('reinforcing signs', crest, crest)]:
        a = tap*((q*key) @ modes)
        b = tap*np.sqrt(N)/2*((masks.conj()*(q*error)) @ modes)
        weak = KAPPA_L*reference*a*b
        exact = np.empty_like(b)
        for start in range(0, N, 4):
            slots = slice(start, start+4)
            # During the write A and B are the two pumps, R is the seeded band.
            generated, residual = phase_matched_output(
                np.broadcast_to(a, b[slots].shape), abs(b[slots]), reference, 48)
            exact[slots] = generated*np.exp(1j*np.angle(b[slots]))
            assert residual < 1e-6

        def represented_increment(waveform):
            harmonics = np.fft.fft(waveform, axis=1)[:, :2*N-1] / (samples*write_scale)
            return np.einsum('mr,mrj->rj', masks, harmonics[:, indices])

        target = np.outer(q*error, q*key)
        np.testing.assert_allclose(represented_increment(weak), target, atol=2e-13)
        relative = np.linalg.norm(represented_increment(exact)-target) / np.linalg.norm(target)
        print(f'one noiseless write, {name}: {100*relative:.3f}% fixed-calibration matrix error')


def main():
    q = np.exp(1j*np.pi*SLOTS**2/(2*N))
    signs_a = golay_signs([0, 1, 2, 3, 4, 5])
    signs_b = signs_a * golay_signs([1, 5, 2, 3, 0, 4])
    # Select real unit-norm inputs that reinforce the encoded teeth at t=0.
    crest = np.linalg.svd(np.stack((q.real, q.imag)), full_matrices=False)[2][0]
    # A fixed offset avoids floating-point decisions exactly at +/- pi/2.
    def reinforce(phases):
        return np.where(np.real(phases*np.exp(-0.17j)) >= 0, 1., -1.)

    diagonal = np.diag(reinforce(q*q))
    reversal = np.diag(reinforce(q*q[::-1]))[:, ::-1]
    tests = cases() + [('identity, crest input', np.eye(N), crest),
                       ('signed diagonal, crest input', diagonal, crest),
                       ('signed reversal, crest input', reversal, crest)]
    print('At 2 pJ with 50 mW on each pump; each matrix normalized to its peak slot. '
          'Fixed read gain +39.8 dB. Errors are deterministic.')
    print('encoding / test                            net dB   shape %   inband %'
          '  fixed %  raw weak peak mW   p_scale    radial FD   backward')
    for label, qa, qb in [('chirp', q, q), ('signs', signs_a, signs_b)]:
        for name, matrix, key in tests if label == 'chirp' else cases():
            out = response(matrix, key, qa, qb, derivative=True)
            print(f'{label} / {name:<33}' + ' '.join(f'{v:9.4f}' for v in out))
            if name in ('identity, equal phases', 'signed diagonal, crest input'):
                refined = response(matrix, key, qa, qb, 4096, 96, derivative=True)
                np.testing.assert_allclose(out[:4], refined[:4], rtol=2e-4, atol=2e-4)
                np.testing.assert_allclose(out[4], refined[4], rtol=0.01)
                np.testing.assert_allclose(out[5:], refined[5:], rtol=2e-4, atol=2e-4)

    # Independent random orthogonal targets and signed real unit vectors.
    # Hold a single weight normalization and weak field gain for the whole bank.
    rng = np.random.default_rng(906)
    programs = []
    for _ in range(32):
        matrix, tri = np.linalg.qr(rng.standard_normal((N, N)))
        matrix *= np.sign(np.diag(tri))
        key = rng.standard_normal(N)
        key /= np.linalg.norm(key)
        programs.append((matrix, key))
    bank_scale = max(optical_program(q[:, None]*matrix*q, N)[2] for matrix, _ in programs)
    ensemble = [response(matrix, key, q, q, 1024, 32, bank_scale=bank_scale)
                for matrix, key in programs]
    for (matrix, key), coarse in zip(programs[:2], ensemble[:2], strict=True):
        refined = response(matrix, key, q, q, 2048, 64, bank_scale=bank_scale)
        np.testing.assert_allclose(coarse[:4], refined[:4], rtol=2e-4, atol=2e-4)
    print('32 chirped random orthogonal programs; one fixed pump scale and read gain. '
          'Min / median / max in same columns')
    print(np.round(np.percentile(ensemble, [0, 50, 100], axis=0), 4))
    weight, _, _ = optical_program(np.diag(q*q), 4096)
    print(f'Chirped identity weight peak/mean {np.max(abs(weight))**2 / PUMP_W:.4f}')
    check_phase_coded_writes(q)


if __name__ == '__main__':
    main()
