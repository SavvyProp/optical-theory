"""Test the weak-FWM assumption in the shared-interleaver read at 2 pJ.

All four bands share a group velocity and photon energy. In each retarded
time slice only the desired, exactly phase-matched interaction is retained.
Pump depletion and signal amplification are integrated together. This is a
stress test, not a dispersive device model. SPM, XPM, group-velocity mismatch,
internal loss, material dynamics, parasitic FWM and noise are omitted.

The effective coupling matches the monochromatic, undepleted, small-seed
power conversion of 0.468 for two 50 mW pumps. That number is extrapolated
from a measured CW converter; see scattering_multipliers.md. Uniform insertion
losses are applied to amplitudes. The FWM output remains in the full generated
band family, including waveform distortion outside the 64 target teeth.
"""

import numpy as np

from hankel_temporal_combiner import N, masked_weights


PACKET_POWER_W = 2e-12 / 20e-12
PUMP_W = 0.050
ETA = 0.468
INPUT_LOSS_DB = 2.2
OUTPUT_LOSS_DB = 0.5 + 2.2 + 0.5  # Pump rejection, return interleaver, final gate
SLOTS = np.arange(N)
TAPS = 1 / np.sqrt(2 * N)  # Magnitudes at the fixed port, signs corrected by R
KAPPA_L = np.arcsinh(np.sqrt(ETA)) / PUMP_W


def field_gain(db):
    return 10 ** (db / 20)


def phase_matched_output(weight, data, reference, steps, *, return_weight=False):
    """Solve the four trilinear amplitude equations by RK4 in retarded time.

    With the generated band initially empty, its phase is the phase of
    i*h*R*conj(A), up to a real sign. Rotate to this phase convention to
    reduce the complex system to real signed amplitudes H, R, A, B. Calibrate
    away the common i when comparing to the weak-field read. Each squared
    amplitude is a power in W in the equal-photon-energy approximation.
    Set return_weight to also get the surviving complex weight field.
    """
    shape = weight.shape
    initial = np.stack((abs(weight), np.full(shape, reference),
                        np.broadcast_to(abs(data), shape), np.zeros(shape)))
    state = initial.copy()
    step = KAPPA_L / steps

    def derivative(fields):
        h, r, a, b = fields
        pair = a * b
        pumps = h * r
        return np.stack((-r * pair, -h * pair, pumps * b, pumps * a))

    for _ in range(steps):
        k1 = derivative(state)
        k2 = derivative(state + 0.5 * step * k1)
        k3 = derivative(state + 0.5 * step * k2)
        k4 = derivative(state + step * k3)
        state += (step / 6) * (k1 + 2*k2 + 2*k3 + k4)

    h, r, a, b = state
    h0, r0, a0, _ = initial
    # Photon-flux Manley–Rowe invariants. Scale the absolute residual in W
    # to the 50 mW reference power, independent of local seed or near-zero h.
    residual = max(np.max(abs(h*h + b*b - h0*h0)),
                   np.max(abs(r*r + b*b - r0*r0)),
                   np.max(abs(a*a - b*b - a0*a0))) / PUMP_W
    generated_phase = np.exp(1j * (np.angle(weight) - np.angle(data)))
    idler = b * generated_phase
    if return_weight:
        return idler, residual, h * np.exp(1j * np.angle(weight))
    return idler, residual


def projected_gain(output, target):
    return np.vdot(target, output) / np.vdot(target, target)


def cases():
    rng = np.random.default_rng(310)
    matrix, triangular = np.linalg.qr(rng.standard_normal((N, N)))
    matrix *= np.sign(np.diag(triangular))
    key = rng.standard_normal(N)
    key /= np.linalg.norm(key)
    return [('identity, one tooth', np.eye(N), np.eye(N)[10]),
            ('identity, equal phases', np.eye(N), np.ones(N)/np.sqrt(N)),
            ('identity, random signed', np.eye(N), key),
            ('dense orthogonal, same key', matrix, key)]


def optical_program(matrix, samples):
    time = np.arange(samples) / samples
    modes = np.exp(2j*np.pi*np.arange(2*N-1)[:, None]*time)
    code = np.exp(-2j*np.pi*np.outer(N-1-SLOTS, SLOTS)/N) / np.sqrt(N)
    program = masked_weights(matrix.T, code.conj())
    norm = np.linalg.norm(program, axis=1).max()
    weight = np.sqrt(PUMP_W) * (program @ modes) / norm
    return weight, modes[:N], norm


def raw_idlers(weight, data, steps, reference_power_w=PUMP_W):
    reference = np.sqrt(reference_power_w)
    data = np.broadcast_to(data, weight.shape)
    weak = (np.sqrt(ETA) / PUMP_W) * weight * reference * data.conj()
    # Independent time slices allow small slot batches to stay in CPU cache.
    depleted = np.empty_like(weak)
    residual = 0
    for start in range(0, N, 4):
        slots = slice(start, start + 4)
        depleted[slots], batch_error = phase_matched_output(weight[slots], data[slots], reference, steps)
        residual = max(residual, batch_error)
    return weak, depleted, residual


def read(weight, modes, vector, boost_db, steps, *, reverse=False, reference_power_w=PUMP_W):
    samples = modes.shape[1]
    shifts = (N-1-SLOTS)[:, None] * (samples // N)
    gain_in = TAPS * np.sqrt(PACKET_POWER_W) * field_gain(boost_db - INPUT_LOSS_DB)
    gain_out = TAPS * field_gain(-OUTPUT_LOSS_DB)
    if reverse:
        # The same 20 ps copier now acts on the incident B band. Advance
        # its launch by 63 delta to sample the conjugate mask in weight slot m.
        seed = vector @ modes
        data = (gain_in / 2) * seed[(np.arange(samples) + shifts) % samples]
        weak, depleted, residual = raw_idlers(weight, data, steps, reference_power_w)
        return gain_out*weak.sum(axis=0), gain_out*depleted.sum(axis=0), weak, depleted, residual

    # For isolated 45 ps slots and the selected 20 ps gate, each raw partial
    # m follows path 63-m and contributes one cyclic shift plus the 1/2 copier
    # field coefficient (verified for all interleaver paths in hankel_comb_codes.py).
    data = gain_in * (vector.conj() @ modes)
    weak, depleted, residual = raw_idlers(weight, data, steps, reference_power_w)
    local_indices = (np.arange(samples) - shifts) % samples

    def collect(field):
        shifted = np.take_along_axis(field, local_indices, axis=1)
        return shifted.sum(axis=0) * (gain_out/2)

    return collect(weak), collect(depleted), weak, depleted, residual


def stress(matrix, key, boost_db, samples, steps):
    weight, modes, norm = optical_program(matrix, samples)
    weak_field, depleted_field, weak, depleted, residual = read(weight, modes, key, boost_db, steps)
    target = np.sqrt(PACKET_POWER_W) * ((matrix @ key) @ modes)
    weak_scale = projected_gain(weak_field, target)
    expected = np.sqrt(ETA/N) * field_gain(boost_db - INPUT_LOSS_DB - OUTPUT_LOSS_DB) / (4*norm)
    np.testing.assert_allclose(weak_field, expected * target, atol=5e-14, rtol=1e-11)
    scale = projected_gain(depleted_field, target)
    shape_error = np.linalg.norm(depleted_field - scale*target) / np.linalg.norm(scale*target)
    intended_modes = np.fft.fft(scale * target)[:N]
    returned_modes = np.fft.fft(depleted_field)[:N]
    in_band_error = np.linalg.norm(returned_modes - intended_modes) / np.linalg.norm(intended_modes)
    return np.array([20*np.log10(abs(weak_scale)), 20*np.log10(abs(scale)),
                     100*shape_error, 1000*np.max(abs(weak))**2,
                     1000*np.max(abs(depleted))**2, residual, 100*in_band_error])


def main():
    # Double temporal sampling at fixed RK step, then double RK steps at
    # the finer grid to test the two discretizations independently.
    print('Power gains are projections on the intended waveform, before restoration.')
    print(f'{"program, input":<28} boost samples  weak dB   ODE dB   shape %  64-mode %  weak peak mW  ODE peak mW')
    grid_delta = np.zeros(7)
    step_delta = np.zeros(7)
    max_invariant_error = 0
    for label, matrix, key in cases():
        for boost in (0, 10):
            # Coincident Dirichlet peaks produce the sharpest depleted-wave
            # features, especially with a boosted input. Resolve those cases
            # more finely rather than relaxing the convergence test.
            samples = 4096
            if label == 'identity, equal phases':
                samples = 16384 if boost else 8192
            coarse = stress(matrix, key, boost, samples // 2, 48)
            finer_grid = stress(matrix, key, boost, samples, 48)
            final = stress(matrix, key, boost, samples, 96)
            grid_delta = np.maximum(grid_delta, abs(finer_grid - coarse))
            step_delta = np.maximum(step_delta, abs(final - finer_grid))
            weak_db, exact_db, error, weak_peak, exact_peak, invariant_error, mode_error = final
            max_invariant_error = max(max_invariant_error, invariant_error)
            print(f'{label:<28} {boost:5d} {samples:7d} {weak_db:8.2f} {exact_db:8.2f} {error:9.2f} {mode_error:10.2f}'
                  f' {weak_peak:13.2f} {exact_peak:12.2f}')
    for name, diff in [('Double samples per period', grid_delta), ('RK4 48→96', step_delta)]:
        print(f'{name}: max projected-gain change {diff[1]:.3g} dB,'
              f' shape-error change {diff[2]:.3g} percentage points')
    print(f'Largest final Manley–Rowe residual / reference power: {max_invariant_error:.3g}')
    assert max(grid_delta[1], step_delta[1]) < 0.01
    assert max(grid_delta[2], step_delta[2], grid_delta[6], step_delta[6]) < 0.1
    assert max_invariant_error < 1e-5

    # Independent real directional-derivative check. Launch a unit-energy
    # error and use exactly the same powers and old weights for both reads.
    # Common output amplifier gains multiply both sides of the test equally.
    # Divide out only the programmed *weak* gain, once for both quantities.
    print('Directional derivative vs reused reverse read (unit weak gain)')
    for label, matrix, key in (cases()[0], cases()[-1]):
        samples, steps = 4096, 96
        weight, modes, norm = optical_program(matrix, samples)
        direction = key  # Test the radial derivative at a full-scale input.
        error = matrix @ direction
        weak_gain = np.sqrt(ETA/N) * field_gain(-INPUT_LOSS_DB - OUTPUT_LOSS_DB) / (4*norm)

        def mode_amplitudes(packet):
            return np.fft.fft(packet)[:N] / (samples*np.sqrt(PACKET_POWER_W)*weak_gain)

        weak_reverse, reverse, *_ = read(weight, modes, error, 0, steps, reverse=True)
        # The A band carries conjugated logical coefficients on return.
        reverse_dot = np.vdot(mode_amplitudes(reverse).conj(), direction).real
        weak_reverse_dot = np.vdot(mode_amplitudes(weak_reverse).conj(), direction).real
        prior_fd = None
        for eps in (1e-3, 1e-4):
            plus = read(weight, modes, key + eps*direction, 0, steps)
            minus = read(weight, modes, key - eps*direction, 0, steps)
            weak_fd = np.vdot(error, mode_amplitudes(plus[0] - minus[0])).real / (2*eps)
            fd = np.vdot(error, mode_amplitudes(plus[1] - minus[1])).real / (2*eps)
            np.testing.assert_allclose([weak_fd, weak_reverse_dot], [1, 1], atol=2e-10, rtol=0)
            if prior_fd is not None:
                np.testing.assert_allclose(fd, prior_fd, atol=2e-5, rtol=1e-5)
            prior_fd = fd
        print(f'{label:<28} weak {weak_fd:.6f}/{weak_reverse_dot:.6f},'
              f' depleted derivative {fd:.6f}, reverse {reverse_dot:.6f}')


if __name__ == '__main__':
    main()
