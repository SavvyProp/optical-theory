"""Discrete fields for the output pulse interleaver and candidate input pulse extenders."""

import numpy as np
from scipy.linalg import hadamard


N = 64
DELAYS = (32, 16, 8, 4, 2, 1)
SPACING_HZ = 50e9


def combine(packets, error_fs=None, offsets_hz=None, *, long_arm_masks=None):
    """Return both output rails. Optional field masks follow the longest-first delays."""
    n_slots, n_channels = packets.shape
    early = np.arange(n_slots) % N < N // 2
    upper = np.zeros((n_slots + N - 1, n_channels), complex)
    lower = np.zeros_like(upper)
    upper[:n_slots] = packets * (~early)[:, None]
    lower[:n_slots] = packets * early[:, None]
    if offsets_hz is None:
        offsets_hz = np.zeros(n_channels)
    if error_fs is None:
        error_fs = np.zeros(len(DELAYS))
    for stage, (delay, error) in enumerate(zip(DELAYS, error_fs, strict=True)):
        # Nominal delay is an integer number of comb periods. Residual group delay
        # contributes a carrier phase after center-frequency trimming. The packet
        # envelopes are assumed orthogonal, with perfect switch timing.
        delayed = np.zeros_like(lower)
        phase = np.exp(-2j * np.pi * offsets_hz * error * 1e-15)
        if long_arm_masks is not None:
            phase = phase * long_arm_masks[stage]
        delayed[delay:] = lower[:-delay] * phase
        upper, lower = (upper + delayed) / np.sqrt(2), (upper - delayed) / np.sqrt(2)
    return upper, lower


def replay(frames, long_arm_masks=None, *, stages=len(DELAYS)):
    """Operate the binary-delay network as a pulse extender, with optional field masks."""
    n_frames, n_channels = frames.shape
    frame_slots = 2**stages
    upper = np.zeros((n_frames * frame_slots, n_channels), complex)
    upper[::frame_slots] = frames
    lower = np.zeros_like(upper)
    # Each balanced coupler now comes before its delay, in shortest-first order.
    for stage in range(stages):
        delay = 2**stage
        upper, long_arm = (upper + lower) / np.sqrt(2), (upper - lower) / np.sqrt(2)
        if long_arm_masks is not None:
            long_arm *= long_arm_masks[stage]
        lower = np.zeros_like(upper)
        lower[delay:] = long_arm[:-delay]
    early = (np.arange(len(upper)) % frame_slots < frame_slots // 2)[:, None]
    return np.where(early, upper, lower), np.where(early, lower, upper)


def taps_and_code():
    # Probing each of the 64 input slots separately yields the selected output row.
    selected = combine(np.eye(N))[0][N - 1]
    np.testing.assert_allclose(np.abs(selected), 1 / np.sqrt(N), atol=1e-14)
    return selected, np.conj(selected) / np.abs(selected)


def masked_weights(matrix, masks):
    """Expand each anti-diagonal of the target matrix in the fixed Walsh masks."""
    rows, columns = matrix.shape
    weights = np.zeros((len(masks), rows + columns - 1), dtype=np.result_type(matrix, masks))
    for col in range(columns):
        weights[:, col:col + rows] += masks[:, col, None] * matrix[:, col]
    return weights


def check_output_mask_training(matrix, masks, arm_masks, taps, slot_phases):
    """Forward, input gradient and dense delta write with the masks on the output band."""
    rng = np.random.default_rng(684)
    vectors = rng.normal(size=(3, N)) / np.sqrt(N)
    if np.iscomplexobj(matrix):
        vectors = (vectors + 1j * rng.normal(size=(3, N)) / np.sqrt(N)) / np.sqrt(2)
    key, target, query = vectors
    index = np.add.outer(np.arange(N), np.arange(N))
    # To reach the output gate, input slot m takes the delays for N-1-m.
    output_codes = masks[::-1]
    weights = masked_weights(matrix.T, output_codes)
    copy_amplitude = 1 / np.sqrt(N)

    def read(stored, vector):
        # Equal phase-aligned copies; the input band carries the conjugate logical field.
        input_field = copy_amplitude * vector.conj()
        partials = stored[:, index] @ input_field.conj()
        packet = combine(partials * slot_phases[:, None], long_arm_masks=arm_masks[::-1])[0][N - 1]
        return packet / copy_amplitude  # Calibrated common read gain.

    prediction = read(weights, key)
    error = target - prediction
    # Used in reverse, the output pulse interleaver emits the masks in the opposite
    # order. Their pointwise product changes row m to row N-1-m for every m.
    correction = arm_masks.prod(axis=0)
    error_slots, unused = replay((correction * error)[None, :], arm_masks)
    error_slots *= (taps[::-1].conj() / abs(taps[::-1]))[:, None]
    np.testing.assert_allclose(unused, 0, atol=1e-14)
    np.testing.assert_allclose(error_slots, output_codes * error, atol=1e-14)

    backward_partials = np.einsum('mij,mi->mj', weights[:, index], error_slots.conj())
    input_band = combine(backward_partials * slot_phases[:, None])[0][N - 1] / copy_amplitude
    input_error = input_band.conj()
    # Each FWM visit convolves the two fields onto their sum-frequency indices.
    write = np.stack([np.convolve(slot, copy_amplitude * key.conj()) for slot in error_slots]) / copy_amplitude
    effective_write = np.einsum('mi,mij->ij', output_codes, write[:, index])
    dense_write = np.outer(error, key.conj())
    rate = 0.125
    updated = matrix + rate * dense_write
    checks = {
        'forward read': (prediction, matrix @ key),
        'input descent signal': (input_error, matrix.conj().T @ error),
        'effective matrix increment': (effective_write, dense_write),
        'post-write query': (read(weights + rate * write, query), updated @ query),
    }
    field_type = 'complex' if np.iscomplexobj(matrix) else 'real'
    for name, (actual, expected) in checks.items():
        relative_error = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
        np.testing.assert_allclose(actual, expected, rtol=1e-12, atol=1e-13)
        print(f'Walsh masks in the output pulse interleaver, {field_type} {name} relative error {relative_error:.2g}')

    direction = rng.normal(size=weights.shape)
    if np.iscomplexobj(matrix):
        direction = direction + 1j * rng.normal(size=weights.shape)
    step = 1e-5
    losses = [0.5 * np.linalg.norm(read(weights + sign * step * direction, key) - target)**2
              for sign in (-1, 1)]
    numerical_derivative = (losses[1] - losses[0]) / (2 * step)
    np.testing.assert_allclose(numerical_derivative, -np.vdot(write, direction).real,
                               rtol=1e-8, atol=1e-10)


def equal_loop_releases(count, loss_db):
    """Extract equal pulse energies from one unit-energy packet, with the first tap at entry."""
    survival = 10**(-loss_db / 10)
    stored = 1.0
    energies = []
    for remaining in range(count, 0, -1):
        tap = 1 / np.sum(survival**(-np.arange(remaining)))
        released = tap * stored
        energies.append(released)
        stored = (stored - released) * survival
    return np.array(energies)


def check_compact_replay(x, masks, long_arm_masks):
    for losses_db in (0.1, 0.89, 1.29):
        for count in (8, 64):
            energies = equal_loop_releases(count, losses_db)
            expected = 1 / np.sum(10**(losses_db * np.arange(count) / 10))
            np.testing.assert_allclose(energies, expected, rtol=1e-12, atol=0)
            print(f"{count} equal releases at {losses_db} dB/lap: "
                  f"{-10*np.log10(energies.sum()):.2f} dB loop penalty")

    # Tap eight groups; apply the three high Walsh-index bits per group and let
    # the short mesh supply the three low bits. Calibrate its common slot phases.
    energies = equal_loop_releases(8, 0.89)
    groups = np.sqrt(energies[:, None]) * np.sign(masks[::8]) * x
    slots, unused = replay(groups, long_arm_masks[:3], stages=3)
    reference, _ = replay(np.ones((8, 1)), stages=3)
    corrected = slots * np.conj(reference) / np.abs(reference)
    np.testing.assert_allclose(unused, 0, atol=1e-14)
    np.testing.assert_allclose(corrected, np.sqrt(energies.sum()) * masks * x, atol=1e-13)

    # A target for a static grating. Complementary autocorrelations imply that
    # |P|² + |Q|² = 2N at every frequency, including between the comb teeth.
    p = q = np.ones(1)
    for _ in DELAYS:
        p, q = np.concatenate((p, q)), np.concatenate((p, -q))
    autocorrelation = np.correlate(p, p, "full") + np.correlate(q, q, "full")
    target = np.zeros(2*N - 1)
    target[N - 1] = 2*N
    np.testing.assert_array_equal(autocorrelation, target)
    grating_taps = p / np.sqrt(2*N)
    print("Isolated-packet energy fraction for the coded grating target", grating_taps @ grating_taps)


def main():
    rng = np.random.default_rng(6332)
    taps, code = taps_and_code()
    z = rng.normal(size=(3, N, N)) + 1j * rng.normal(size=(3, N, N))
    # One general complex frame, one exact null and one perfectly matched frame.
    z[1, -1] = -z[1, :-1].sum(axis=0)
    z[2] = z[2, 0].copy()
    packets = (z * code[None, :, None]).reshape(-1, N)
    upper, lower = combine(packets)
    read_slots = np.arange(1, len(z) + 1) * N - 1
    collected = upper[read_slots]
    target = z.sum(axis=1) / np.sqrt(N)
    np.testing.assert_allclose(collected, target, atol=2e-13)
    np.testing.assert_allclose(np.linalg.norm(upper)**2 + np.linalg.norm(lower)**2,
                               np.linalg.norm(packets)**2, rtol=1e-13)
    print("Selected tap magnitudes", abs(taps).min(), abs(taps).max())
    random_error = np.linalg.norm(collected[0] - target[0]) / np.linalg.norm(target[0])
    print("Random frame relative field error", random_error)
    print("Null amplitude / incident field norm", np.linalg.norm(collected[1]) / np.linalg.norm(z[1]))
    print("Matched frame energy fraction", np.linalg.norm(collected[2])**2 / np.linalg.norm(z[2])**2)

    # Feed the gated comb packets from the three frames into a second mesh.
    # Reciprocity gives the collection taps in reverse time order for replay.
    replay_taps = taps[::-1]
    phase_trim = np.conj(replay_taps) / np.abs(replay_taps)
    train, unused = replay(collected)
    np.testing.assert_allclose(unused, 0, atol=1e-14)
    train = train.reshape(len(collected), N, N)
    np.testing.assert_allclose(train, collected[:, None, :] * replay_taps[None, :, None], atol=1e-13)
    print("Fraction of gated energy on the candidate input pulse extender's selected waveguide",
          np.linalg.norm(train)**2 / np.linalg.norm(collected)**2)
    # Compensate the two fixed path codes, bypass the multiplier and recollect.
    coded_train = train * (phase_trim * code)[None, :, None]
    restacked = combine(coded_train.reshape(-1, N))[0][read_slots]
    np.testing.assert_allclose(restacked, collected, atol=2e-13)
    print("Input pulse extender followed by output pulse interleaver, relative field error",
          np.linalg.norm(restacked - collected) / np.linalg.norm(collected))

    masks = hadamard(N) / np.sqrt(N)
    index = np.add.outer(np.arange(N), np.arange(N))
    peaks_squared = []
    for trial in range(100):
        q, r = np.linalg.qr(rng.normal(size=(N, N)))
        matrix = q * np.sign(np.diag(r))
        weights = masked_weights(matrix, masks)
        peak = np.max(np.linalg.norm(weights, axis=1))
        peaks_squared.append(peak**2)
        if trial == 0:
            example_matrix = matrix
            example_weights = weights

    x = rng.normal(size=N) + 1j * rng.normal(size=N)
    # In delay arm 2^b, flip the sign of teeth whose b-th index bit is set.
    # The paths to slot m then implement Walsh row m, including replay normalization.
    delay_bits = np.array(DELAYS[::-1])[:, None]
    long_arm_masks = np.where((delay_bits & np.arange(N)) != 0, -1, 1)
    check_compact_replay(x, masks, long_arm_masks)
    masked_input, _ = replay(x.conj()[None, :], long_arm_masks)
    np.testing.assert_allclose(masked_input, replay_taps[:, None] * np.sign(masks) * x.conj(),
                               atol=1e-14)
    partials = np.einsum("mij,mj->mi", example_weights[:, index], masked_input.conj())
    # The FWM reference can compensate the conjugated replay phase and the collection phase.
    reference_phase = phase_trim.conj() * code
    result = combine(partials * reference_phase[:, None])[0][N - 1]
    expected = example_matrix @ x / np.sqrt(N)
    np.testing.assert_allclose(result, expected, atol=1e-13)
    print("Walsh masks in the candidate input pulse extender, dense read relative field error",
          np.linalg.norm(result - expected) / np.linalg.norm(expected))

    # Ideal input replay, fixed pump cap per slot, flat weak-conversion model.
    efficiency = 0.468
    ideal_gain_db = 10 * np.log10(efficiency / (N * np.array(peaks_squared)))
    print("Ideal identity seed-to-target gain (dB)", 10 * np.log10(efficiency / N))
    print("Ideal orthogonal-matrix gain p10/50/90 (dB)", np.quantile(ideal_gain_db, [0.1, 0.5, 0.9]))

    # Error in relative slot weights after the best fixed complex output scale
    # per tooth. Use a deterministic +10 fs error on each of the six long arms.
    teeth = np.arange(N)
    offset = (teeth - (N - 1) / 2) * SPACING_HZ
    path_bits = ((N - 1 - teeth[:, None]) & np.array(DELAYS)) != 0
    errors_fs = np.full(len(DELAYS), 10.0)
    path_error_s = (path_bits @ errors_fs)[:, None] * 1e-15
    phase = np.exp(-2j * np.pi * path_error_s * offset)
    erroneous = combine(packets, errors_fs, offset)[0][read_slots]
    predicted = np.einsum("fmj,mj->fj", z, phase) / np.sqrt(N)
    np.testing.assert_allclose(erroneous, predicted, atol=2e-13)
    best_scale = phase.mean(axis=0).conj()
    tap_error = np.sqrt(np.mean(np.abs(phase * best_scale - 1)**2))
    print("Tap RMS field error with 10 fs excess in each long arm", tap_error)

    # A static delay error adds a linear phase versus idler tooth i. In the FWM
    # product the weight index is s=i+j and the input mask on j is conjugated.
    # Ramps on those two combs thus generate a calibrated ramp on i for each slot.
    slope = 2j * np.pi * SPACING_HZ * path_error_s
    weight_ramp = np.exp(slope * (np.arange(2 * N - 1) - (N - 1) / 2))
    input_ramp = np.exp(slope * teeth)
    corrected = np.einsum("mij,mj->mi", (example_weights * weight_ramp)[:, index],
                          (masks * input_ramp * x.conj()).conj())
    recovered = combine(corrected * code[:, None], errors_fs, offset)[0][N - 1]
    np.testing.assert_allclose(recovered, expected, atol=1e-13)
    residual = np.linalg.norm(recovered - expected) / np.linalg.norm(expected)
    print("Dense read error after compensating the carrier phases", residual)
    check_output_mask_training(example_matrix, masks, long_arm_masks, taps, code)
    complex_matrix = example_matrix + 0.5j * np.roll(example_matrix, 5, axis=0)
    check_output_mask_training(complex_matrix, masks, long_arm_masks, taps, code)


if __name__ == "__main__":
    main()
