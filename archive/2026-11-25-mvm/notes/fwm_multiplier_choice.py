"""Compare dense FWM reads and check timing-based reads, adjoints and writes."""

import numpy as np
from scipy.constants import c, h
from scipy.linalg import hadamard

from hankel_temporal_combiner import N, masked_weights


HANKEL = np.add.outer(np.arange(N), np.arange(N))
MASKS = hadamard(N) / np.sqrt(N)
FOURIER = np.exp(-2j * np.pi * np.multiply.outer(np.arange(N), np.arange(N)) / N) / np.sqrt(N)
PAIR_SLOTS = N // 2 + 1
SLOT_PS = 20
PUMP_MW = 100  # Combined reference and weight power at one single-pass guide.
ETA = 0.468  # Family power conversion for one seed and a unit-norm weight comb.
# Match the available time-integrated pump energy with two guides per pair slot.
PAIR_PUMP_RATIO = N / (2 * PAIR_SLOTS)
PAIR_ETA = ETA * PAIR_PUMP_RATIO**2
NF = 10**0.5
SIGNAL_PHOTONS = 2e-12 / (h * c / 1550e-9) / N


def diagonal_difference(matrix):
    return matrix[1:, 1:] - matrix[:-1, :-1]


def fixed_pair_program(matrix):
    """Assign each column of the diagonal differences to one boundary of a pair."""
    half = N // 2
    first = np.zeros((PAIR_SLOTS, 2 * N - 1))
    second = np.zeros_like(first, dtype=np.result_type(matrix, float))
    diff = diagonal_difference(matrix)
    for slot in range(half):
        high_col = half + slot
        first[slot, slot] = 1
        if high_col < N - 1:
            first[slot, N + high_col] = 1
            first[slot] /= np.sqrt(2)
            second[slot, N:] = diff[:, high_col] / first[slot, N + high_col]
        second[slot, :N - 1] = -diff[:, slot] / first[slot, slot]

    # The remainder has constant diagonals. Reversing its columns gives a Hankel.
    remainder = matrix - (second[:-1, HANKEL] @ first[:-1, HANKEL]).sum(axis=0)
    np.testing.assert_allclose(diagonal_difference(remainder), 0, atol=2e-13)
    first[-1, N - 1] = 1
    reversed_columns = remainder[:, ::-1]
    second[-1] = np.r_[reversed_columns[0], reversed_columns[1:, -1]]
    return first, second


def ase(gain):
    """Added photons in one matched mode. A gain of one bypasses the amplifier."""
    return (NF * gain - 1) / 2 if gain > 1 else 0


def restored_error(transfer, noise):
    gain = 1 / transfer
    return 100 * np.sqrt((gain * noise + ase(gain)) / SIGNAL_PHOTONS)


def power_db(transfer):
    return 10 * np.log10(transfer)


def single_hankel_program(matrix, masks, copies):
    # Each anti-diagonal is expanded in an orthonormal code. Equal power replay
    # and collection supply the field normalizations; a guard copy costs energy.
    weights = masked_weights(matrix, masks.conj())
    mask_norms = np.linalg.norm(weights, axis=1)
    np.testing.assert_allclose(np.einsum('mij,mj->ij', weights[:, HANKEL], masks),
                               matrix, atol=2e-13)
    masked_gain = ETA / (copies * mask_norms.max()**2)
    masked_noise = ETA * np.sum(np.abs(weights[:, HANKEL])**2) / (
        mask_norms.max()**2 * N * N)
    return masked_gain, masked_noise, np.sum((mask_norms / mask_norms.max())**2)


def compare(matrix):
    """Compare reads of a unitary target at the same peak and integrated pump caps."""
    masked_gain, masked_noise, mask_power = single_hankel_program(matrix, MASKS, N)
    stagger_gain, stagger_noise, stagger_power = single_hankel_program(matrix, FOURIER, N + 1)

    first, second = fixed_pair_program(matrix)
    first_maps, second_maps = first[:, HANKEL], second[:, HANKEL]
    products = second_maps @ first_maps.conj()
    np.testing.assert_allclose(products.sum(axis=0), matrix, atol=2e-13)
    np.testing.assert_allclose(np.linalg.norm(first, axis=1), 1, atol=2e-13)
    pair_norms = np.linalg.norm(second, axis=1)
    pair_peak = pair_norms.max()
    pair_gain = (PAIR_ETA / (PAIR_SLOTS * pair_peak))**2
    diff_nuclear = np.linalg.svd(diagonal_difference(matrix), compute_uv=False).sum()
    pair_bound = (PAIR_ETA / diff_nuclear)**2 if diff_nuclear > 0 else np.inf
    assert pair_gain <= pair_bound * (1 + 1e-12)

    # After the equal-weight collector, trace the independent conjugator vacuum
    # contributions over accepted output modes and slots, then average per tooth.
    divisor = PAIR_SLOTS * N * pair_peak**2
    both_noise = PAIR_ETA**2 * np.sum(np.abs(products)**2) / divisor
    second_noise = PAIR_ETA * np.sum(np.abs(second_maps)**2) / divisor
    row_gain = ETA / N  # A unit-norm row, followed by lossless wavelength retiming.
    pump_pj = PUMP_MW * SLOT_PS * 1e-3 / 2  # One reference or full weight comb/slot.

    result = {
        'diagonal-difference nuclear norm': diff_nuclear,
        'row useful energy gain, dB': power_db(row_gain),
        'masked useful energy gain, dB': power_db(masked_gain),
        'staggered useful energy gain, dB': power_db(stagger_gain),
        'pair useful energy gain, dB': power_db(pair_gain),
        'pair upper bound, dB': power_db(pair_bound),
        'row guide pump energy, pJ': 2 * N * pump_pj,
        'masked guide pump energy, pJ': pump_pj * (N + mask_power),
        'staggered guide pump energy, pJ': pump_pj * (N + stagger_power),
        'pair guide pump energy, pJ': pump_pj * PAIR_PUMP_RATIO * (
            3 * PAIR_SLOTS + np.sum((pair_norms / pair_peak)**2)),
        'row added field error, %': restored_error(row_gain, ETA),
        'masked added field error, %': restored_error(masked_gain, masked_noise),
        'staggered added field error, %': restored_error(stagger_gain, stagger_noise),
        'row error, 10 dB after replay, %': restored_error(10 * row_gain, (ase(10) + 1) * ETA),
        'masked error, 10 dB after replay, %': restored_error(10 * masked_gain, (ase(10) + 1) * masked_noise),
    }
    for gain_db in (0, 10, 20):
        gain = 10**(gain_db / 10)
        noise = gain * both_noise + (ase(gain) + 1) * second_noise
        result[f'pair added field error, {gain_db} dB between guides, %'] = restored_error(
            gain * pair_gain, noise)
    before_second = PAIR_ETA * np.sum(abs(first_maps)**2) / (PAIR_SLOTS * N)
    result['pair signal energy added by 20 dB middle amp, pJ'] = 2 * 99 * before_second
    return result


def check_staggered_read(matrix, vector):
    """Sample the full generated idler family in each 20 ps gate, then align and add."""
    # Time is in units of one comb period; each input copy starts from local time 0.
    samples = (np.arange(512) + 0.5) / 512
    start = np.arange(N) * (1 + 1 / N)
    time = start[:, None] + samples
    assert int(time.max()) == N  # The last partial uses the extra replay period.
    phase = 2j * np.pi
    input_basis = np.exp(phase * (time % 1)[:, :, None] * np.arange(N))
    data = (input_basis @ vector.conj()) / np.sqrt(N + 1)
    weight_basis = np.exp(phase * samples[:, None] * np.arange(2 * N - 1))
    weights = masked_weights(matrix, FOURIER.conj())
    peak = np.linalg.norm(weights, axis=1).max()
    pump = weights @ weight_basis.T / peak
    collected = np.sqrt(ETA / N) * (pump * data.conj()).sum(axis=0)
    target = weight_basis[:, :N] @ (matrix @ vector) * np.sqrt(ETA / (N + 1)) / peak
    error = np.linalg.norm(collected - target) / np.linalg.norm(target)
    print(f'Staggered read, complete ideal gated waveform relative error: {error:.3g}')
    np.testing.assert_allclose(collected, target, atol=2e-12, rtol=2e-12)


def check_staggered_training(matrix, key, error, query, *, advance=1):
    """Check three optical passes on one time grid with whole-comb routing throughout."""
    # One beat period is T. A window starts every (1 + advance/N)T. Midpoint
    # sampling resolves all generated idlers; optical hardware operates on the fields.
    samples_per_period = 512
    assert samples_per_period % N == 0 and np.gcd(advance, N) == 1
    pitch = samples_per_period + advance * (samples_per_period // N)
    window = np.arange(N)[:, None] * pitch + np.arange(samples_per_period)
    periods = N + advance
    local_time = (np.arange(samples_per_period) + 0.5) / samples_per_period
    weight_basis = np.exp(2j * np.pi * np.arange(2 * N - 1)[:, None] * local_time)
    data_basis = weight_basis[:N]
    fourier = FOURIER[(advance * np.arange(N)) % N]
    weights = masked_weights(matrix, fourier.conj())
    learning_rate = 0.01
    ideal_increment = np.outer(error, key.conj())
    ideal_write = masked_weights(ideal_increment, fourier.conj())
    # Reserve one common pump scale for the original and updated reads.
    peak = max(np.linalg.norm(program, axis=1).max()
               for program in (weights, weights + learning_rate * ideal_write))
    pump = weights @ weight_basis / peak
    read_gain = np.sqrt(ETA / periods) / peak

    def input_windows(vector):
        period = vector.conj() @ data_basis
        held = np.tile(period, periods) / np.sqrt(periods)
        return held[window]

    def collect_at_weight_pitch(partials):
        return partials.sum(axis=0) / np.sqrt(N)

    key_windows = input_windows(key)
    forward = collect_at_weight_pitch(np.sqrt(ETA) * pump * key_windows.conj())
    # Equal error copies start on the weight clock. Each generated return window
    # crosses a boundary of the original data clock. Align ALL data-clock periods.
    error_packet = (error @ data_basis) / np.sqrt(N)
    returning = np.zeros(periods * samples_per_period, complex)
    returning[window] = np.sqrt(ETA) * pump * error_packet.conj()
    backward = returning.reshape(periods, samples_per_period).sum(axis=0) / np.sqrt(periods)

    # The FWM write term is input * error * conjugate(reference). A calibrated
    # common write factor accounts for the data split and the stored pump scale.
    write = key_windows * error_packet
    calibrated_write = write * np.sqrt(periods)
    updated_pump = pump + learning_rate * calibrated_write / peak
    updated = collect_at_weight_pitch(np.sqrt(ETA) * updated_pump * input_windows(query).conj())

    expected = (
        ('forward', forward / read_gain, (matrix @ key) @ data_basis),
        ('adjoint', backward / read_gain, (matrix.conj().T @ error).conj() @ data_basis),
        ('increment waveform', calibrated_write, ideal_write @ weight_basis),
        ('post-write query', updated / read_gain, ((matrix + learning_rate * ideal_increment) @ query) @ data_basis),
    )
    for label, actual, target in expected:
        relative_error = np.linalg.norm(actual - target) / np.linalg.norm(target)
        assert relative_error < 2e-12, (label, relative_error)
        print(f'Timing only, {periods} data periods, {label} relative field error: {relative_error:.3g}')


def replay_envelope_errors(matrix):
    """Whole gated-waveform error for all basis inputs with two imperfect replays."""
    samples = (np.arange(512) + 0.5) / 512
    local_time = (np.arange(N)[:, None] / N + samples) % 1
    input_phase = np.exp(-2j * np.pi * np.arange(N)[:, None] * samples)
    weight_basis = np.exp(2j * np.pi * np.arange(2 * N - 1)[:, None] * samples)
    weights = masked_weights(matrix, FOURIER.conj())
    weight_pulses = weights @ weight_basis
    target = matrix.T @ weight_basis[:N]
    # The edge taper runs from zero to unity over 1 ps at each end of every copy.
    distance_ps = np.minimum(local_time, 1 - local_time) * SLOT_PS
    taper = np.sin(np.pi / 2 * np.minimum(distance_ps, 1))**2
    ripple = 1 + 0.05 * np.cos(2 * np.pi * local_time)
    errors = []
    for envelope in (ripple, taper):
        envelope = envelope / np.sqrt(np.mean(envelope**2))
        measured = input_phase * (FOURIER.T @ (weight_pulses * envelope))
        scale = np.vdot(target, measured) / np.vdot(target, target)
        errors.append(100 * np.linalg.norm(measured / scale - target) / np.linalg.norm(target))
    return errors


def main():
    rng = np.random.default_rng(5073)
    readings = []
    envelope_errors = []
    for trial in range(100):
        q, r = np.linalg.qr(rng.normal(size=(N, N)))
        matrix = q * np.sign(np.diag(r))
        readings.append(compare(matrix))
        if trial < 10:
            envelope_errors.append(replay_envelope_errors(matrix))
    print(f'{N} teeth, {SLOT_PS} ps, {SIGNAL_PHOTONS:.0f} input photons per mean tooth')
    print(f'Single guide {PUMP_MW} mW; pair {PUMP_MW * PAIR_PUMP_RATIO:.2f} mW per guide')
    print('100 signed-real orthogonal targets, 10th / 50th / 90th percentiles')
    for name in readings[0]:
        quantiles = np.percentile([row[name] for row in readings], (10, 50, 90))
        print(f'{name:52s} {quantiles[0]:8.2f} {quantiles[1]:8.2f} {quantiles[2]:8.2f}')

    # Universality includes arbitrary complex entries, with two conjugating stages.
    target = rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))
    first, second = fixed_pair_program(target)
    np.testing.assert_allclose((second[:, HANKEL] @ first[:, HANKEL].conj()).sum(axis=0),
                               target, atol=2e-13, rtol=0)
    check_staggered_read(target, rng.normal(size=N) + 1j * rng.normal(size=N))
    key, error, query = (rng.normal(size=(3, N)) + 1j * rng.normal(size=(3, N))) / np.sqrt(2 * N)
    for advance in (1, 17):
        check_staggered_training(target, key, error, query, advance=advance)
    print('Staggered whole-waveform errors over first 10 targets, median %')
    print('5% field ripple / 1 ps rising and falling edges:', np.median(envelope_errors, axis=0))


if __name__ == '__main__':
    main()
