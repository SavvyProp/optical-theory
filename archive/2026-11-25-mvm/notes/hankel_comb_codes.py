"""Finite-packet checks for dense FWM cells. All gain factors are field gains."""

import numpy as np

from hankel_temporal_combiner import N, masked_weights


SAMPLES = 512  # Per 20 ps period, resolving every tooth of the generated family.
TIME = (np.arange(SAMPLES) + 0.5) / SAMPLES
WEIGHT_MODES = np.exp(2j * np.pi * np.arange(2 * N - 1)[:, None] * TIME)
DATA_MODES = WEIGHT_MODES[:N]
SLOTS = np.arange(N)


def code(physical_delays, n=N):
    # A positive delay adds a negative spectral phase. The FWM read conjugates it.
    return np.exp(2j * np.pi * np.outer(physical_delays, np.arange(n)) / n) / np.sqrt(n)


def pulse_delay_and_gate(packets, delay_bins):
    """Delay each finite packet, form two copies with a passive MZI, gate [T,2T).

    The common 45 ps pitch leaves 5 ps beyond 2T. Adjacent slots cannot enter
    the selected windows even when both differential branches are illuminated.
    """
    pitch = 9 * SAMPLES // 4
    first_start = SLOTS * pitch + (delay_bins % N) * (SAMPLES // N)
    field = np.zeros(N * pitch + 3 * SAMPLES, dtype=complex)
    for start, packet in zip(first_start, np.broadcast_to(packets, (N, SAMPLES)), strict=True):
        field[start:start + SAMPLES] += packet / 2
        field[start + SAMPLES:start + 2 * SAMPLES] += packet / 2
    windows = SLOTS[:, None] * pitch + SAMPLES + np.arange(SAMPLES)
    return field[windows]


def collect(packets):
    return packets.sum(axis=0) / np.sqrt(N)


def check(label, measured, expected):
    error = np.linalg.norm(measured - expected) / np.linalg.norm(expected)
    print(f'{label:<40} {error:.2e}')
    assert error < 1e-12


def passive_taps(n=N):
    """Two reciprocal rails, a binary delay lattice, and one static port combiner."""
    if n < 1 or n & (n - 1):
        raise ValueError('The number of slots must be a positive power of two.')
    upper, lower = np.ones(1), np.zeros(1)
    for stage in range(n.bit_length() - 1):
        delay = 2**stage
        short = (upper + lower) / np.sqrt(2)
        long = (upper - lower) / np.sqrt(2)
        upper = np.pad(short, (0, delay))
        lower = np.pad(long, (delay, 0))
    return (upper + lower) / np.sqrt(2)


def mesh(field, stride, taps):
    """Propagate every delayed copy, including paths outside the intended gate."""
    result = np.zeros(len(field) + (len(taps) - 1) * stride, dtype=complex)
    for m, tap in enumerate(taps):
        start = m * stride
        result[start:start + len(field)] += tap * field
    return result


def two_copies(field):
    return (np.pad(field, (0, SAMPLES)) + np.pad(field, (SAMPLES, 0))) / 2


def check_shared_interleaver(matrix, key, error, query, rate):
    pitch = 9 * SAMPLES // 4  # P = 45 ps in band A
    extra = SAMPLES // N      # delta = 0.3125 ps in band B
    window = np.arange(SAMPLES)
    slot_windows = SLOTS[:, None] * pitch + window
    taps = passive_taps()
    np.testing.assert_allclose(abs(taps), 1 / np.sqrt(2 * N))
    masks = code(-(N - 1 - SLOTS))  # The collected code labels the output tooth.
    weights = masked_weights(matrix.T, masks.conj())
    reference = np.sign(taps * taps[::-1])[:, None]
    gain = abs(taps[0])**2 * np.sqrt(N) / 2

    def partial_train(packets):
        train = np.zeros(slot_windows[-1, -1] + 1, dtype=complex)
        train[slot_windows] = packets
        return train

    def data_copies(vector):
        return mesh(vector.conj() @ DATA_MODES, pitch, taps)[slot_windows]

    def coded_error(vector):
        # Launch band B earlier by 63 delta. The old weights keep their addresses.
        train = mesh(two_copies(vector @ DATA_MODES), pitch + extra, taps)
        return train[slot_windows + (N - 1) * extra]

    def read(pump, vector, read_reference=reference):
        raw = partial_train(read_reference * pump * data_copies(vector).conj())
        field = two_copies(mesh(raw, pitch + extra, taps))
        return field[(N - 1) * pitch + SAMPLES + window] / gain

    def adjoint(pump, vector, read_reference=reference):
        raw = partial_train(read_reference * pump * coded_error(vector).conj())
        field = mesh(raw, pitch, taps)
        return field[(N - 1) * pitch + window] / gain

    pump = weights @ WEIGHT_MODES
    delta = np.outer(error, key.conj())
    write = data_copies(key) * coded_error(error) / gain
    expected_increment = masked_weights(delta.T, masks.conj()) @ WEIGHT_MODES
    updated = matrix + rate * delta
    # A read can apply the coupler signs to either the reference or the stored
    # weights. Each physical write must use the same signs as the initial load.
    for label, read_reference, weight_sign in (
        ('shared mesh', reference, 1),
        ('constant read reference', 1, reference),
    ):
        stored = (weight_sign * weights) @ WEIGHT_MODES
        increment = weight_sign * write
        check(f'{label} / forward', read(stored, key, read_reference), (matrix @ key) @ DATA_MODES)
        check(f'{label} / adjoint', adjoint(stored, error, read_reference),
              (matrix.conj().T @ error).conj() @ DATA_MODES)
        check(f'{label} / written weight fields', increment, weight_sign * expected_increment)
        after_write = stored + rate * increment
        check(f'{label} / post-write query', read(after_write, query, read_reference),
              (updated @ query) @ DATA_MODES)
        check(f'{label} / post-write adjoint', adjoint(after_write, error, read_reference),
              (updated.conj().T @ error).conj() @ DATA_MODES)
    # First-order simultaneous pass: keep the transmitted A and weight fields.
    # A perfectly transmitted copy of the saved input is dark at the adjoint gate.
    direct = mesh(partial_train(data_copies(key)), pitch, taps)[(N - 1) * pitch + window]
    np.testing.assert_allclose(direct, 0, atol=1e-12)
    joint_adjoint = direct / gain + adjoint(pump, error)
    check('shared mesh / ideal joint adjoint', joint_adjoint,
          (matrix.conj().T @ error).conj() @ DATA_MODES)
    # The adjoint reference imprints a known slot sign on the write. This
    # check assumes ideal coherent subtraction of the transmitted old weight.
    joint_weight_port = pump + rate * reference.conj() * write
    corrected_increment = reference * (joint_weight_port - pump)
    check('shared mesh / isolated joint write', corrected_increment, rate * expected_increment)
    # One illustrative device perturbation on the weight arm, identical in
    # every slot. Fit a scalar gain so simple attenuation does not count as error.
    edge_ps = np.minimum(TIME, 1 - TIME) * 20
    for rise_ps in (0.5, 1.0):
        envelope = np.sin(np.pi / 2 * np.minimum(edge_ps / rise_ps, 1))**2
        shaped = read(pump * envelope, key)
        target = (matrix @ key) @ DATA_MODES
        scale = np.vdot(target, shaped) / np.vdot(target, target)
        relative = np.linalg.norm(shaped / scale - target) / np.linalg.norm(target)
        print(f'shared mesh / {rise_ps} ps weight edge, fitted-gain error {relative:.2%}')


def check_local_delay(matrix, key, error, query, rate):
    masks = code(SLOTS)
    weights = masked_weights(matrix, masks.conj())
    pump = weights @ WEIGHT_MODES
    error_field = error @ DATA_MODES

    def coded_key(vector):
        # The A-band stores conjugated logical coefficients. Restore the known
        # 1/2 passive field transmission to compare the shapes at unit gain.
        return 2 * pulse_delay_and_gate(vector.conj() @ DATA_MODES, SLOTS)

    def read(pump_field, vector):
        return collect(pump_field * coded_key(vector).conj())

    check('local / forward', read(pump, key), (matrix @ key) @ DATA_MODES)
    return_packets = pump * error_field.conj()
    back = 2 * collect(pulse_delay_and_gate(return_packets, -SLOTS))
    check('local / adjoint', back, (matrix.conj().T @ error).conj() @ DATA_MODES)
    write = coded_key(key) * error_field / np.sqrt(N)
    delta = np.outer(error, key.conj())
    check('local / written weight fields', write, masked_weights(delta, masks.conj()) @ WEIGHT_MODES)
    check('local / post-write query', read(pump + rate * write, query),
          ((matrix + rate * delta) @ query) @ DATA_MODES)


def check_transpose_copy(matrix, key, error, query, rate):
    # On the T-periodic burst the m-th weight window starts 17m/64
    # periods ahead. No backward fold is performed on this continuous burst.
    advance = 17
    masks = code(-advance * SLOTS)
    pitch = SAMPLES + advance * (SAMPLES // N)
    windows = SLOTS[:, None] * pitch + np.arange(SAMPLES)

    def burst_windows(physical_vector):
        return np.tile(physical_vector @ DATA_MODES, N + advance)[windows]

    def pump_for(target):
        return masked_weights(target, masks.conj()) @ WEIGHT_MODES

    def read(pump, physical_vector):
        return collect(pump * burst_windows(physical_vector).conj())

    normal, transposed = pump_for(matrix), pump_for(matrix.T)
    check('copy / forward', read(normal, key.conj()), (matrix @ key) @ DATA_MODES)
    check('copy / adjoint', read(transposed, error), (matrix.conj().T @ error).conj() @ DATA_MODES)
    write = burst_windows(key.conj()) * (error @ DATA_MODES) / np.sqrt(N)
    transpose_write = burst_windows(error) * (key.conj() @ DATA_MODES) / np.sqrt(N)
    delta = np.outer(error, key.conj())
    check('copy / normal weight fields', write, pump_for(delta))
    check('copy / transpose weight fields', transpose_write, pump_for(delta.T))
    updated = matrix + rate * delta
    check('copy / post-write query', read(normal + rate * write, query.conj()),
          (updated @ query) @ DATA_MODES)
    check('copy / post-write adjoint', read(transposed + rate * transpose_write, query),
          (updated.conj().T @ query).conj() @ DATA_MODES)


if __name__ == '__main__':
    rng = np.random.default_rng(527)
    matrix = (rng.normal(size=(N, N)) + 1j * rng.normal(size=(N, N))) / np.sqrt(2 * N)
    key, error, query = (rng.normal(size=(3, N)) + 1j * rng.normal(size=(3, N))) / np.sqrt(2 * N)
    check_shared_interleaver(matrix, key, error, query, 0.1)
    check_local_delay(matrix, key, error, query, 0.1)
    check_transpose_copy(matrix, key, error, query, 0.1)
