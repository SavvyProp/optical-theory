"""Check a forward-written linear-attention memory using a guarded FWM comb.

Run python3 notes/hankel_guarded_product.py. Coordinates are complex optical
fields on ideal frequency bins. Each guide visit computes H(h) conj(x), with
H[i,j] = h[i+j]. The model assumes unit central-tooth field gain, flat FWM
transfer, exact passband masks and calibrated delays. It excludes finite pulse
envelopes, loss, noise, saturation, unwanted mixing families and write optics.
"""

import numpy as np


def encode_dense(matrix):
    """Initialize physical pump packets by Fourier-coding the matrix diagonals."""
    width = len(matrix)
    entries = np.zeros((width, 2 * width - 1), complex)
    for i, j in np.ndindex(matrix.shape):
        entries[j, i - j + width - 1] = matrix[i, j]
    coeffs = np.fft.fft(entries, axis=0) / width
    n = 3 * width - 1
    center = n - 1
    offsets = np.arange(width, n)
    pumps = np.zeros((width, 2*n - 1), complex)
    pumps[:, center] = 1
    for layer in range(width):
        if layer % 2 == 0:
            pumps[layer, center - offsets] = coeffs[layer].conj()
        else:
            pumps[layer, center + offsets] = coeffs[layer]
    return pumps


def add_outer_product(program, value, key, rate, codes):
    """Add physical FWM products directly into the resident pump packets."""
    width = len(key)
    sideband_length = 2 * width - 1
    for layer, phase in enumerate(codes):
        coded_key = phase * key
        # On each parity the old input and output-error carriers mix with the
        # conjugated reference. Their frequency indices add. The opposite grid
        # orientation on one of those carriers supplies the logical correlation.
        if layer % 2 == 0:
            update = np.convolve(coded_key, value[::-1].conj())
            program[layer, :sideband_length] += rate * update / width
        else:
            update = np.convolve(coded_key[::-1].conj(), value)
            program[layer, -sideband_length:] += rate * update / width


def physical_read(program, query, tooth_phase, mask_guard=True):
    width = len(query)
    n = 3 * width - 1
    # Logical inputs occupy the low band and sums occupy the high band.
    field = np.zeros(n, complex)
    field[:width] = query
    for layer, pump in enumerate(program):
        field = np.convolve(pump, field[::-1].conj(), mode='valid')
        if mask_guard:
            field[width:n-width] = 0
        # Code only the query after it has contributed to this visit's sum.
        if layer % 2 == 0:
            field[-width:] *= tooth_phase[::-1].conj()
        else:
            field[:width] *= tooth_phase
    if len(program) % 2:
        field = field[::-1].conj()
    return field[-width:]


def norm_error(actual, reference):
    return np.linalg.norm(actual - reference) / np.linalg.norm(reference)


def check(width, seed):
    rng = np.random.default_rng(seed)

    def random_field(shape):
        return rng.standard_normal(shape) + 1j * rng.standard_normal(shape)

    tooth_phase = np.exp(2j * np.pi * np.arange(width) / width)
    codes = tooth_phase[None, :] ** np.arange(width)[:, None]
    initial = random_field((width, width)) / np.sqrt(2 * width)
    program = encode_dense(initial)
    # Applying the optical path to every coordinate tests the full read matrix.
    realized = np.column_stack([physical_read(program, q, tooth_phase)
                                for q in np.eye(width)])
    read_error = norm_error(realized, initial)
    q = random_field(width)
    no_mask_error = norm_error(physical_read(program, q, tooth_phase, False), initial @ q)

    state = initial.copy()
    keys = random_field((min(width, 12), width))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    rates = rng.choice([0.2, 0.6, 1.0], size=2 * len(keys))
    indices = np.concatenate([np.arange(len(keys)), rng.permutation(len(keys))])
    worst_prediction = 0.0
    for index, rate in zip(indices, rates):
        key = keys[index]
        value = random_field(width) / np.sqrt(2 * width)
        optical_prediction = physical_read(program, key, tooth_phase)
        add_outer_product(program, value - optical_prediction, key, rate, codes)
        state += rate * np.outer(value - state @ key, key.conj())
        probe = random_field(width)
        measured = physical_read(program, probe, tooth_phase)
        worst_prediction = max(worst_prediction, norm_error(measured, state @ probe))
    final_matrix = np.column_stack([physical_read(program, q, tooth_phase)
                                    for q in np.eye(width)])
    state_error = norm_error(final_matrix, state)
    assert max(read_error, worst_prediction, state_error) < 1e-10
    print(f'd={width:2d}  writes={len(indices):2d}  initial matrix={read_error:.2e}  '
          f'worst post-write query={worst_prediction:.2e}  final matrix={state_error:.2e}  '
          f'unmasked initial probe={no_mask_error:.3f}')


if __name__ == '__main__':
    for d in (8, 16, 64):
        check(d, seed=3820 + d)
