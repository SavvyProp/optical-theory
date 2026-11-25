"""Check both Hankel gradients and the earlier fixed-first-factor recall example.

Run with OPENBLAS_NUM_THREADS=1 python3 notes/reduced_hankel_memory.py.
For recall only, the joint least-squares fit is an offline reference and the
online writer sees each key and value once. Both use the same fixed references.
"""

import numpy as np


WIDTH = 64
BRANCHES = 8
SEEDS = (17, 23, 29)


def check_both_factor_gradients():
    """Compare the two convolution writes with loss differences at every tooth."""
    rng = np.random.default_rng(85)
    indices = np.add.outer(np.arange(WIDTH), np.arange(WIDTH))
    a, b = rng.normal(size=(2, BRANCHES, 2 * WIDTH - 1)) / np.sqrt(WIDTH)
    key, value = rng.normal(size=(2, WIDTH)) / np.sqrt(WIDTH)
    z = a[:, indices] @ key
    contributions = np.einsum("bij,bj->bi", b[:, indices], z)
    prediction = contributions.sum(axis=0)
    residual = value - prediction
    backward = b[:, indices].transpose(0, 2, 1) @ residual
    analytic = np.array([
        [np.convolve(key, field) for field in backward],
        [np.convolve(residual, field) for field in z],
    ])

    numerical = np.empty_like(analytic)
    step = 1e-5
    for branch in range(BRANCHES):
        other_branches = prediction - contributions[branch]

        def loss():
            output = b[branch, indices] @ (a[branch, indices] @ key)
            error = value - (other_branches + output)
            return 0.5 * (error @ error)

        for stage, weights in enumerate((a[branch], b[branch])):
            for tooth, original in enumerate(weights):
                weights[tooth] = original - step
                lower = loss()
                weights[tooth] = original + step
                upper = loss()
                weights[tooth] = original
                numerical[stage, branch, tooth] = (lower - upper) / (2 * step)
    relative_errors = [
        np.linalg.norm(write - reference) / np.linalg.norm(reference)
        for write, reference in zip(analytic, numerical)
    ]
    assert max(relative_errors) < 1e-7, relative_errors
    return relative_errors


def associations(rng, count):
    keys, _ = np.linalg.qr(rng.normal(size=(WIDTH, count)), mode="reduced")
    values = rng.normal(size=(count, WIDTH))
    values /= np.linalg.norm(values, axis=1, keepdims=True)
    return keys.T, values


def read_design(references, keys):
    """Each block maps the stacked writable comb coefficients to W @ key."""
    teeth = 2 * WIDTH - 1
    intermediate = np.einsum("bij,tj->tbi", references, keys)
    design = np.zeros((len(keys), WIDTH, BRANCHES, teeth))
    for i in range(WIDTH):
        design[:, i, :, i:i + WIDTH] = intermediate
    return design.reshape(len(keys), WIDTH, BRANCHES * teeth)


def online_error(design, values, step):
    coefficients = np.zeros(design.shape[-1])
    for read, value in zip(design, values):
        error = value - read @ coefficients
        coefficients += step * (read.T @ error)
    return np.linalg.norm(design @ coefficients - values) / np.linalg.norm(values)


def evaluate(seed):
    rng = np.random.default_rng(seed)
    indices = np.add.outer(np.arange(WIDTH), np.arange(WIDTH))
    weights = rng.normal(size=(BRANCHES, 2 * WIDTH - 1)) / np.sqrt(WIDTH)
    references = weights[:, indices]
    steps = np.array([0.125, 0.25, 0.375, 0.5, 0.75, 1.0]) / BRANCHES
    calibration_error = np.zeros(len(steps))
    for _ in range(8):
        keys, values = associations(rng, 12)
        design = read_design(references, keys)
        calibration_error += [online_error(design, values, step)**2 for step in steps]
    step = steps[calibration_error.argmin()]

    keys, values = associations(rng, 32)
    design = read_design(references, keys)
    # Check the physical convolution against the coefficient gradient.
    residual = values[0]
    convolutions = [np.convolve(residual, ref @ keys[0]) for ref in references]
    np.testing.assert_allclose(np.ravel(convolutions), design[0].T @ residual, atol=1e-12)
    results = []
    for count in (8, 12, 32):
        matrix = design[:count].reshape(count * WIDTH, -1)
        target = values[:count].ravel()
        fit, _, _, _ = np.linalg.lstsq(matrix, target, rcond=None)
        joint = np.linalg.norm(matrix @ fit - target) / np.linalg.norm(target)
        online = online_error(design[:count], values[:count], step)
        results.append((joint, online))
    return results


def lag_attention_example():
    """Exact decayed linear attention for seven streams of tapped-delay features."""
    rng = np.random.default_rng(420)
    streams, decay, n = BRANCHES - 1, 0.98, WIDTH
    indices = np.add.outer(np.arange(n), np.arange(n))
    value, key = np.zeros((streams, n)), np.zeros((streams, n))
    lag_sum = np.zeros(2 * n - 1)
    dense = np.zeros((n, n))
    max_error = 0.
    for token in range(1, 513):
        # Entries that fall off the two lag vectors update the Toeplitz remainder.
        negative = key[:, -1] @ value[:, :-1]
        nonnegative = value[:, -1] @ key[:, ::-1]
        lag_sum = decay * (lag_sum + np.r_[negative, nonnegative])
        for history in (value, key):
            history[:, 1:] = np.sqrt(decay) * history[:, :-1]
            history[:, 0] = rng.normal(size=streams)
        dense = decay * dense + value.T @ key
        if token in (1, 16, 64, 128, 256, 512):
            first = np.pad(key, ((0, 0), (0, n-1)))[:, indices]
            second = np.pad(value, ((0, 0), (0, n-1)))[:, indices]
            bank = (second @ first).sum(axis=0) + lag_sum[indices][:, ::-1]
            max_error = max(max_error, np.linalg.norm(bank - dense) / np.linalg.norm(dense))
    assert max_error < 1e-12
    return max_error, np.linalg.matrix_rank(dense)


if __name__ == "__main__":
    first, second = check_both_factor_gradients()
    print(f"Relative gradient error with both factors writable: first {first:.2e}, second {second:.2e}")
    error, rank = lag_attention_example()
    print(f"Seven-stream lag-feature attention over 512 tokens: relative error {error:.2e}, final rank {rank}")
    errors = np.median([evaluate(seed) for seed in SEEDS], axis=0)
    print("Recall with the first factors fixed")
    print("Associations   Joint fit    One-pass TTT")
    for count, (joint, online) in zip((8, 12, 32), errors):
        print(f"{count:12d}   {joint:9.2e}   {online:12.3f}")
