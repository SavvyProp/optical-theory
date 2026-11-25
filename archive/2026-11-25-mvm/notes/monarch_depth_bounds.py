"""Verify local matrix freedom and an ideal dense-block Monarch construction.

Run OPENBLAS_NUM_THREADS=1 python3 notes/monarch_depth_bounds.py.
The 1,024-mode check counts matrix-unit directions without building a Jacobian.
The numerical construction uses the same eight-port router with eight teeth
per lane so a complete reconstruction can run quickly. Each lane can apply an
arbitrary real block. No FWM factorization, power limit, or noise is modeled.
"""
import numpy as np
from scipy.linalg import lu
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import maximum_bipartite_matching


LANES = 8


def gf8_times(left, right):
    """Multiply two three-bit field elements modulo x^3 + x + 1."""
    product = np.zeros_like(right)
    for _ in range(3):
        if left & 1:
            product ^= right
        right = (right << 1) ^ ((right >> 2) * 0b1011)
        left >>= 1
    return product


def parallel_classes(first, last):
    """Nine partitions of the 64 bundles, with the requested endpoints."""
    return [first] + [first ^ gf8_times(slope, last) for slope in range(1, LANES)] + [last]


def check_route(source, dest):
    counts = np.bincount(LANES * source + dest, minlength=LANES**2)
    np.testing.assert_array_equal(counts, np.full(LANES**2, len(source) // LANES**2))


def check_directions():
    # Modes are grouped as (x, y, within-bundle). At an all-permutation
    # setting a local weight derivative is a global matrix unit, up to a
    # common output permutation. Count their union in input coordinates.
    first, last, _ = np.indices((LANES, LANES, 16)).reshape(3, -1)
    groups = parallel_classes(first, last)
    for i, group in enumerate(groups):
        for earlier in groups[:i]:
            check_route(earlier, group)
    directions = np.logical_or.reduce([np.equal.outer(group, group) for group in groups])
    assert directions.all()
    return np.count_nonzero(directions)


def middle_partition(first, last):
    """Group 16 (one in the small test) perfect matchings per middle lane."""
    n = len(first)
    counts = np.bincount(LANES * first + last, minlength=LANES**2).reshape(LANES, LANES)
    middle = np.full(n, -1)
    for match_number in range(n // LANES):
        partners = maximum_bipartite_matching(csr_matrix(counts), perm_type="column")
        assert np.all(partners >= 0)
        for i, j in enumerate(partners):
            mode = np.flatnonzero((first == i) & (last == j) & (middle == -1))[0]
            middle[mode] = match_number // (n // LANES**2)
            counts[i, j] -= 1
    check_route(first, middle)
    check_route(middle, last)
    return middle


def product(factors):
    result = factors[0]
    for factor in factors[1:]:
        result = factor @ result
    return result


def triangular_factors(target, groups, upper):
    """Each direct entry can correct all paths through smaller index gaps."""
    n = len(target)
    factors = [np.eye(n) for group in groups]
    owner = np.full((n, n), -1)
    for i, group in enumerate(groups):
        owner[np.equal.outer(group, group)] = i
    assert (owner >= 0).all()
    for gap in range(1, n):
        current = product(factors)
        for low in range(n - gap):
            i, j = (low, low + gap) if upper else (low + gap, low)
            factors[owner[i, j]][i, j] = target[i, j] - current[i, j]
    np.testing.assert_allclose(product(factors), target, rtol=1e-10, atol=1e-10)
    return factors


def physical_product(factors, groups, input_order, output_order):
    """Program the lane blocks, then route physical tooth j to lane (i+j)%8."""
    n = len(input_order)
    width = n // LANES
    incoming = np.argsort(input_order).reshape(LANES, width)
    response = np.eye(n)
    teeth = np.arange(width)
    lanes = np.arange(LANES)[:, None]
    for step, (factor, group) in enumerate(zip(factors, groups)):
        np.testing.assert_array_equal(group[incoming], np.broadcast_to(lanes, incoming.shape))
        assert np.all(factor[np.not_equal.outer(group, group)] == 0)
        last_step = step + 1 == len(groups)
        if last_step:
            outgoing = np.argsort(output_order).reshape(LANES, width)
        else:
            next_group = groups[step + 1]
            check_route(group, next_group)
            outgoing = np.empty_like(incoming)
            for i in range(LANES):
                for j in range(LANES):
                    outgoing[i, (teeth + i) % LANES == j] = np.flatnonzero((group == i) & (next_group == j))
        after = np.empty((LANES, width, n))
        for lane in range(LANES):
            block = factor[np.ix_(outgoing[lane], incoming[lane])]
            after[lane] = block @ response[lane * width:(lane + 1) * width]
        if last_step:
            return after.reshape(n, n)
        destinations = (lanes + teeth) % LANES
        incoming[destinations, teeth] = outgoing
        routed = np.empty_like(after)
        routed[destinations, teeth] = after
        response = routed.reshape(n, n)


def check_reconstruction():
    rng = np.random.default_rng(3817)
    width = 8
    n = LANES * width
    target = rng.normal(size=(n, n))
    # The permutations give logical-to-physical socket labels at each end.
    input_order = rng.permutation(n)
    permutation, lower, upper = lu(target[:, input_order])
    output_order = np.argmax(permutation, axis=0)
    diagonal = np.diag(upper).copy()
    upper = upper / diagonal[:, None]
    first = input_order // width
    last = output_order // width
    middle = middle_partition(first, last)
    up_groups = parallel_classes(first, middle)
    low_groups = parallel_classes(middle, last)
    up = triangular_factors(upper, up_groups, True)
    low = triangular_factors(lower, low_groups, False)
    groups = up_groups + low_groups[1:]
    assert len(groups) == 17
    for singular in (False, True):
        if singular:
            diagonal[n//2:] = 0
            target[np.ix_(output_order, input_order)] = (lower * diagonal) @ upper
        center = (low[0] * diagonal) @ up[-1]
        factors = up[:-1] + [center] + low[1:]
        assembled = physical_product(factors, groups, input_order, output_order)
        error = np.linalg.norm(assembled - target) / np.linalg.norm(target)
        assert error < 1e-11
        print(f"{n} modes, 17 banks, rank {np.linalg.matrix_rank(target)}, "
              f"relative Frobenius error {error:.3g}")


if __name__ == "__main__":
    print(f"1,024 modes, nine banks, {check_directions():,} distinct matrix-unit directions")
    check_reconstruction()
