"""Global 1,024-dimensional associative writes with three optical banks.

Run OPENBLAS_NUM_THREADS=1 python3 notes/global_monarch_writes.py.
Signed-real, calibrated field transfer; excludes losses, gain noise and timing.
"""

import numpy as np

from hankel_forward_write_recall import local_delta_fields
from monarch_association_write import fixed_bank, route


LANES, WIDTH = 8, 128
DIM = LANES * WIDTH
ACTIVE = 120
GROUPS = ACTIVE // LANES
HANKEL = np.add.outer(np.arange(WIDTH), np.arange(WIDTH))


def middle_bank():
    # Groups modulo 15 intersect every route residue modulo 8 exactly once.
    h = ((np.arange(2 * WIDTH - 1) % GROUPS) == 0).astype(float) / LANES
    matrix = h[HANKEL]
    matrix[ACTIVE:, :] = 0  # Contiguous output guard.
    matrix[:, ACTIVE:] = 0  # Contiguous input guard.
    return np.broadcast_to(matrix, (LANES, WIDTH, WIDTH))


def feature_indices(lane, outgoing):
    teeth = np.arange(ACTIVE)
    if outgoing:
        # First bank emitting toward the hub.
        return ((lane + teeth) % LANES) * GROUPS + teeth % GROUPS
    # Last bank receiving from the hub; a Hankel reverses the residue mod 15.
    return ((lane - teeth) % LANES) * GROUPS + (-teeth) % GROUPS


def packed_read(keys, values, queries):
    """Ideal independently programmable endpoint rows and columns."""
    first = np.zeros((LANES, WIDTH, WIDTH))
    last = np.zeros_like(first)
    for lane in range(LANES):
        coords = slice(lane * WIDTH, (lane + 1) * WIDTH)
        first[lane, :ACTIVE] = keys[coords][:, feature_indices(lane, True)].T
        last[lane, :, :ACTIVE] = values[coords][:, feature_indices(lane, False)]
    at_hub = route(first, queries)
    return (last @ route(middle_bank(), at_hub)).reshape(DIM, -1)


def streamed_record(key, value, queries):
    """One directly stored key and value, with a single selected dot per input lane."""
    first = np.zeros((LANES, WIDTH, WIDTH))
    last = np.zeros_like(first)
    for lane in range(LANES):
        coords = slice(lane * WIDTH, (lane + 1) * WIDTH)
        tooth = -lane % LANES
        h = np.zeros(2 * WIDTH - 1)
        h[tooth:tooth + WIDTH] = key[coords]
        # A single-tooth filter after the input Hankel keeps the complete dot.
        first[lane, tooth] = h[HANKEL[tooth]]
        h.fill(0)
        h[lane:lane + WIDTH] = value[coords]
        last[lane] = h[HANKEL]
    at_hub = route(first, queries)
    # Only hub 0 is illuminated. The flat first 15 Hankel weights sum the
    # first eight teeth; select its first eight outputs before the shuffle.
    h = np.zeros(2 * WIDTH - 1)
    h[:2 * LANES - 1] = 1 / LANES
    middle = np.zeros_like(first)
    middle[0, :LANES] = h[HANKEL[:LANES]]
    return (last @ route(middle, at_hub)).reshape(DIM, -1)


def check_transfers():
    rng = np.random.default_rng(391)
    keys = rng.normal(size=(DIM, ACTIVE)) / np.sqrt(DIM)
    values = rng.normal(size=(DIM, ACTIVE)) / np.sqrt(DIM)
    queries = rng.normal(size=(DIM, 12))
    worst = 0.0
    for _ in range(2):
        expected = values @ (keys.T @ queries)
        actual = LANES * packed_read(keys, values, queries)
        worst = max(worst, np.linalg.norm(actual - expected) / np.linalg.norm(expected))
        values[:, 17] = rng.normal(size=DIM) / np.sqrt(DIM)
    expected = np.outer(values[:, 0], keys[:, 0] @ queries)
    actual = LANES * streamed_record(keys[:, 0], values[:, 0], queries)
    one_err = np.linalg.norm(actual - expected) / np.linalg.norm(expected)
    assert max(worst, one_err) < 1e-12
    # Equal and opposite lane scores must cancel at every output wavelength.
    key = np.zeros(DIM)
    key[[0, WIDTH]] = 1 / np.sqrt(2)
    query = key.copy()
    query[WIDTH] *= -1
    np.testing.assert_allclose(streamed_record(key, values[:, 0], query), 0, atol=1e-14)
    print(f"One physical Hankel record: {one_err:.2e}; 120 packed records with "
          f"direct outer-bank rows and columns, before/after replacement: {worst:.2e}")


def shuffle_mean(partials, radices):
    """All lanes contribute to the same field under the cyclic AWGR port rule."""
    if np.prod(radices) != len(partials):
        raise ValueError("The product of the shuffle radices must equal the lane count.")
    fields = partials
    stride = 1
    for radix in radices:
        # A copy on a tooth of residue b*stride travels to lane j+b*stride.
        fields = sum(np.roll(fields, b * stride, axis=0) for b in range(radix)) / radix
        stride *= radix
    return fields


def routed_hankel_record(key, value, queries, radices):
    """Literal Hankel frequency fanout, cyclic shuffles and masked coherent sums."""
    if np.prod(radices) != LANES:
        raise ValueError("The score exchanges must cover all lanes.")
    fields = queries.reshape(LANES, WIDTH, -1)
    # A row at tooth zero gives a complete 128-component local dot.
    dots = np.einsum('lw,lwq->lq', key.reshape(LANES, WIDTH), fields)
    fields = np.zeros_like(fields)
    fields[:, 0] = dots
    incoming = np.array([0])
    stride = 1
    # The last (unshuffled) step collects the last copies into a single tooth.
    for stage, radix in enumerate((*radices, 1)):
        outgoing = stride * np.arange(radix)
        h = np.zeros(2 * WIDTH - 1)
        h[np.add.outer(incoming, outgoing)] = 1 / np.sqrt(len(incoming) * len(outgoing))
        matrix = np.zeros((WIDTH, WIDTH))
        matrix[outgoing] = h[HANKEL[outgoing]]
        bank = np.broadcast_to(matrix, (LANES, WIDTH, WIDTH))
        if stage < len(radices):
            fields = route(bank, fields)
        else:
            fields = bank @ fields
        incoming = outgoing
        stride *= radix
    # A value comb in the first column synthesizes an arbitrary local value
    # vector from the single occupied tooth, simultaneously on all lanes.
    return (value.reshape(LANES, WIDTH, 1) * fields[:, :1]).reshape(DIM, -1)


def check_global_records():
    rng = np.random.default_rng(183)
    keys = rng.normal(size=(20, DIM))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    keys[-1] = keys[0]
    values = rng.normal(size=keys.shape)
    queries = rng.normal(size=(DIM, 4))
    # The residual log and dense delta give identical reads while all records live.
    W = np.zeros((DIM, DIM))
    stored = []
    for k, v in zip(keys, values):
        old = sum((e * (previous @ k) for previous, e in stored), np.zeros(DIM))
        np.testing.assert_allclose(old, W @ k, atol=1e-12)
        stored.append((k, v - old))
        W += np.outer(v - W @ k, k)
    expected = W @ queries
    for radices in ((8,), (2, 4), (2, 2, 2), (1, 2, 2, 2)):
        optical = routed_hankel_record(keys[0], values[0], queries, radices)
        np.testing.assert_allclose(LANES * optical, np.outer(values[0], keys[0] @ queries), atol=1e-12)
        read = np.zeros_like(expected)
        for k, e in stored:
            partials = np.einsum('lw,lwq->lq', k.reshape(LANES, WIDTH),
                                 queries.reshape(LANES, WIDTH, -1))
            # Correct the calibrated 1/8 field factor explicitly.
            scores = LANES * shuffle_mean(partials, radices)
            np.testing.assert_allclose(scores, np.broadcast_to(k @ queries, scores.shape), atol=1e-12)
            read += (e.reshape(LANES, WIDTH, 1) * scores[:, None, :]).reshape(DIM, -1)
        relative = np.linalg.norm(read - expected) / np.linalg.norm(expected)
        assert relative < 1e-12
        print(f"Radices {radices}, full delta vs 20 stored residuals: {relative:.2e}")


def check_additive_attention():
    """Separate optical arithmetic error from interference among stored keys."""
    rng = np.random.default_rng(20261125)
    for geometry in ("orthogonal", "independent"):
        keys = rng.normal(size=(DIM, 64))
        if geometry == "orthogonal":
            keys = np.linalg.qr(keys, mode="reduced")[0]
        else:
            keys /= np.linalg.norm(keys, axis=0)
        values = rng.normal(size=keys.shape)
        values /= np.linalg.norm(values, axis=0)
        probes = np.column_stack((keys, rng.normal(size=(DIM, 8)) / np.sqrt(DIM)))
        output = np.zeros_like(probes)
        for k, v in zip(keys.T, values.T):
            output += LANES * routed_hankel_record(k, v, probes, (2, 2, 2))
        reference = values @ (keys.T @ probes)
        arithmetic = np.linalg.norm(output - reference) / np.linalg.norm(reference)
        retrieval = np.linalg.norm(output[:, :64] - values) / np.linalg.norm(values)
        assert arithmetic < 1e-12
        if geometry == "orthogonal":
            assert retrieval < 1e-12
        print(f"64 {geometry} keys: additive arithmetic error {arithmetic:.2e}, "
              f"stored-value retrieval RMS {retrieval:.3%}", flush=True)


def multiplicative_step(reads, address, value):
    p = reads[:, address].copy()
    partials = np.einsum('lw,lwq->lq', p.reshape(LANES, WIDTH),
                         reads.reshape(LANES, WIDTH, -1))
    scores = LANES * shuffle_mean(partials, (2, 2, 2)) / (p @ p)
    e = value - p
    change = (e.reshape(LANES, WIDTH, 1) * scores[:, None, :]).reshape(reads.shape)
    reads += change
    return change


def multiplicative_writes(count, repeats, trials=64):
    """Left-append a single global rank-one correction and retain every factor."""
    rng = np.random.default_rng(15101 + count + repeats)
    total_error = total_value = other_change = intended_change = 0.0
    worst_immediate = 0.0
    for _ in range(trials):
        keys = np.linalg.qr(rng.normal(size=(DIM, count)), mode="reduced")[0]
        reads = keys.copy()  # Start at W=I, preserve only the probes Wk in this test.
        targets = np.zeros_like(keys)
        for index, address in enumerate(np.r_[np.arange(count), rng.integers(count, size=repeats)]):
            value = rng.normal(size=DIM)
            value /= np.linalg.norm(value)
            change = multiplicative_step(reads, address, value)
            targets[:, address] = value
            worst_immediate = max(worst_immediate, np.linalg.norm(reads[:, address] - value))
            if index >= count:
                other_change += np.sum(change[:, np.arange(count) != address]**2) / (count - 1)
                intended_change += np.sum(change[:, address]**2)
        total_error += np.sum((reads - targets)**2)
        total_value += np.sum(targets**2)
    assert worst_immediate < 1e-12
    print(f"{count} keys + {repeats} overwrites, global product: immediate {worst_immediate:.2e}, "
          f"final bank RMS {np.sqrt(total_error/total_value):.1%}, "
          f"other/target RMS change {np.sqrt(other_change/intended_change):.1%}", flush=True)


def inserts_only(trials=64):
    """Compare identical one-write streams by reading all stored analog values."""
    checkpoints = (4, 8, 16, 32, 64)
    count = max(checkpoints)
    for geometry in ("orthogonal", "independent"):
        rng = np.random.default_rng(261124)
        errors = np.zeros((len(checkpoints), 3))
        for _ in range(trials):
            keys = rng.normal(size=(DIM, count))
            if geometry == "orthogonal":
                keys = np.linalg.qr(keys, mode="reduced")[0]
            else:
                keys /= np.linalg.norm(keys, axis=0)
            values = rng.normal(size=(DIM, count))
            values /= np.linalg.norm(values, axis=0)
            gram = keys.T @ keys
            product = keys.copy()  # W_0 = I
            delta = keys.copy()    # Same initial W for the dense delta control.
            for index, value in enumerate(values.T):
                multiplicative_step(product, index, value)
                delta += np.outer(value - delta[:, index], gram[index] / gram[index, index])
                live = index + 1
                if live in checkpoints:
                    target = values[:, :live]
                    additive = target @ gram[:live, :live]  # S_0 = 0
                    for method, result in enumerate((product[:, :live], additive, delta[:, :live])):
                        errors[checkpoints.index(live), method] += np.sum((result - target)**2)
        for live, row in zip(checkpoints, errors):
            rms = np.sqrt(row / (trials * live))
            print(f"{geometry}, {live} inserts: product {rms[0]:.2%}, additive {rms[1]:.2%}, "
                  f"dense delta from identity {rms[2]:.2%}", flush=True)
            if geometry == "orthogonal":
                assert max(rms[1:]) < 1e-12


def dense_delta(code, values, repeated):
    predictions = np.zeros_like(values[:, :-1])
    for index, key_index in enumerate([*range(code.shape[1]), repeated]):
        k = code[:, key_index]
        residual = values[:, index] - predictions[:, key_index]
        predictions += np.outer(residual, k @ code / (k @ k))
    return predictions


def test_writable_hankels(count, trials=128):
    bank = fixed_bank(np.random.default_rng(399), LANES, WIDTH)
    middle = middle_bank()
    rng = np.random.default_rng(241130 + count)
    stats = np.zeros((2, 4))
    for _ in range(trials):
        keys = np.linalg.qr(rng.normal(size=(DIM, count)), mode="reduced")[0]
        values = rng.normal(size=(DIM, count + 1))
        values /= np.linalg.norm(values, axis=0)
        address = rng.integers(count)
        # Fixed amplitude calibration restores mean per-lane key power.
        codes = np.sqrt(DIM / ACTIVE) * route(middle, route(bank, keys))
        canonical = codes[0, np.argsort(feature_indices(0, False))]
        output = np.empty((LANES, WIDTH, count))
        for lane, local in enumerate(codes):
            np.testing.assert_allclose(local[np.argsort(feature_indices(lane, False))],
                                       canonical, atol=1e-13)
            writes = np.vstack((local.T, local[:, address]))
            target = values[lane * WIDTH:(lane + 1) * WIDTH].T
            output[lane] = local_delta_fields(writes, target, rates=(0.05,))[0, :, :count]
        other = np.arange(count) != address
        for score, prediction in zip(stats, (dense_delta(canonical, values, address),
                                            output.reshape(DIM, count))):
            winner = (values.T @ prediction).argmax(axis=0)
            score[0] += winner[address] == count
            score[1] += np.mean(winner[other] == np.arange(count)[other])
            score[2] += values[:, -1] @ prediction[:, address] / np.linalg.norm(prediction[:, address])
            score[3] += winner[address] == address
    for label, row in zip(("dense projected delta", "next-factor Hankel"), stats / trials):
        print(f"{count} keys, {label}: new top-1 {row[0]:.1%}, others {row[1]:.1%}, "
              f"new value cosine {row[2]:.3f}, stale {row[3]:.1%}", flush=True)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--inserts-only", action="store_true", help="compare unique-key writes with linear attention")
    if parser.parse_args().inserts_only:
        inserts_only()
        raise SystemExit
    check_transfers()
    check_global_records()
    check_additive_attention()
    for live, rewrites in ((16, 1), (32, 1), (32, 32), (64, 1)):
        multiplicative_writes(live, rewrites)
    for count in (16, 32):
        test_writable_hankels(count)
