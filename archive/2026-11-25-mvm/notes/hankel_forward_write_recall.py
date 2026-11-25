"""Associative recall after forward-only writes to a product of real Hankel factors.

Run with OPENBLAS_NUM_THREADS=1 python3 notes/hankel_forward_write_recall.py.
Use --local-delta for a second test that computes the newest factor's gradient
from its input and the current prediction error as the 34 slots first fill.
Use --rewrite --width 128 to compare overwrite capacity with the width-64 default.
Keys are random orthonormal columns, values independent unit vectors. Each write
replaces one central reversal J by J + alpha H(h), in chronological read order.
The token packet h is the linear convolution of a value and a key, with one
operand's tooth order reversed according to slot parity. This favorable coding
assumes two parity-specific encodings or a way to reverse spectral order.
The optical read returns Mq. Scoring (Mq-q)/alpha assumes coherent access to
an aligned copy of q and ideal gain calibration. All optics are noiseless.
This is an untrained write rule. Its performance measures this encoding and
initialization, with no optimization of the factor biases or token projections.
"""

import argparse
from functools import lru_cache

import numpy as np


WIDTH = 64
STREAMS = 128
ALPHAS = np.array([0.01, 0.05, 0.10, 0.20, 0.50, 1.00])
LOCAL_RATES = np.array([0.05, 0.10, 0.20, 0.40, 0.80, 1.00])

@lru_cache(maxsize=None)
def toeplitz_indices(width):
    return np.subtract.outer(np.arange(width), np.arange(width)) + width - 1


def recall_fields(keys, values, alphas=ALPHAS, depth=34):
    """Probe all keys after replacing at most the newest depth slots."""
    indices = toeplitz_indices(keys.shape[1])
    queries = keys.T.copy()
    fields = np.repeat(queries[None, :, :], len(alphas), axis=0)
    first_order = np.zeros_like(queries)
    for key, value in zip(keys[-depth:], values[-depth:]):
        # In coordinates that undo the J reversals, both parities implement
        # the same I + alpha D. D[i,j] sums v[a] k[b] over a-b = i-j.
        perturbation = np.correlate(value, key, mode="full")[indices]
        first_order += perturbation @ queries
        fields += alphas[:, None, None] * (perturbation @ fields)
    residual = (fields - queries[None, :, :]) / alphas[:, None, None]
    return np.concatenate([first_order[None, :, :], residual], axis=0)


def local_delta_fields(keys, values, rates=LOCAL_RATES):
    """Use the old product's current key prediction to write its new last factor.

    In reversal-corrected coordinates, the newest slot initially acts as I.
    Its input is x=M_old k. The target for M_new k is k+v because the memory
    read subtracts the query. The derivative of 0.5 ||(I+D)x-(k+v)||^2 with
    respect to Toeplitz lag l is -sum_{i-j=l} error[i]*x[j] at D=0.
    We grant the writer a copy of x, a coherent residual and scalar 1/||x||^2.
    These require physical resources in addition to the separate FWM write.
    Only the current key's column of reads participates in computing its update.
    """
    indices = toeplitz_indices(keys.shape[1])
    outputs = []
    for rate in rates:
        reads = keys.T.copy()
        for i, (key, value) in enumerate(zip(keys, values)):
            prediction = reads[:, i]
            error = key + value - prediction
            taps = np.correlate(error, prediction, mode="full") / (prediction @ prediction)
            reads += rate * (taps[indices] @ reads)
        outputs.append(reads - keys.T)
    return np.array(outputs)


def verify_factor_order(width, depth):
    """Compare the accelerated read with the literal Hankel optical program."""
    rng = np.random.default_rng(1034)
    h_index = np.add.outer(np.arange(width), np.arange(width))
    reversal = np.eye(width)[::-1]
    alpha_index = 2
    alpha = ALPHAS[alpha_index]
    for writes in (7, depth, width):
        keys = np.linalg.qr(rng.normal(size=(width, writes)))[0].T
        values = rng.normal(size=(writes, width))
        values /= np.linalg.norm(values, axis=1, keepdims=True)
        active_keys, active_values = keys[-depth:], values[-depth:]
        literal = keys.T.copy()
        for slot in range(depth):
            if slot >= len(active_keys):
                literal = reversal @ literal
                continue
            key, value = active_keys[slot], active_values[slot]
            # At even depth, reverse the output basis on even slots and the
            # input basis on odd slots to account for the remaining reversals.
            if slot % 2 == 0:
                value = value[::-1]
            else:
                key = key[::-1]
            taps = np.convolve(value, key)
            literal = (reversal + alpha * taps[h_index]) @ literal
        expected = recall_fields(keys, values, depth=depth)[1 + alpha_index]
        np.testing.assert_allclose((literal - keys.T) / alpha, expected, atol=3e-13)

    prediction, error = rng.normal(size=(2, width))
    direction = rng.normal(size=2 * width - 1)
    output_change = direction[toeplitz_indices(width)] @ prediction
    np.testing.assert_allclose(np.correlate(error, prediction, mode="full") @ direction,
                               error @ output_change, rtol=1e-12)


def recall_scores(fields, values):
    # All candidate values have unit norm. Their inner-product ranking equals
    # their cosine ranking for each query even when the read norm varies.
    similarities = values @ fields
    correct = np.argmax(similarities, axis=1) == np.arange(len(values))
    matched = similarities.diagonal(axis1=1, axis2=2)
    cosine = matched / np.maximum(np.linalg.norm(fields, axis=1), 1e-30)
    return correct, cosine


def rewrite_test(width, loads, depth, trials=512, rate=0.05):
    """Rewrite one of the keys already in memory, including the stale value in scoring."""
    rng = np.random.default_rng(241127)
    rates = np.array([rate])
    for count in loads:
        totals = np.zeros((2, 10))
        for _ in range(trials):
            keys = np.linalg.qr(rng.normal(size=(width, count)), mode="reduced")[0]
            values = rng.normal(size=(width, count + 1))
            values /= np.linalg.norm(values, axis=0)
            address = rng.integers(count)
            other = np.arange(count) != address
            # Compare a 64-coordinate cosine readout on the same optical fields.
            readout = slice(None, None, width // 64)
            candidates = values[readout] / np.linalg.norm(values[readout], axis=0)
            updated_values = values[:, :count].T.copy()
            updated_values[address] = values[:, -1]
            # Re-evaluate the same ordered product with only one packet changed.
            direct = recall_fields(keys.T, values[:, :count].T, rates, depth=depth)[1]
            replaced = recall_fields(keys.T, updated_values, rates, depth=depth)[1]
            local = local_delta_fields(keys.T, values[:, :count].T, rates)[0]
            repeated_keys = np.vstack((keys.T, keys[:, address]))
            appended = local_delta_fields(repeated_keys, values.T, rates)[0, :, :count]
            for row, (before, after) in zip(totals, ((direct, replaced), (local, appended))):
                old_winner = np.argmax(values.T @ before, axis=0)
                winner = np.argmax(values.T @ after, axis=0)
                row[0] += winner[address] == count
                row[1] += winner[address] == address
                row[2] += np.mean(winner[other] == np.arange(count)[other])
                change = after - before
                row[3] += np.sum(change[:, other]**2) / (count - 1)
                row[4] += np.sum(change[:, address]**2)
                row[5] += np.mean(old_winner[other] == np.arange(count)[other])
                was_correct = other & (old_winner == np.arange(count))
                row[6] += np.count_nonzero(was_correct & (old_winner == winner))
                row[7] += np.count_nonzero(was_correct)
                row[8] += (values[:, -1] @ after[:, address]) / np.linalg.norm(after[:, address])
                row[9] += np.argmax(candidates.T @ after[readout, address]) == count
        for label, row in zip(("known slot", "next slot, local delta"), totals):
            new, stale, other_correct, other_shift, target_shift, before_correct, retained, tested, cosine, readout_new = row
            print(f"{count:2} keys, {label}: new {new/trials:.1%}, stale {stale/trials:.1%}, "
                  f"other top-1 {before_correct/trials:.1%} -> {other_correct/trials:.1%}, "
                  f"previously correct others retained {retained/tested:.1%}, "
                  f"other/target RMS field change {np.sqrt(other_shift/target_shift):.1%}, "
                  f"cosine with new value {cosine/trials:.3f}, "
                  f"new value top-1 on 64 coordinates {readout_new/trials:.1%}", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--local-delta", action="store_true",
                        help="test the forward prediction-error write during initial filling")
    parser.add_argument("--rewrite", action="store_true",
                        help="replace one existing association after the initial writes")
    parser.add_argument("--width", choices=(64, 128), type=int, default=WIDTH)
    args = parser.parse_args()
    width = args.width
    depth = width // 2 + 2
    loads = (8, 16, 32) if width == 64 else (8, 16, 32, 64)
    verify_factor_order(width, depth)
    if args.rewrite:
        print(f"Width {width}, {depth} factor slots, 512 independent streams per load.", flush=True)
        rewrite_test(width, loads, depth)
        return
    rng = np.random.default_rng(241126)
    print("128 independent streams per load. Candidate pool includes every value written.")
    if args.local_delta:
        loads += (depth,)
        labels = [f"{rate:.2f}" for rate in LOCAL_RATES]
        print("Newest-factor normalized gradient update. Step size is beta.")
    else:
        loads += (depth, width)
        labels = ["linear"] + [f"{a:.2f}" for a in ALPHAS]
        print("Direct token convolution. Step size is alpha; linear means the derivative at zero.")
    print("stored / queried      step     top-1 (%)   stream SE (pp)   value cosine")
    for writes in loads:
        groups = {"retained": slice(max(0, writes - depth), None)}
        if writes > depth:
            groups["evicted"] = slice(0, writes - depth)
        records = {name: [] for name in groups}
        for _ in range(STREAMS):
            keys = np.linalg.qr(rng.normal(size=(width, writes)), mode="reduced")[0].T.copy()
            values = rng.normal(size=(writes, width))
            values /= np.linalg.norm(values, axis=1, keepdims=True)
            fields = local_delta_fields(keys, values) if args.local_delta else recall_fields(keys, values, depth=depth)
            correct, cosine = recall_scores(fields, values)
            for name, group in groups.items():
                records[name].append((correct[:, group].mean(axis=1),
                                      cosine[:, group].mean(axis=1)))
        for name, streams in records.items():
            scores = np.array(streams)
            accuracy, cosine = scores.mean(axis=0)
            error = scores[:, 0].std(axis=0, ddof=1) / np.sqrt(STREAMS)
            for label, acc, se, cos in zip(labels, accuracy, error, cosine):
                print(f"{min(writes, depth):2}/{writes:<2} {name:8} {label:>7}"
                      f"{100 * acc:13.2f}{100 * se:17.2f}{cos:15.3f}")


if __name__ == "__main__":
    main()
