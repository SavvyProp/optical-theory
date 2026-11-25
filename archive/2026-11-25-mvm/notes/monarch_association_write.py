"""Known-slot writes in the output bank of an ideal signed-real Monarch stage.

Run OPENBLAS_NUM_THREADS=1 python3 notes/monarch_association_write.py.
"""

import numpy as np

from hankel_forward_write_recall import recall_fields


def fixed_bank(rng, lanes, width):
    """A real orthogonal circulant per lane, realizable by two Hankels."""
    phase = rng.uniform(-np.pi, np.pi, size=(lanes, width // 2 + 1))
    spectrum = np.exp(1j * phase)
    spectrum[:, (0, -1)] = 1
    first_column = np.fft.irfft(spectrum, n=width)
    offsets = np.subtract.outer(np.arange(width), np.arange(width)) % width
    return first_column[:, offsets]


def route(bank, inputs):
    """Apply the first bank and send tooth t of lane j to (j+t) mod lanes."""
    lanes, width, _ = bank.shape
    mixed = bank @ inputs.reshape(lanes, width, -1)
    teeth = np.arange(width)
    source_lanes = (np.arange(lanes)[:, None] - teeth) % lanes
    return mixed[source_lanes, teeth]


def check_router(bank):
    lanes, width, _ = bank.shape
    labels = np.arange(lanes * width)[:, None]
    identity = np.broadcast_to(np.eye(width), bank.shape)
    routed = route(identity, labels)[:, :, 0]
    for source in range(lanes):
        for tooth in range(width):
            assert routed[(source + tooth) % lanes, tooth] == source * width + tooth
    np.testing.assert_allclose(bank @ bank.transpose(0, 2, 1), identity, atol=1e-12)


def run(lanes, width, count, trials=256):
    bank = fixed_bank(np.random.default_rng(173), lanes, width)
    check_router(bank)
    # Same global keys, values and chosen slots for both 512-dimensional layouts.
    rng = np.random.default_rng(241128 + count)
    global_hits = local_hits = other_hits = cosine = 0.0
    other_change = target_change = 0.0
    rate = np.array([0.05])
    for _ in range(trials):
        dimension = lanes * width
        keys = np.linalg.qr(rng.normal(size=(dimension, count)), mode="reduced")[0]
        keys_by_lane = route(bank, keys)
        values = rng.normal(size=(dimension, count + 1))
        values /= np.linalg.norm(values, axis=0)
        address = rng.integers(count)
        other = np.arange(count) != address
        predictions = np.empty((lanes, width, count))
        for s, (local_keys, candidates) in enumerate(zip(keys_by_lane, values.reshape(lanes, width, -1))):
            # Each global key has mean squared norm 1/lanes in this lane.
            # The writer uses one fixed scale across lanes and tokens.
            targets = lanes * candidates[:, :count].T
            before = recall_fields(local_keys.T, targets, rate)[1]
            targets[address] = lanes * candidates[:, -1]
            predictions[s] = recall_fields(local_keys.T, targets, rate)[1]
            change = predictions[s] - before
            other_change += np.sum(change[:, other]**2) / (count - 1)
            target_change += np.sum(change[:, address]**2)
            scores = candidates.T @ predictions[s, :, address] / np.linalg.norm(candidates, axis=0)
            local_hits += np.argmax(scores) == count
        field = predictions.reshape(dimension, count)
        winners = np.argmax(values.T @ field, axis=0)
        global_hits += winners[address] == count
        other_hits += np.mean(winners[other] == np.arange(count)[other])
        cosine += values[:, -1] @ field[:, address] / np.linalg.norm(field[:, address])
    print(f"{lanes} x {width}, {count} keys: new global {global_hits/trials:.1%}, "
          f"other global {other_hits/trials:.1%}, new single-lane {local_hits/(lanes*trials):.1%}, "
          f"new cosine {cosine/trials:.3f}, other/target RMS change {np.sqrt(other_change/target_change):.1%}",
          flush=True)


if __name__ == "__main__":
    for count in (16, 32):
        for lanes, width in ((8, 64), (4, 128)):
            run(lanes, width, count)
