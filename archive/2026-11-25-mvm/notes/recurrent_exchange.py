"""Check a two-waveguide recurrent memory. Run with Python and NumPy."""

import numpy as np


BANKS = 32
TEETH = 64


def exchange(pair, value, read_angle, write_angle):
    """Return the old projection and two new columns of optical field amplitudes."""
    a, b = pair.T
    c, s = np.cos(read_angle), np.sin(read_angle)
    output = c * a + s * b
    retained = -s * a + c * b
    c, s = np.cos(write_angle), np.sin(write_angle)
    updated = np.column_stack((c * value - s * retained, s * value + c * retained))
    return output, updated


def check_delta_rule():
    rng = np.random.default_rng(24680)
    state = rng.normal(size=(TEETH, 2 * BANKS))
    reference = state.copy()
    worst = 0.0
    for _ in range(1024):
        bank = rng.integers(BANKS)
        columns = slice(2 * bank, 2 * bank + 2)
        angle = rng.uniform(-np.pi, np.pi)
        value = rng.normal(size=TEETH)
        _, state[:, columns] = exchange(state[:, columns], value, angle, angle)
        key = np.zeros(2 * BANKS)
        key[columns] = [np.cos(angle), np.sin(angle)]
        reference += np.outer(value - reference @ key, key)
        worst = max(worst, np.linalg.norm(state - reference) / np.linalg.norm(reference))
    return worst


def recall_error(jitter_degrees):
    # The router is given the correct bank and direction for every key. Each
    # update emits the last value for that key and writes a fresh signed vector.
    data_rng = np.random.default_rng(37512)
    jitter_rng = np.random.default_rng(2607)
    base_angles = data_rng.uniform(-np.pi, np.pi, BANKS)
    angles = (base_angles[:, None] + [0, np.pi / 2]).ravel()
    state = np.zeros((BANKS, TEETH, 2))
    last = np.zeros((2 * BANKS, TEETH))
    initialize = data_rng.permutation(2 * BANKS)
    accesses = data_rng.integers(2 * BANKS, size=8192)
    squared_error = squared_signal = 0.0
    for index, key in enumerate(np.concatenate((initialize, accesses))):
        bank = key // 2
        value = data_rng.normal(size=TEETH)
        offsets = np.deg2rad(jitter_degrees) * jitter_rng.normal(size=2)
        output, state[bank] = exchange(state[bank], value, angles[key] + offsets[0],
                                       angles[key] + offsets[1])
        if index >= len(initialize):
            squared_error += np.sum((output - last[key])**2)
            squared_signal += np.sum(last[key]**2)
        last[key] = value
    return np.sqrt(squared_error / squared_signal)


if __name__ == "__main__":
    print(f"Maximum relative state error vs. 1,024 dense delta updates: {check_delta_rule():.3g}")
    print("Read/write rotation jitter (degrees RMS)    Relative recall field error")
    for jitter in (0, 1, 3):
        print(f"{jitter:6.1f}                                    {100 * recall_error(jitter):.3g}%")
