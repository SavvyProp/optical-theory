"""Ideal real-field checks for exchanging one tensor-product key in fiber.

Run: OPENBLAS_NUM_THREADS=1 python3 notes/temporal_mode_exchange.py
"""

import numpy as np


LEVELS = 6
WIDTH = 2**LEVELS


def key_from_angles(angles):
    key = np.ones(1)
    for angle in angles:
        key = np.kron([np.cos(angle), np.sin(angle)], key)
    return key


def mix_slots(state, angles):
    """One forward traversal. Columns are time slots, rows are comb teeth."""
    fields = state.copy(order="C")
    for level, angle in enumerate(angles):
        stride = 2**level
        pairs = fields.reshape(fields.shape[0], -1, 2, stride)
        first = pairs[:, :, 0, :].copy()
        second = pairs[:, :, 1, :].copy()
        c, s = np.cos(angle), np.sin(angle)
        pairs[:, :, 0, :] = c * first + s * second
        pairs[:, :, 1, :] = s * first - c * second
    return fields


def exchange(state, angles, value):
    mixed = mix_slots(state, angles)
    prediction = mixed[:, 0].copy()
    mixed[:, 0] = value
    return prediction, mix_slots(mixed, angles)


def verify_orthogonal_writes(rng):
    bits = (np.arange(WIDTH)[:, None] >> np.arange(LEVELS)) & 1
    angles = (1 - 2 * bits) * np.pi / 4
    keys = np.column_stack([key_from_angles(a) for a in angles])
    values = np.zeros((WIDTH, WIDTH))
    state = np.zeros_like(values)
    worst = 0.0
    for address in np.r_[rng.permutation(WIDTH), rng.integers(WIDTH, size=128)]:
        new_value = rng.normal(size=WIDTH)
        prediction, state = exchange(state, angles[address], new_value)
        np.testing.assert_allclose(prediction, values[:, address], atol=1e-11)
        values[:, address] = new_value
        error = np.linalg.norm(state @ keys - values) / np.linalg.norm(values)
        worst = max(worst, error)
    assert worst < 1e-11
    return worst


def verify_soft_writes(rng):
    state = rng.normal(size=(WIDTH, WIDTH))
    oracle = state.copy()
    worst = 0.0
    for _ in range(128):
        angles = rng.uniform(-np.pi, np.pi, LEVELS)
        value = rng.normal(size=WIDTH)
        key = key_from_angles(angles)
        prediction, state = exchange(state, angles, value)
        np.testing.assert_allclose(prediction, oracle @ key, atol=1e-11)
        oracle += np.outer(value - oracle @ key, key)
        query_angles = rng.uniform(-np.pi, np.pi, LEVELS)
        expected = oracle @ key_from_angles(query_angles)
        actual = mix_slots(state, query_angles)[:, 0]
        worst = max(worst, np.linalg.norm(actual - expected) / np.linalg.norm(expected))
    assert worst < 1e-11
    return worst


def single_pass_recall(rng, count, trials=128):
    """One delta write per key from an empty bank, then read every key."""
    error_energy = np.zeros(2)
    target_energy = 0.0
    for _ in range(trials):
        angles = rng.uniform(-np.pi, np.pi, size=(count, LEVELS))
        product_keys = np.column_stack([key_from_angles(a) for a in angles])
        sphere_keys = rng.normal(size=(WIDTH, count))
        sphere_keys /= np.linalg.norm(sphere_keys, axis=0)
        values = rng.normal(size=(WIDTH, count))
        target_energy += np.sum(values**2)
        for index, keys in enumerate((product_keys, sphere_keys)):
            state = np.zeros((WIDTH, WIDTH))
            for key, value in zip(keys.T, values.T):
                state += np.outer(value - state @ key, key)
            error_energy[index] += np.sum((state @ keys - values)**2)
    return np.sqrt(error_energy / target_energy)


def angle_jitter(jitter_deg, same_on_both_traversals):
    """Repeat random overwrites of the orthogonal codebook; compare old values."""
    rng = np.random.default_rng(150316)
    noise_rng = np.random.default_rng(8732)
    bits = (np.arange(WIDTH)[:, None] >> np.arange(LEVELS)) & 1
    angles = (1 - 2 * bits) * np.pi / 4
    values = rng.normal(size=(WIDTH, WIDTH))
    keys = np.column_stack([key_from_angles(a) for a in angles])
    state = values @ keys.T
    error_energy = signal_energy = 0.0
    for _ in range(2048):
        address = rng.integers(WIDTH)
        noise = noise_rng.normal(scale=np.deg2rad(jitter_deg), size=(2, LEVELS))
        first, second = angles[address] + noise
        mixed = mix_slots(state, first)
        error_energy += np.sum((mixed[:, 0] - values[:, address])**2)
        signal_energy += np.sum(values[:, address]**2)
        values[:, address] = rng.normal(size=WIDTH)
        mixed[:, 0] = values[:, address]
        state = mix_slots(mixed, first if same_on_both_traversals else second)
    return np.sqrt(error_energy / signal_energy)


if __name__ == "__main__":
    rng = np.random.default_rng(241126)
    print(f"Orthogonal overwrite, worst bank-relative error: {verify_orthogonal_writes(rng):.2e}")
    print(f"Independent soft query, worst relative error: {verify_soft_writes(rng):.2e}")
    for count in (16, 32, 64):
        tensor, sphere = single_pass_recall(rng, count)
        print(f"{count} random keys: recall field RMS, product {tensor:.1%}, dense sphere {sphere:.1%}")
    for degrees in (0.25, 1):
        shared = angle_jitter(degrees, True)
        independent = angle_jitter(degrees, False)
        print(f"{degrees:g} deg RMS per angle: recall field RMS, shared {shared:.1%}, independent {independent:.1%}")
    photon_j = 6.62607015e-34 * 299792458 / 1.55e-6
    for packet_pj in (0.2, 2.0):
        photons_per_tooth = packet_pj * 1e-12 / (WIDTH * photon_j)
        for loss_db in (0.5, 1, 4.8):
            gain = 10**(loss_db / 10)
            added_pj = WIDTH * packet_pj * (1 - 1 / gain)
            rail_fj_per_mac = added_pj * 1000 / (0.121 * WIDTH**2)
            ase_per_visit = 1.58 * (gain - 1)
            field_rms = np.sqrt(np.array([64, 1000]) * ase_per_visit / photons_per_tooth)
            print(f"{packet_pj:g} pJ/column, {loss_db:.1f} dB: {rail_fj_per_mac:.1f} fJ/MAC "
                  f"restoration; ASE field RMS at 64/1000 visits {field_rms[0]:.1%}/{field_rms[1]:.1%}")
