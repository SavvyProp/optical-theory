"""Check forward dense-memory reads and writes with two bands in one Hankel loop.

Run with python3 notes/hankel_displacement_memory.py. Fields are normalized.
The physical read model keeps only the selected frequencies of an ideal,
linearized phase-conjugating FWM process. It assumes perfect storage and gain.
"""

from math import comb

import numpy as np


def encode(matrix):
    """Slot l stores column c minus the next column shifted up one row."""
    extended = np.pad(matrix, ((0, 1), (0, 1)))
    differences = extended[:-1, :-1] - extended[1:, 1:]
    return differences[:, ::-1].T.copy()


def write(bank, residual, key):
    """Add residual @ key† by evaluating the selected FWM difference sideband.

    Light the two key teeth with fields k[c] and -k[c+1]. At output i, the
    corresponding pairs at (i,0) and (i+1,1) have the same frequency difference.
    The opposite physical sideband contains the conjugate, read on alternate slots.
    """
    padded_key = np.r_[key, 0]
    for slot, column in enumerate(range(key.size - 1, -1, -1)):
        tones = np.array([padded_key[column], -padded_key[column + 1]])
        difference_sideband = np.convolve(residual, tones[::-1].conj())
        bank[slot] += difference_sideband[1:]


def read(bank, queries, carrier_ratio=0.0):
    """Propagate physical fields at all 2d retained frequencies for each probe.

    S maps e_j to e_(j+1) and drops the last tooth. The optional shifter model
    is (S + r I)/(1 + |r|) in the logical query band; both terms together are
    a contraction. The residual carrier has power |r|² relative to the shifted
    field. This is a sensitivity model, not a fit to a particular device.
    """
    width = len(bank)
    bands = 2 * width
    center = bands - 1
    offsets = center - np.arange(width)
    indices = np.add.outer(np.arange(bands), np.arange(bands))
    probes = np.asarray(queries).reshape(width, -1)
    packet = np.vstack((np.zeros_like(probes), probes)).astype(complex)
    for slot, coefficient in enumerate(bank):
        weights = np.zeros(2 * bands - 1, dtype=complex)
        weights[center] = 1.0
        if slot % 2 == 0:
            weights[center + offsets] = coefficient.conj()
        else:
            weights[center - offsets] = coefficient
        packet = np.einsum("ij,jk->ik", weights[indices], packet.conj())

        # Each conjugating Hankel visit reverses the physical indexing of the
        # two logical bands. A reversed chart also conjugates the carrier spur.
        if slot % 2 == 0:
            band = packet[:width]
            shifted = np.roll(band, -1, axis=0)
            shifted[-1] = 0
            ratio = np.conj(carrier_ratio)
        else:
            band = packet[width:]
            shifted = np.roll(band, 1, axis=0)
            shifted[0] = 0
            ratio = carrier_ratio
        band[:] = (shifted + ratio * band) / (1 + abs(ratio))

    if width % 2:
        packet = packet[::-1].conj()
    return packet[:width].reshape(np.shape(queries))


def two_hankel_read(bank, queries, guard):
    """Replace the band-specific shifter by a second, two-tone FWM visit.

    After the pair, y and q exchange bands and both have ascending logical
    tooth order. An internal stop band isolates the odd-slot programmable
    shear from the two diagonal blocks.
    """
    width = len(bank)
    separation = width + guard
    bands = separation + width
    center = bands - 1
    indices = np.add.outer(np.arange(bands), np.arange(bands))
    probes = np.asarray(queries).reshape(width, -1)
    packet = np.zeros((bands, probes.shape[1]), complex)
    packet[separation:] = probes
    for slot, coefficient in enumerate(bank):
        y_start, q_start = (0, separation) if slot % 2 == 0 else (separation, 0)
        # Raw shear visit conjugates and reverses the two logical bands.
        shear = np.zeros(2 * bands - 1, complex)
        shear[center] = 1.0
        offsets = q_start - y_start + width - 1 - np.arange(width)
        shear[center + offsets] = coefficient.conj()
        packet = np.einsum("ij,jk->ik", shear[indices], packet.conj())
        packet[width:separation] = 0

        # Shift visit reverses each occupied band locally. y is now in the
        # old q band. The next anti-diagonal over also shifts q up one tooth.
        shift = np.zeros(2 * bands - 1)
        shift[2 * q_start + width - 1] = 1.0
        shift[2 * y_start + width] = 1.0
        packet = np.einsum("ij,jk->ik", shift[indices], packet.conj())
        packet[width:separation] = 0
    y_start = separation if width % 2 else 0
    return packet[y_start:y_start + width].reshape(np.shape(queries))


def random_fields(rng, width, count):
    array = rng.standard_normal((width, count)) + 1j * rng.standard_normal((width, count))
    return array / np.linalg.norm(array, axis=0)


def relative_error(actual, expected):
    return np.linalg.norm(actual - expected) / np.linalg.norm(expected)


def association_test(rng, width, count):
    keys, _ = np.linalg.qr(random_fields(rng, width, count))
    values = random_fields(rng, width, count)
    probes = random_fields(rng, width, 16)
    matrix = np.zeros((width, width), complex)
    bank = encode(matrix)

    def associate(key, value):
        dense_residual = value - matrix @ key
        optical_residual = value - read(bank, key)
        matrix[:] += np.outer(dense_residual, key.conj())
        write(bank, optical_residual, key)

    for key, value in zip(keys.T, values.T):
        associate(key, value)
    first_error = relative_error(read(bank, keys), values)

    overwrite = rng.choice(count, 8, replace=False)
    values[:, overwrite] = random_fields(rng, width, len(overwrite))
    for index in overwrite:
        associate(keys[:, index], values[:, index])
    matches_dense = relative_error(read(bank, probes), matrix @ probes)
    recall_error = relative_error(read(bank, keys), values)
    print(f"{count} associations: relative value RMS {first_error:.3g}; after 8 "
          f"overwrites {recall_error:.3g}; independent probes vs dense {matches_dense:.3g}")
    assert max(first_error, recall_error, matches_dense) < 1e-12


def shifter_sensitivity(rng, width):
    # Static uncorrected carrier leakage, 16 gain pilots and 32 disjoint probes.
    matrix, _ = np.linalg.qr(random_fields(rng, width, width))
    bank = encode(matrix)
    probes = random_fields(rng, width, 48)
    target = matrix @ probes
    for dbc in (-50, -40, -30):
        ratio = 1j * 10 ** (dbc / 20)
        estimate = read(bank, probes, carrier_ratio=ratio)
        pilots = estimate[:, :16]
        scalar = np.vdot(pilots, target[:, :16]) / np.vdot(pilots, pilots)
        raw = relative_error(estimate[:, 16:], target[:, 16:])
        calibrated = relative_error(scalar * estimate[:, 16:], target[:, 16:])
        # If the fixed operator is S' = a S + b I, expand each S^l as
        # a^(-l) (S' - b I)^l. This calibration mixes stored weight slots.
        a = 1 / (1 + abs(ratio))
        b = a * ratio
        compensated_bank = np.zeros_like(bank)
        for l, coefficients in enumerate(bank):
            for m in range(l + 1):
                compensated_bank[m] += comb(l, m) * (-b)**(l-m) * a**(-l) * coefficients
        compensated = read(compensated_bank, probes[:, 16:], carrier_ratio=ratio)
        encoded = relative_error(compensated, target[:, 16:])
        sideband_energy = (np.linalg.norm(compensated_bank) / np.linalg.norm(bank))**2
        print(f"{dbc} dBc quadrature spur: held-out relative field RMS "
              f"{raw:.2%} raw, {calibrated:.2%} after one fitted complex gain, "
              f"{encoded:.3g} with known-spur encoding "
              f"(summed side-comb squared norm {sideband_energy:.3g}x)")
        assert encoded < 1e-12


def main():
    rng = np.random.default_rng(711)
    for width in (3, 64):
        matrix = random_fields(rng, width, width)
        probes = random_fields(rng, width, 7)
        bank = encode(matrix)
        assert relative_error(read(bank, probes), matrix @ probes) < 1e-12
        residual, key = random_fields(rng, width, 2).T
        write(bank, residual, key)
        updated = matrix + np.outer(residual, key.conj())
        assert relative_error(bank, encode(updated)) < 1e-12
        assert relative_error(read(bank, probes), updated @ probes) < 1e-12
        widened = two_hankel_read(bank, probes, guard=width-1)
        restored = relative_error(widened, updated @ probes)
        assert restored < 1e-12
        if width == 64:
            narrow = two_hankel_read(bank, probes, guard=1)
            print(f"two FWM visits per slot, {width} channels: relative RMS "
                  f"{restored:.3g} with {width-1} guard teeth, "
                  f"{relative_error(narrow, updated @ probes):.3g} with one guard tooth")
    for count in (32, 64):
        association_test(rng, 64, count)
    print("64-dimensional head (128 data teeth)")
    shifter_sensitivity(rng, 64)
    # This head also fits the original 64 data / 127 weight frequency grids.
    smaller_head = np.random.default_rng(732)
    print("32-dimensional head (64 data teeth)")
    association_test(smaller_head, 32, 32)
    shifter_sensitivity(smaller_head, 32)


if __name__ == "__main__":
    main()
