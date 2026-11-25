"""Check query-selected vector injection into the original real 34-factor loop.

Ideal linear field model. Address signals, coherent taps, addition and timed
replacement are supplied. No energy or learned address-generation model.
Run OPENBLAS_NUM_THREADS=1 python3 notes/hankel_addressed_injection.py.
"""

import numpy as np
from scipy.linalg import hadamard

from hankel_product_viability import DEPTH, INDICES, N, initial_factors


def read(factors, packets, signals, features):
    """Each column is a query. A feature controls an entire packet, once per lap."""
    fields = signals.copy()
    for h, b, gate in zip(factors, packets, features.T):
        fields = h @ fields + b[:, None] * gate
    return fields


def write_one_slot(factors, packets, value, address):
    field = value.copy()
    for h in factors[:address + 1]:
        field = h @ field
    packets[address] = field


def write_additive(factors, packets, value, features):
    field = value.copy()
    for h, b, gate in zip(factors, packets, features):
        field = h @ field
        b += gate * field


def relative_error(actual, reference):
    return np.linalg.norm(actual - reference) / np.linalg.norm(reference)


def check():
    rng = np.random.default_rng(17031)
    factors = initial_factors()["perturbed"][:, INDICES]
    empty = np.zeros((DEPTH, N))
    directions = rng.normal(size=(N, DEPTH))
    values = rng.normal(size=(N, DEPTH))
    unit_features = np.eye(DEPTH)

    # The dense oracle uses the end-to-end product. The writer sees only
    # forward factors and the value field present at the injection point.
    product = np.eye(N)
    for h in factors:
        product = h @ product
    packets = rng.normal(size=(DEPTH, N))
    constant = np.ones_like(unit_features)
    shift = read(factors, packets, directions, constant) - product @ directions
    np.testing.assert_allclose(shift, np.repeat(shift[:, :1], DEPTH, axis=1), atol=1e-10)

    packets = empty.copy()
    for a, value in enumerate(values.T):
        write_one_slot(factors, packets, value, a)
    previous = read(factors, packets, directions, unit_features)
    worst_crosstalk = 0.
    for address in rng.integers(DEPTH, size=100):
        values[:, address] = rng.normal(size=N)
        write_one_slot(factors, packets, values[:, address], address)
        new = read(factors, packets, directions, unit_features)
        other = np.arange(DEPTH) != address
        worst_crosstalk = max(worst_crosstalk, np.max(abs(new[:, other] - previous[:, other])))
        previous = new
    hard_error = relative_error(previous, product @ (directions + values))
    assert hard_error < 1e-12 and worst_crosstalk < 1e-12
    print(f"34 supplied addresses, 100 overwrites: relative query error {hard_error:.2e}, "
          f"largest change at another address {worst_crosstalk:.2e}")

    # Signed code gates after 32 of the 34 factors. All 34 still act on the data.
    width = 32
    codes = np.pad(hadamard(width) / np.sqrt(width), ((0, 0), (0, DEPTH-width)))
    packets = empty.copy()
    for value, code in zip(values.T[:width], codes):
        write_additive(factors, packets, value, code)
    output = read(factors, packets, directions[:, :width], codes)
    signed_error = relative_error(output, product @ (directions[:, :width] + values[:, :width]))
    assert signed_error < 1e-12
    print(f"32 Walsh addresses, one additive write each: relative query error {signed_error:.2e}")

    # The same rule works for continuous supplied features under their actual
    # inner product; an independent dense feature-space memory is the oracle.
    packets = empty.copy()
    keys = rng.normal(size=(DEPTH, DEPTH))
    keys /= np.linalg.norm(keys, axis=1, keepdims=True)
    queries = rng.normal(size=(DEPTH, DEPTH))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    for value, k in zip(values.T, keys):
        write_additive(factors, packets, value, k)
    prediction = read(factors, packets, directions, queries)
    continuous_error = relative_error(prediction, product @ (directions + values @ keys @ queries.T))
    assert continuous_error < 1e-12
    print(f"continuous 34-coordinate features: relative additive-attention error {continuous_error:.2e}")


if __name__ == "__main__":
    check()
