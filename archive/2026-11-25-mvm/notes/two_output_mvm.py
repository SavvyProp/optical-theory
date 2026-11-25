"""Two constructions of independent output rows and their weak source budgets.

Run OPENBLAS_NUM_THREADS=1 python3 notes/two_output_mvm.py. Energy comparisons
charge the delivered squared coefficient norms in every layout. The absolute
conversion slope and passive transmission are conditional on the hardware.
"""

import numpy as np
from scipy.optimize import minimize_scalar
from two_beam_mvm import (COLLECTION, DATA_EFF, LO_EFF, N, OUTPUT_PHOTON_FJ,
                          PUMP_FJ, Q, SIGNAL_PHOTON_FJ, WEIGHT_EFF)


def encode_row_pair(first, second):
    # The anti-diagonal index s equals j for row zero and j+1 for row one.
    a = np.r_[first, 0.]
    b = np.r_[0., second]
    return (a-b)/2, (a+b)/2


def verify_matrix(matrix):
    """Verify all entries including edge coefficients, both 64-input tiles."""
    coded = np.empty_like(matrix)
    interleaved = np.empty_like(matrix)
    for columns in (slice(0, N//2), slice(N//2, N)):
        for row in range(0, N, 2):
            tile = matrix[row:row+2, columns]
            first, second = encode_row_pair(*tile)
            coded[row, columns] = first[:-1] + second[:-1]
            coded[row+1, columns] = -first[1:] + second[1:]
            np.testing.assert_allclose(first@first + second@second, np.sum(tile**2)/2)
    # The single-pass alternative splits by even/odd input teeth instead.
    j = np.arange(N//2)
    output_bins = 2*j[-1] + np.arange(2)
    pump_bins = output_bins[:, None] - 2*j
    assert np.unique(pump_bins).size == N
    for parity in (0, 1):
        columns = slice(parity, None, 2)
        for row in range(0, N, 2):
            pump = np.zeros(N)
            # The odd-input tile shifts all pump carriers by -1 bin so that
            # the completed partials reach the same two output frequencies.
            pump[pump_bins] = matrix[row:row+2, columns]
            interleaved[row:row+2, columns] = pump[output_bins[:, None] - 2*j]
    np.testing.assert_allclose(coded, matrix, atol=1e-14)
    np.testing.assert_allclose(interleaved, matrix, atol=1e-14)


def source_budget(scheme, group, slope_ratio=1., loss_db=.2, rms=.10):
    """Common target field t at each output, unit full-row norms, white inputs.

    The phase element and both wavelength bands in the coded design have the
    same interstage transmission. The time slots overlap perfectly. Ignore
    depletion when computing energy, as in the separate one-row budget.
    """
    transmission = 10**(-loss_db/10)
    kappa = Q/PUMP_FJ * slope_ratio
    ages = np.arange(group)
    if scheme == "interleaved":
        ages = ages // 2  # two completed rows on one visit
    weight_cost = np.mean(transmission**(-ages)) / (kappa*WEIGHT_EFF*N)
    if scheme == "coded":
        # Across the two column tiles, both full rows have norm one.
        # Two singles pay (1+T)*T^(-r-1) for row pair r,r+1; each coded
        # pair pays T^(-r-1). This includes propagation of the first output.
        weight_cost /= 1 + transmission
    data_cost = 2*SIGNAL_PHOTON_FJ / (DATA_EFF*COLLECTION*group)
    lo_cost = OUTPUT_PHOTON_FJ/(N*LO_EFF)

    def cost(t2):
        # Finite-reference homodyne, minimized over the reference brightness.
        return t2*weight_cost + (np.sqrt(data_cost/t2) + np.sqrt(lo_cost))**2/(4*rms**2)

    fit = minimize_scalar(cost, bounds=(1e-7, .02), method="bounded", options={"xatol": 1e-13})
    return fit.fun, fit.x, fit.x*weight_cost


def main():
    rng = np.random.default_rng(284)
    dense = rng.normal(size=(N, N))
    for matrix in (dense, np.eye(N), np.repeat(dense[:1], N, axis=0)):
        verify_matrix(matrix)
    print("Verified two independently encoded rows in every tile of all three matrices.")
    for scheme in ("single", "coded", "interleaved"):
        fixed, t2, weight = source_budget(scheme, 16)
        best, group = min((source_budget(scheme, g)[0], g) for g in (2, 4, 8, 16, 32, 64, 128))
        print(f"{scheme}: 16 rows/copy {fixed:.3f} fJ/MAC, {weight:.3f} weights, "
              f"t^2 {t2:.7f}; best equal group {group} rows, {best:.3f} fJ/MAC")
    ratio = 1/(1+10**(-.2/10))
    np.testing.assert_allclose(source_budget("coded", 16, ratio)[0],
                               source_budget("single", 16)[0])
    print(f"Coded vs single break-even slope ratio at 0.2 dB/transit: {ratio:.4f}")


if __name__ == "__main__":
    main()
