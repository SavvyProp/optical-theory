"""Energy and propagation checks for signed MVMs using weak SFG row projections.

Run OPENBLAS_NUM_THREADS=1 python3 notes/two_beam_mvm.py. The published
64-bin experiment supplies the pump and conversion anchor. All efficiencies,
excess loss, operation on bright data and multirow layouts are assumptions.
"""

import numpy as np
from scipy.constants import Boltzmann, c, elementary_charge, h
from scipy.optimize import minimize_scalar


N = 128
PUMP_FJ = 125.0
Q = 0.001  # matched-mode photon conversion, before collection
SIGNAL_PHOTON_FJ = h*c/1540e-9*1e15
PUMP_PHOTON_FJ = h*c/860e-9*1e15
OUTPUT_PHOTON_FJ = h*c/551e-9*1e15
DATA_EFF, WEIGHT_EFF, COLLECTION = 0.40, 0.25, 0.80
LO_EFF = 0.25  # source to detected reference-mode photons


def budget(group, tiles, loss_db, rms, pump_fj=PUMP_FJ, reset_variance=0.):
    # The weak-projection slope is calibrated at one pump energy. Scaling
    # away from that point assumes a mode-selective beamsplitter interaction.
    q = np.sin(np.arcsin(np.sqrt(Q)) * np.sqrt(pump_fj/PUMP_FJ))**2
    transmission = 10**(-loss_db/10)
    t2 = q * transmission**(group-1) / (1 + (group-1)*q)
    # A fixed coherent combiner of two tile outputs costs a factor of two
    # in signal photons. Data in each tile contains distinct input entries.
    data_cost = tiles * SIGNAL_PHOTON_FJ / (DATA_EFF*COLLECTION*group*t2)
    lo_cost = OUTPUT_PHOTON_FJ / (N*LO_EFF)
    # In units of detected electrons, signal and LO give difference-current
    # variance photons + lo_photons. Two reset capacitors can add noise.
    # Choose their optical levels to minimize the sum of the two source rails.
    lo_photons = (1 + np.sqrt(data_cost/lo_cost * (1+4*rms**2*reset_variance))) / (4*rms**2)
    photons = (lo_photons + reset_variance) / (4*rms**2*lo_photons - 1)
    data_fj = data_cost * photons
    weight_fj = tiles * pump_fj / (WEIGHT_EFF*N)
    lo_fj = lo_cost * lo_photons
    return data_fj, weight_fj, lo_fj, t2, photons


def mode_converter(data, pump, steps):
    """Ideal trilinear SFG with matched pairs and one allowed output mode.

    Each row of data is a test input. Fields are real square roots of photon
    counts. Include classical back-action on both beams but omit other output
    modes, dispersion, and quantum fluctuations in the propagation check.
    """
    k = np.arcsin(np.sqrt(Q)) / np.sqrt(PUMP_FJ/PUMP_PHOTON_FJ)
    width = data.shape[1]
    state = np.c_[data, np.broadcast_to(pump, data.shape), np.zeros(len(data))]
    data_photons = np.sum(data**2, axis=1)
    pump_photons = np.sum(pump**2)

    def derivative(fields):
        a, b, out = fields[:, :width], fields[:, width:-1], fields[:, -1:]
        return np.c_[-b*out, -a*out, np.sum(a*b, axis=1)] * k

    for _ in range(steps):
        k1 = derivative(state)
        k2 = derivative(state + k1/(2*steps))
        k3 = derivative(state + k2/(2*steps))
        k4 = derivative(state + k3/steps)
        state += (k1 + 2*k2 + 2*k3 + k4)/(6*steps)
    residual, weights, out = state[:, :width], state[:, width:-1], state[:, -1]
    np.testing.assert_allclose(np.sum(residual**2, axis=1) + out**2, data_photons, rtol=1e-8)
    np.testing.assert_allclose(np.sum(weights**2, axis=1) + out**2, pump_photons, rtol=1e-8)
    return residual, out


def check_rows(matrix, inputs, group=16, tiles=2, loss_db=0.2, rms=0.1,
               pump_fj=PUMP_FJ, steps=8):
    _, _, _, t2, photons = budget(group, tiles, loss_db, rms, pump_fj)
    q_max = np.sin(np.arcsin(np.sqrt(Q)) * np.sqrt(pump_fj/PUMP_FJ))**2
    target = inputs @ matrix.T
    outputs = np.zeros_like(target)
    scale = np.sqrt(tiles*photons/(COLLECTION*t2))
    width = N//tiles
    transit_amplitude = 10**(-loss_db/20)
    for block in range(tiles):
        cols = slice(block*width, (block+1)*width)
        for first in range(0, N, group):
            residual_map = np.eye(width)
            data = inputs[:, cols] * scale
            for j in range(first, first+group):
                wanted = np.sqrt(t2) * matrix[j, cols]
                programmed = np.linalg.solve(residual_map.T, wanted)
                np.testing.assert_allclose(programmed @ residual_map, wanted, atol=1e-13)
                q = programmed @ programmed
                assert q <= q_max + 1e-12
                if q > 0:
                    u = programmed / np.sqrt(q)
                    pump = (np.sqrt(PUMP_FJ/PUMP_PHOTON_FJ) *
                            np.arcsin(np.sqrt(q))/np.arcsin(np.sqrt(Q))) * u
                    data, out = mode_converter(data, pump, steps)
                    outputs[:, j] += out / np.sqrt(tiles)
                    residual_map -= (1 - np.sqrt(1-q)) * np.outer(u, u @ residual_map)
                data *= transit_amplitude
                residual_map *= transit_amplitude
    return outputs / np.sqrt(photons/COLLECTION), target


def main():
    print('Held pump setting, fJ of source energy per useful MAC (includes optical LO)')
    print('rows/copy  bins/guide  rms  dB/transit  weights  data     LO    total')
    for group, tiles, loss, rms in [(1, 1, 0, .10), (8, 1, .2, .10), (1, 2, .2, .10),
                                    (8, 2, .2, .10), (16, 2, .2, .10), (128, 2, .2, .10)]:
        data, weight, lo, _, _ = budget(group, tiles, loss, rms)
        print(f'{group:9d} {N//tiles:11d} {rms:4.0%} {loss:11.1f} {weight:8.3f} '
              f'{data:6.3f} {lo:6.3f} {data+weight+lo:8.3f}')
    # Optimize the source-only model in its weak-conversion range for this
    # geometry. Other source efficiencies, matrices and losses move the result.
    def optimum(group, loss_db, rms, reset_variance=0.):
        return minimize_scalar(lambda p: sum(budget(group, 2, loss_db, rms, p, reset_variance)[:3]),
                               bounds=(1., 4000.), method='bounded')

    for loss_db in (0., .2):
        print(f'64-bin tiles, {loss_db:g} dB/transit, 10% readout RMS')
        for group in (1, 2, 4, 8, 16, 32, 64, 128):
            fit = optimum(group, loss_db, .10)
            print(f'{group:3d} rows/copy: {fit.x:6.2f} fJ pump/tile, {fit.fun:6.2f} fJ/MAC')
    for rms in (.10, .08, .05, .025, .01, 1/256):
        fit = optimum(16, .2, rms)
        print(f'16 rows/copy, {rms:g} RMS: {fit.x:.2f} fJ maximum pump per tile, '
              f'{fit.fun:.2f} fJ/MAC')
    capacitance = 10e-15
    reset_variance = 2 * Boltzmann * 300 * capacitance / elementary_charge**2
    fit = optimum(16, .2, .1, reset_variance)
    reset_fj_per_mac = 2 * capacitance * .8**2 * 1e15 / N
    print(f'Two 10 fF reset nodes at 300 K, 10% RMS: {fit.fun:.2f} fJ/MAC for sources '
          f'plus {reset_fj_per_mac:.2f} fJ/MAC for a two-node full-swing reset at 0.8 V')

    rng = np.random.default_rng(7567)
    dense = rng.normal(size=(N, N))
    dense /= np.linalg.norm(dense, axis=1, keepdims=True)
    shared = np.broadcast_to(dense[0], dense.shape)
    inputs = rng.normal(size=(12, N))
    inputs = np.vstack((inputs, dense[0]))
    inputs *= np.sqrt(N)/np.linalg.norm(inputs, axis=1, keepdims=True)
    best_pump = optimum(16, .2, .10).x
    for label, matrix in [('dense', dense), ('identical rows', shared)]:
        actual, ideal = check_rows(matrix, inputs, pump_fj=best_pump)
        refined, _ = check_rows(matrix, inputs[-1:], pump_fj=best_pump, steps=16)
        np.testing.assert_allclose(actual[-1], refined[0], rtol=1e-8, atol=1e-9)
        for case, subset in [('random', slice(0, -1)), ('aligned', slice(-1, None))]:
            rel = np.linalg.norm(actual[subset]-ideal[subset])/np.linalg.norm(ideal[subset])
            print(f'Ideal selective three-wave model, {label}, {case}: {rel:.3%} '
                  'deterministic vector RMS')


if __name__ == '__main__':
    main()
