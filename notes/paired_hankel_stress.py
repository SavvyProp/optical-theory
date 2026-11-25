"""Classical bright-field propagation for two independent rows from two Hankel visits.

Run OPENBLAS_NUM_THREADS=1 python3 notes/paired_hankel_stress.py. The 128 inputs
occupy two 64-tooth tiles. One input copy serves 16 rows in eight coded pairs.
Input and sum-frequency fields both propagate between the visits in a pair.
The generated band starts empty for every new pair; the data continues.

Assume a periodic comb, a perfectly flat and phase-matched travelling-wave SFG
interaction, and ideal selection of the two output teeth. The 0/pi phase step
is exact in this discrete model. Treat the published single-mode conversion
slope as a hypothesis for each spectral triplet in the broad mixer. No
finite-packet walk-off, modal dispersion, stochastic noise or electronic
receiver is simulated. The reported errors use the nominal gain without a fit.
"""

import numpy as np

from two_beam_mvm import (COLLECTION, DATA_EFF, LO_EFF, N, OUTPUT_PHOTON_FJ,
                          PUMP_FJ, PUMP_PHOTON_FJ, Q, SIGNAL_PHOTON_FJ)
from two_output_mvm import encode_row_pair, source_budget


TILES = 2
WIDTH = N // TILES
VISITS = 16
HOP = 10**(-0.2 / 20)  # field transmission
TARGET_POWER_GAIN = source_budget("coded", VISITS)[1]
COUPLING = np.sqrt((Q/PUMP_FJ) * PUMP_PHOTON_FJ)  # weak field gain per sqrt(photon)


def to_time(tones, samples):
    return np.fft.ifft(tones, n=samples, axis=-1) * samples


def to_spectrum(field):
    return np.fft.fft(field, axis=-1) / field.shape[-1]


def mix(data, pump, idler, steps):
    """Integrate the three complex fields in sqrt(photons per comb period)."""
    state = np.stack((data, np.broadcast_to(pump, data.shape), idler))
    initial_data_sum = np.mean(abs(data)**2 + abs(idler)**2, axis=1)
    initial_pump_sum = np.mean(abs(pump)**2 + abs(idler)**2, axis=1)

    def derivative(fields):
        a, p, c = fields
        return COUPLING * np.stack((-p.conj()*c, -a.conj()*c, a*p))

    for _ in range(steps):
        k1 = derivative(state)
        k2 = derivative(state + k1/(2*steps))
        k3 = derivative(state + k2/(2*steps))
        k4 = derivative(state + k3/steps)
        state += (k1 + 2*k2 + 2*k3 + k4)/(6*steps)
    a, p, c = state
    # One data photon and one pump photon are consumed per generated photon.
    np.testing.assert_allclose(np.mean(abs(a)**2 + abs(c)**2, axis=1),
                               initial_data_sum, rtol=1e-8, atol=2e-8)
    np.testing.assert_allclose(np.mean(abs(p)**2 + abs(c)**2, axis=1),
                               initial_pump_sum, rtol=1e-8, atol=2e-8)
    return a, c


def input_scale():
    # Photon allocation for the same ideal 10% balanced receiver as the source
    # budget; collection and the equal-weight two-tile combiner cost photons.
    data_cost = TILES*SIGNAL_PHOTON_FJ/(VISITS*TARGET_POWER_GAIN*DATA_EFF*COLLECTION)
    lo_cost = OUTPUT_PHOTON_FJ/(N*LO_EFF)
    detected = (1 + np.sqrt(lo_cost/data_cost))/(4*0.10**2)
    return np.sqrt(TILES*detected/(COLLECTION*TARGET_POWER_GAIN))


def project_group(rows, inputs, samples, steps):
    """Read 16 consecutive rows, starting with a fresh copy of the input."""
    answer = np.zeros((len(inputs), VISITS), complex)
    scale = input_scale()
    indices = (-np.arange(WIDTH)) % samples
    for tile in range(TILES):
        columns = slice(WIDTH*tile, WIDTH*(tile+1))
        incident = np.zeros((len(inputs), samples), complex)
        incident[:, indices] = scale*inputs[:, columns]
        data = to_time(incident, samples)
        for row in range(0, VISITS, 2):
            first, second = encode_row_pair(rows[row, columns], rows[row+1, columns])
            # Every useful contribution has traversed row+1 intervening hops
            # when the second visit completes, in either the input or idler band.
            pump_scale = np.sqrt(TARGET_POWER_GAIN) / (HOP**(row+1)*COUPLING)
            data, idler = mix(data, pump_scale*to_time(first, samples), np.zeros_like(data), steps)
            # All nonlinear products have been generated above. Keep the two
            # output bins across the gap, flipping the second. Other idler
            # light is discarded and has already perturbed the original data.
            selected = to_spectrum(idler)[:, :2].copy()
            selected[:, 1] *= -1
            data, idler = mix(HOP*data, pump_scale*to_time(second, samples),
                              HOP*to_time(selected, samples), steps)
            answer[:, row:row+2] += to_spectrum(idler)[:, :2]/np.sqrt(TILES)
            data *= HOP
    return answer/(scale*np.sqrt(TARGET_POWER_GAIN/TILES))


def main():
    rng = np.random.default_rng(32)
    rows = rng.normal(size=(VISITS, N))
    rows /= np.linalg.norm(rows, axis=1, keepdims=True)
    white = rng.normal(size=(12, N))
    scale = input_scale()
    print(f'Data energy in a tile averaged over isotropic inputs: '
          f'{WIDTH*scale**2*SIGNAL_PHOTON_FJ/1000:.3f} pJ')
    for name, matrix in [('dense', rows), ('repeated', np.broadcast_to(rows[:1], rows.shape)),
                         ('positive', np.ones_like(rows)/np.sqrt(N))]:
        inputs = np.vstack((white, matrix[0], np.ones(N)))
        inputs = inputs*np.sqrt(N)/np.linalg.norm(inputs, axis=1, keepdims=True)
        target = inputs @ matrix.T
        actual = project_group(matrix, inputs, 512, 16)
        fine = project_group(matrix, inputs[[-1]], 1024, 32)
        refinement = np.linalg.norm(fine-actual[-1:])/np.linalg.norm(target[-1:])
        assert refinement < 1e-6
        for label, group in [('12 random', slice(0,-2)), ('first row', -2), ('all positive', -1)]:
            error = np.linalg.norm(actual[group]-target[group])/np.linalg.norm(target[group])
            target_rms = np.sqrt(np.mean(target[group]**2))
            print(f'{name:>8} rows, {label:>12} input: field error {error:8.3%}, '
                  f'target RMS {target_rms:.4g}')
        print(f'  Last-probe refinement relative to target: {refinement:.2g}')


if __name__ == '__main__':
    main()
