"""Probe single 64-tooth forward reads on frozen pretrained TTT-Linear states.

Usage: OPENBLAS_NUM_THREADS=1 python3 notes/hankel_trained_reads.py PATH/states.npz
Extract the pinned checkpoint's states with ttt_state_probe.py. Each article
supplies one prefix and eight subsequent normalized queries against that snapshot.
"""

import argparse

import numpy as np

from hankel_amplifier_sweep import INPUT_PHOTONS, Read, added_photons
from hankel_fwm_stress import (
    ETA, INPUT_LOSS_DB, N, OUTPUT_LOSS_DB, PACKET_POWER_W, PUMP_W, SLOTS,
    field_gain, optical_program, read,
)
from hankel_tooth_calibration import correction


def forward(matrix, queries, scale, q, samples=512, steps=32):
    physical = q[:, None] * matrix * q
    weights, modes, local_scale = optical_program(physical, samples)
    weights *= local_scale / scale
    weak_gain = (np.sqrt(ETA / N) * field_gain(-INPUT_LOSS_DB - OUTPUT_LOSS_DB)
                 / (4 * scale))
    decode = samples * np.sqrt(PACKET_POWER_W) * weak_gain * q
    output = []
    for query in queries:
        weak, generated, _, _, residual = read(weights, modes, q.conj() * query, 0, steps)
        np.testing.assert_allclose(np.fft.fft(weak)[:N] / decode, matrix @ query, atol=1e-10)
        assert residual < 1e-5
        output.append(np.fft.fft(generated)[:N] / decode)

    # Added field variance over 64 teeth in units of squared logical vector norm.
    # This weak-conversion covariance estimate is evaluated separately from depletion.
    budget = Read(physical, scale)
    g1, g2 = 10.0, 10**(31.07 / 10)
    photons = (budget.spontaneous * g1 * g2 + added_photons(g1, 6) * budget.raw * g2
               + added_photons(g2, 5))
    variance = photons / (INPUT_PHOTONS * budget.signal * g1 * g2)
    return np.array(output), variance, PUMP_W * (local_scale / scale)**2


def main(path):
    with np.load(path) as data:
        rows, layers, heads, prefixes = (data[k] for k in ('row', 'layer', 'head', 'prefix'))
        priors, delta, queries = (data[k] for k in ('priors', 'delta', 'queries'))
    lookup = {key: i for i, key in enumerate(zip(rows, layers, heads, prefixes))}
    articles = np.unique(rows)
    q = np.exp(1j * np.pi * SLOTS**2 / (2*N))
    slots = np.arange(0, 64, 8)
    # Three calibration articles followed by three disjoint evaluation articles.
    cases = list(zip(articles[[0, 1, 2, 5, 8, 11]], (64, 256, 1024, 64, 256, 1024)))
    lane_totals = []
    held_powers = []
    max_refinement = 0.0
    print('Layer/head  deterministic %  added noise %  independent sum %')
    for layer in range(12):
        head = (5 * layer + 1) % 12
        indices = [lookup[row, layer, head, prefix] for row, prefix in cases]
        matrices = priors[layer, head].astype(float) + delta[indices].astype(float)
        # Reserve 15% field headroom over the largest calibration weight comb.
        # Both the optical scale and the fitted complex gain stay fixed per lane.
        scale = 1.15 * max(optical_program(q[:, None]*w*q, N)[2] for w in matrices[:3])
        actual, targets, variances = [], [], []
        for i, (idx, matrix) in enumerate(zip(indices, matrices)):
            keys = queries[idx, slots].astype(float)
            keys /= np.linalg.norm(keys, axis=1, keepdims=True)
            out, var, power = forward(matrix, keys, scale, q)
            actual.append(out)
            targets.append(keys @ matrix.T)
            if i >= 3:
                variances.append(len(keys) * var)
                held_powers.append(power)
            if i == 3:
                refined = forward(matrix, keys[:1], scale, q, samples=1024, steps=64)[0]
                diff = np.linalg.norm(refined - out[:1]) / np.linalg.norm(targets[-1][:1])
                max_refinement = max(max_refinement, diff)
                assert diff < 1e-4

        actual, targets = np.asarray(actual), np.asarray(targets)
        gain = correction(actual[:3], targets[:3], axis=None)
        bias = np.linalg.norm(gain * actual[3:] - targets[3:])**2
        noise = abs(gain)**2 * np.sum(variances)
        energy = np.linalg.norm(targets[3:])**2
        totals = np.array([bias, noise, bias + noise, energy])
        print(f'{layer:2}/{head:2}', np.round(100 * np.sqrt(totals[:3]/energy), 3), flush=True)
        lane_totals.append(totals)

    sums = np.sum(lane_totals, axis=0)
    print('Pooled deterministic, added noise, independent sum (%):',
          (100 * np.sqrt(sums[:3]/sums[3])).round(3))
    lanes = np.array(lane_totals)
    print('Lane min/median/max in same columns (%):',
          np.percentile(100 * np.sqrt(lanes[:, :3] / lanes[:, 3:]), [0, 50, 100], axis=0).round(3))
    count = len(lane_totals) * 3 * len(slots)
    print(f'{len(lane_totals)} lanes, {count} calibration and {count} held-out reads, '
          f'{len(held_powers)} held-out snapshots, {np.sum(np.asarray(held_powers) > PUMP_W)} '
          f'over 50 mW, peak slot {1000*max(held_powers):.2f} mW.')
    print(f'Maximum relative refinement difference across {len(lane_totals)} probes: {max_refinement:.2e}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('states')
    main(parser.parse_args().states)
