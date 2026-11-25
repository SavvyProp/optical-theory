"""Global loss gradients on real 8 x 128 Hankel banks and successive cyclic shuffles.

Run OPENBLAS_NUM_THREADS=1 python3 -B notes/monarch_global_gradient.py --shuffles 1 3.
Rates are chosen on independent simulated writes. No optical gain/noise or address oracle.
"""

import argparse

import numpy as np


LANES, WIDTH = 8, 128
TEETH = 2 * WIDTH - 1
TOOTH = np.arange(WIDTH)
LANE = np.arange(LANES)[:, None]


def program(rng, banks, depth):
    """Independent orthogonal Hankels alternating periodic and antiperiodic boundaries."""
    weights = np.empty((banks, depth, LANES, TEETH))
    for parity in (0, 1):
        selected = weights[:, parity::2]
        phases = np.ones((*selected.shape[:-1], WIDTH), dtype=complex)
        for index in range(WIDTH):
            partner = (-index - parity) % WIDTH
            if index < partner:
                phases[..., index] = np.exp(1j * rng.uniform(-np.pi, np.pi, phases.shape[:-1]))
                phases[..., partner] = phases[..., index].conj()
            elif index == partner:
                phases[..., index] = rng.choice([-1, 1], phases.shape[:-1])
        first = (np.fft.ifft(phases) * np.exp(1j * parity * np.pi * TOOTH / WIDTH)).real
        selected[...] = np.concatenate((first, (-1)**parity * first[..., :-1]), axis=-1)
    return weights


def spectrum(weights):
    return np.fft.rfft(weights, n=2 * WIDTH, axis=-1)


def multiply(factor, fields):
    """Fields have shape [probe, lane, tooth]; a finite Hankel is symmetric."""
    product = spectrum(fields[..., ::-1]) * factor
    return np.fft.irfft(product, n=2 * WIDTH)[..., WIDTH-1:TEETH]


def read(factors, fields, save=None):
    for bank, stack in enumerate(factors):
        if bank:
            fields = fields[:, (LANE - TOOTH) % LANES, TOOTH]
        for slot, factor in enumerate(stack):
            if save is not None:
                save[bank, slot] = fields[0]
            fields = multiply(factor, fields)
    return fields


def gradients(factors, key, value):
    banks, depth = factors.shape[:2]
    activations = np.empty((banks, depth, LANES, WIDTH))
    prediction = read(factors, key[None], activations)[0]
    error = (value - prediction)[None]
    grad = np.empty((banks, depth, LANES, TEETH))
    for bank in reversed(range(banks)):
        for slot in reversed(range(depth)):
            product = spectrum(error[0]) * spectrum(activations[bank, slot])
            grad[bank, slot] = np.fft.irfft(product)[..., :TEETH]
            error = multiply(factors[bank, slot], error)
        if bank:
            error = error[:, (LANE + TOOTH) % LANES, TOOTH]
    return prediction, grad


def episode(rng, count):
    vectors = rng.normal(size=(count, LANES, WIDTH))
    key = vectors[0]
    key /= np.linalg.norm(key)
    if count > 1:
        others = vectors[1:].reshape(count - 1, -1)
        others -= (others @ key.ravel())[:, None] * key.ravel()
        others /= np.linalg.norm(others, axis=1, keepdims=True)
    value = rng.normal(size=(LANES, WIDTH))
    return vectors, value / np.linalg.norm(value)


def verify(rng, weights, factors):
    fields, target = episode(rng, 1)
    prediction, gradient = gradients(factors, fields[0], target)
    np.testing.assert_allclose(np.linalg.norm(prediction), 1, atol=1e-12)
    direction = rng.normal(size=weights.shape)
    # Normalize each lane's tap perturbation to keep the signal above roundoff.
    direction /= np.linalg.norm(direction, axis=-1, keepdims=True)
    step = 1e-5
    delta = spectrum(direction)
    plus = read(factors + step * delta, fields)[0] - target
    minus = read(factors - step * delta, fields)[0] - target
    numeric = (np.sum(minus**2) - np.sum(plus**2)) / (4 * step)
    analytic = np.sum(gradient * direction)
    np.testing.assert_allclose(numeric, analytic, rtol=1e-6, atol=1e-8)
    return abs(numeric - analytic)


def eligible(gradient, turn, scheduled):
    if not scheduled:
        return gradient
    # One common clock address in each bank; it cycles through all slots.
    chosen = np.zeros_like(gradient)
    slot = turn % gradient.shape[1]
    chosen[:, slot] = gradient[:, slot]
    return chosen


def assess(shuffles, depth, trials):
    banks = shuffles + 1
    weights = program(np.random.default_rng(1472), banks, depth)
    factors = spectrum(weights)
    error = verify(np.random.default_rng(564), weights, factors)
    print(f"{shuffles} shuffles, {depth} factors per bank; gradient FD absolute error {error:.3g}", flush=True)
    for scheduled in (True, False):
        count = banks if scheduled else banks * depth
        rates = np.array([.125, .25, .5, .75, 1, 1.25, 1.5, 2]) * LANES / count
        calibration = np.zeros_like(rates)
        rng = np.random.default_rng(573)
        # Every episode starts at the same old matrix, with different key/target and slot.
        for turn in range(12):
            fields, target = episode(rng, 1)
            _, grad = gradients(factors, fields[0], target)
            direction = spectrum(eligible(grad, turn, scheduled))
            for index, rate in enumerate(rates):
                updated = read(factors + rate * direction, fields)[0]
                calibration[index] += np.sum((updated - target)**2)
        rate = rates[np.argmin(calibration)]
        totals = np.zeros(5)
        rng = np.random.default_rng(901)
        for turn in range(trials):
            fields, target = episode(rng, 9)
            prediction, grad = gradients(factors, fields[0], target)
            direction = spectrum(eligible(grad, turn, scheduled))
            before = read(factors, fields)
            after = read(factors + rate * direction, fields)
            change = after - before
            totals += [np.sum((target - prediction)**2), np.sum((after[0] - target)**2),
                       np.sum(change[1:]**2) / (len(fields)-1), np.sum(change[0]**2),
                       np.sum(target**2)]
        name = "one slot per bank" if scheduled else "all factors"
        base, outcome, other, intended, target_norm = totals
        print(f"  {name:17s} rate {rate:.4g}, residual remaining {np.sqrt(outcome/base):.1%}, "
              f"field error / target {np.sqrt(outcome/target_norm):.1%}, "
              f"orthogonal-query change / on-key change {np.sqrt(other/intended):.1%}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shuffles", nargs="+", type=int, default=[1, 3])
    parser.add_argument("--depth", type=int, default=34)
    parser.add_argument("--trials", type=int, default=32)
    args = parser.parse_args()
    if min(args.shuffles) < 0 or args.depth < 1 or args.trials < 1:
        parser.error("use nonnegative shuffle counts and positive depth and trials")
    for n in args.shuffles:
        assess(n, args.depth, args.trials)
