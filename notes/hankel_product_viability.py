"""Check local matrix control and one-example writes on the 34-pass signed-real FWM model."""

import argparse

import numpy as np
from scipy.linalg import svdvals

from hankel_online_recall import N, TEETH, INDICES, episodes


DEPTH = 34


def initial_factors(seed=216, *, width=N, depth=DEPTH):
    """Make distinct real orthogonal Hankels on alternating periodic and antiperiodic grids."""
    rng = np.random.default_rng(seed)
    teeth = 2 * width - 1
    bins = np.arange(width)
    balanced = np.empty((depth, teeth))
    for layer in range(depth):
        offset = layer % 2
        angles = 2 * np.pi * (bins + offset / 2) / width
        phases = np.empty(width, complex)
        for k in range(width):
            partner = (-k - offset) % width
            if partner > k:
                phases[k] = np.exp(1j * rng.uniform(-np.pi, np.pi))
                phases[partner] = phases[k].conj()
            elif partner == k:
                phases[k] = rng.choice([-1., 1.])
        balanced[layer] = (np.exp(1j * np.outer(np.arange(teeth), angles)) @ phases / width).real
    factors = balanced[:, np.add.outer(bins, bins)]
    np.testing.assert_allclose(factors @ factors.transpose(0, 2, 1),
                               np.broadcast_to(np.eye(width), factors.shape), atol=1e-12)

    # Break exact orthogonality at unchanged per-comb energy.
    norm = np.linalg.norm(balanced, axis=1, keepdims=True)
    perturbation = np.random.default_rng(seed+1).normal(size=balanced.shape)
    shifted = balanced + 0.05 * norm * perturbation / np.linalg.norm(perturbation, axis=1, keepdims=True)
    shifted *= norm / np.linalg.norm(shifted, axis=1, keepdims=True)
    return {"orthogonal": balanced, "perturbed": shifted}


def prefixes_and_suffixes(weights):
    factors = weights[:, INDICES]
    prefixes = [np.eye(N)]
    for factor in factors:
        prefixes.append(factor @ prefixes[-1])
    suffixes = [np.eye(N)]
    for factor in factors[::-1]:
        suffixes.append(suffixes[-1] @ factor)
    return prefixes, suffixes[::-1]


def jacobian_spectrum(prefixes, suffixes):
    # Column (l,s) is vec(suffix[l+1] @ anti_diagonal_mask[s] @ prefix[l]).
    jacobian = np.empty((N*N, DEPTH*TEETH), order="F")
    for layer in range(DEPTH):
        left = np.fft.rfft(suffixes[layer+1], n=2*N, axis=1)
        right = np.fft.rfft(prefixes[layer], n=2*N, axis=0).T
        block = np.fft.irfft(left[:, None, :] * right[None, :, :], n=2*N, axis=-1)
        jacobian[:, layer*TEETH:(layer+1)*TEETH] = block[:, :, :TEETH].reshape(N*N, TEETH)
    return svdvals(jacobian, overwrite_a=True, check_finite=False)


def best_scalar_write_errors(prefixes, suffixes, tooth_scale):
    """Minimum relative matrix error for a scalar multiple of the factor-gradient direction."""
    rng = np.random.default_rng(218)
    errors = []
    for _ in range(64):
        key, residual = rng.normal(size=(2, N))
        key /= np.linalg.norm(key)
        residual /= np.linalg.norm(residual)
        target = np.outer(residual, key)
        induced = np.zeros_like(target)
        for before, after in zip(prefixes[:-1], suffixes[1:]):
            increment = tooth_scale * np.convolve(before @ key, after.T @ residual)
            induced += after @ increment[INDICES] @ before
        scale = np.vdot(target, induced) / np.vdot(induced, induced)
        errors.append(np.linalg.norm(target - scale * induced))
    return np.quantile(errors, [0.1, 0.5, 0.9])


def recall_test(initial, data, rate, tooth_scale=1.):
    """Read all 64 keys after one write per association. rate=None runs the dense delta control."""
    keys, targets, _ = data
    starting_map = prefixes_and_suffixes(initial)[0][-1]
    errors, recalls = [], []
    for key_stream, target_stream in zip(keys, targets):
        if rate is None:
            matrix = starting_map.copy()
            for key, target in zip(key_stream, target_stream):
                matrix += np.outer(target - matrix @ key, key)
        else:
            weights = initial.copy()
            for key, target in zip(key_stream, target_stream):
                activations = [key]
                for factor in weights:
                    activations.append(np.convolve(factor, activations[-1][::-1], "valid"))
                error = target - activations[-1]
                for layer in reversed(range(DEPTH)):
                    delta = np.convolve(activations[layer], error)
                    if layer:
                        error = np.convolve(weights[layer], error[::-1], "valid")
                    weights[layer] += rate * tooth_scale * delta
                if not np.isfinite(weights).all():
                    break
            if not np.isfinite(weights).all():
                errors.append(np.inf)
                recalls.append(0.)
                continue
            matrix = prefixes_and_suffixes(weights)[0][-1]
        prediction = key_stream @ matrix.T
        errors.append(np.linalg.norm(prediction - target_stream) / np.linalg.norm(target_stream))
        nearest = np.argmax(prediction @ target_stream.T, axis=1)
        recalls.append(np.mean(nearest == np.arange(N)))
    return np.array(errors), np.array(recalls)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--spectrum", action="store_true", help="Compute both full 4096 by 4318 Jacobian spectra.")
    parser.add_argument("--seed", type=int, default=216, help="Seed for the initial factor program.")
    parser.add_argument("--map-scale", type=float, default=1., help="Initial gain of the whole matrix.")
    args = parser.parse_args()
    if not np.isfinite(args.map_scale) or args.map_scale <= 0:
        parser.error("--map-scale must be finite and positive")
    calibration = episodes("associations", 6, 219)
    held_out = episodes("associations", 24, 220)
    rates = np.array([0.125, 0.25, 0.5, 0.75, 1, 1.5, 2, 2.5, 3, 4, 6, 8, 12, 16]) / DEPTH
    if args.map_scale != 1.:
        tangent_scale = args.map_scale ** (1 - 1/DEPTH)
        extra_rates = np.array([.002, .01, .04, .125, .25, .5, .75, 1, 2, 4]) / (DEPTH * tangent_scale**2)
        rates = np.union1d(rates, extra_rates)
    counts = np.convolve(np.ones(N), np.ones(N))
    # Dividing by multiplicity projects a dense gradient onto an isolated Hankel.
    # The common multiplier keeps the learning-rate grid comparable.
    write_rules = {"factor": 1., "multiplicity-corrected": counts.mean()/counts}
    for label, initial in initial_factors(args.seed).items():
        initial = initial * args.map_scale**(1/DEPTH)
        prefixes, suffixes = prefixes_and_suffixes(initial)
        print(label, "forward condition", np.linalg.cond(prefixes[-1]), flush=True)
        if args.spectrum:
            s = jacobian_spectrum(prefixes, suffixes)
            print("Jacobian singular value min/median/max", np.quantile(s, [0., 0.5, 1.]),
                  "ranks at relative 1e-10, 1e-3, 1e-2",
                  [np.count_nonzero(s > t * s[0]) for t in [1e-10, 1e-3, 1e-2]], flush=True)
        with np.errstate(over="ignore", invalid="ignore"):
            for method, scale in write_rules.items():
                print(method, "best scalar relative matrix-write error p10/50/90",
                      best_scalar_write_errors(prefixes, suffixes, scale), flush=True)
                trials = [recall_test(initial, calibration, rate, scale) for rate in rates]
                errors_by_rate = np.array([errors.mean() for errors, _ in trials])
                recalls_by_rate = np.array([recalls.mean() for _, recalls in trials])
                stable = np.isfinite(errors_by_rate)
                by_error = np.argmin(np.where(stable, errors_by_rate, np.inf))
                by_recall = np.argmax(np.where(stable, recalls_by_rate, -np.inf))
                for criterion, index in [("vector error", by_error), ("top-1", by_recall)]:
                    errors, recalls = recall_test(initial, held_out, rates[index], scale)
                    print(method, "tuned for", criterion, "step", rates[index], "relative value error p10/50/90",
                          np.quantile(errors, [0.1, 0.5, 0.9]), "top-1", recalls.mean(), flush=True)
            errors, recalls = recall_test(initial, held_out, None)
            print("dense relative value error median", np.median(errors), "top-1", recalls.mean(), flush=True)


if __name__ == "__main__":
    main()
