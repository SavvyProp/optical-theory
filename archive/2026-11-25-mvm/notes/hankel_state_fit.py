"""Measure Hankel expressivity on saved 64-by-64 TTT states and real queries.

Run after ttt_state_probe.py with one BLAS thread per process. Fits have access
to the entire past state. Future queries are evaluation data only.
"""

import argparse
import csv
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.optimize import minimize


WIDTH = 64
BRANCHES = 8
BUDGETS = (4, 8, 12, 16)
TEETH = 2 * WIDTH - 1
INDICES = np.add.outer(np.arange(WIDTH), np.arange(WIDTH))
BIN_INDICES = (INDICES + TEETH * np.arange(BRANCHES)[:, None, None]).ravel()


def pad(vector):
    return np.pad(vector, (WIDTH - 1, 0))


def shift(vector):
    return np.r_[0, vector[:-1]]


def pair_fields(parameters):
    a, b = parameters.reshape(2, BRANCHES, TEETH)
    return np.ascontiguousarray(a[:, INDICES]), np.ascontiguousarray(b[:, INDICES])


def matrix_of(parameters):
    first, second = pair_fields(parameters)
    return (second @ first).sum(axis=0)


def fit_objective(parameters, target, covariance=None):
    first, second = pair_fields(parameters)
    error = (second @ first).sum(axis=0) - target
    weighted = error if covariance is None else error @ covariance
    gradients = (second.transpose(0, 2, 1) @ weighted, weighted @ first.transpose(0, 2, 1))
    grad = np.concatenate([np.bincount(BIN_INDICES, weights=g.ravel()) for g in gradients])
    return 0.5 * np.sum(error * weighted), grad


def feasible_starts(target):
    # Two Hankel pairs implement each rank-one SVD term, including its sign.
    left, singular, right = np.linalg.svd(target, full_matrices=False)
    first, second = [], []
    for u, s, v in zip(left.T[:BRANCHES // 2], singular, right):
        u, v = u * np.sqrt(s), v * np.sqrt(s)
        first.extend((pad(v), pad(shift(v))))
        second.extend((pad(u), -pad(shift(u))))
    yield np.array([first, second])

    # G = M - Z M Z^T. Integrating each rank-one generator yields P(u) P(v).
    stein = target.copy()
    stein[1:, 1:] -= target[:-1, :-1]
    left, singular, right = np.linalg.svd(stein, full_matrices=False)
    yield np.array([[pad(v * np.sqrt(s)) for v, s in zip(right[:BRANCHES], singular)],
                    [pad(u * np.sqrt(s)) for u, s in zip(left.T[:BRANCHES], singular)]])


def fit_parameters(target, iterations, covariance=None):
    scale = np.linalg.norm(target) / np.sqrt(WIDTH)
    scaled = target / scale
    best, best_loss = None, np.inf
    for initial in feasible_starts(scaled):
        result = minimize(fit_objective, initial.ravel(), args=(scaled, covariance), jac=True,
                          method="L-BFGS-B", options={"maxiter": iterations, "ftol": 1e-11})
        if result.fun < best_loss:
            best, best_loss = result.x, result.fun
    return best.reshape(2, BRANCHES, TEETH) * np.sqrt(scale)


def fit_state(target, iterations, covariance=None):
    return matrix_of(fit_parameters(target, iterations, covariance))


def fit_prior(matrix):
    return fit_parameters(matrix.astype(float), 180)


def fit_calibrated_prior(pair):
    matrix, covariance = pair
    return fit_parameters(matrix.astype(float), 180, covariance)


def ordered_basis(covariance):
    eigenvalues, basis = np.linalg.eigh(covariance)
    basis = basis[:, ::-1]
    pivots = np.abs(basis).argmax(axis=0)
    basis = basis * np.sign(basis[pivots, np.arange(len(basis))])
    return eigenvalues[::-1], basis


def calibration_bases(changes, queries):
    # Query and value coordinates are shared across passages. Learn the
    # two orders without access to the evaluation article or its queries.
    covariance = np.einsum("dtk,dtj->kj", queries, queries, optimize=True)
    covariance *= WIDTH / np.trace(covariance)
    eigenvalues, key_basis = ordered_basis(covariance)
    covariance = 0.95 * covariance + 0.05 * np.eye(WIDTH)
    read_covariance = (changes @ covariance @ changes.transpose(0, 2, 1)).sum(axis=0)
    _, value_basis = ordered_basis(read_covariance)
    return key_basis, value_basis, 0.95 * eigenvalues + 0.05


def calibrated_fit(args):
    meta, delta, queries, bases, iterations = args
    key_basis, value_basis, weights = bases
    target = value_basis.T @ delta @ key_basis
    q = key_basis.T @ queries.T
    answer = target @ q
    aligned = fit_state(target, iterations, np.diag(weights))
    native_covariance = (key_basis * weights) @ key_basis.T
    native = fit_state(delta, iterations, native_covariance)
    # Exact minimum of the regularized query-weighted rank-16 objective.
    u, s, vt = np.linalg.svd(target * np.sqrt(weights))
    low_rank = ((u[:, :16] * s[:16]) @ vt[:16]) / np.sqrt(weights)
    # Only the 64x32 state is writable in this last baseline.
    projected = target[:, :32] @ q[:32]
    return {**meta, "hankel_native_weighted": relative(native @ queries.T - delta @ queries.T, delta @ queries.T),
            "hankel_aligned_weighted": relative(aligned @ q - answer, answer),
            "hankel_aligned_matrix": relative(aligned-target, target),
            "rank16_weighted": relative(low_rank @ q - answer, answer),
            "fixed32_read": relative(projected - answer, answer)}


def spectrum(matrix):
    s = np.linalg.svd(matrix, compute_uv=False)
    return s, int(np.searchsorted(np.cumsum(s**2), 0.95 * np.sum(s**2)) + 1)


def relative(error, target):
    return np.linalg.norm(error) / np.linalg.norm(target)


def score_one(args):
    meta, delta, queries, iterations = args
    delta = delta.astype(np.float64)
    queries = queries.astype(np.float64)
    fit = fit_state(delta, iterations)
    left, singular, right = np.linalg.svd(delta, full_matrices=False)
    low_rank = (left[:, :16] * singular[:16]) @ right[:16]
    floor = np.linalg.norm(spectrum(delta[1:, 1:] - delta[:-1, :-1])[0][16:]) / (2 * np.linalg.norm(delta))
    fit_error = relative(fit - delta, delta)
    assert fit_error + 1e-9 >= floor
    heldout = delta @ queries.T
    return {**meta, "hankel_lower": floor, "hankel_matrix": fit_error,
            "hankel_read": relative((fit - delta) @ queries.T, heldout),
            "rank16_matrix": relative(low_rank - delta, delta),
            "rank16_read": relative((low_rank - delta) @ queries.T, heldout)}


def summarize(records, keys, measures):
    buckets = {}
    for record in records:
        group = tuple(record[k] for k in keys)
        buckets.setdefault(group, []).append(record)
    for group, rows in sorted(buckets.items()):
        for name in measures:
            values = np.array([r[name] for r in rows])
            lo, median, hi = np.quantile(values, (0.1, 0.5, 0.9))
            yield dict(zip(keys, group), metric=name, count=len(rows), p10=lo, median=median, p90=hi)


def write_csv(path, rows):
    rows = iter(rows)
    first = next(rows)
    with path.open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(first))
        writer.writeheader()
        writer.writerow(first)
        writer.writerows(rows)


def main(args):
    args.output.mkdir(parents=True, exist_ok=True)
    if args.priors or args.prior_metrics:
        with np.load(args.states) as saved:
            priors = saved["priors"]
        matrices = priors.reshape(-1, WIDTH, WIDTH)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            if args.prior_metrics:
                metrics = np.load(args.prior_metrics).reshape(-1, WIDTH, WIDTH)
                if args.frames:
                    key_frames, value_frames = np.load(args.frames).reshape(-1, 2, WIDTH, WIDTH).transpose(1, 0, 2, 3)
                    matrices = value_frames.transpose(0, 2, 1) @ matrices @ key_frames
                    metrics = key_frames.transpose(0, 2, 1) @ metrics @ key_frames
                factors = list(pool.map(fit_calibrated_prior, zip(matrices, metrics), chunksize=2))
            else:
                factors = list(pool.map(fit_prior, matrices, chunksize=2))
        name = "initial_calibrated.npy" if args.prior_metrics else "initial_factors.npy"
        if args.frames:
            name = "initial_aligned.npy"
        np.save(args.output / name,
                np.array(factors, dtype=np.float32).reshape(*priors.shape[:2], 2, BRANCHES, TEETH))
        return
    with np.load(args.states) as saved:
        delta, queries, priors = saved["delta"], saved["queries"], saved["priors"]
        metadata = [dict(row=int(r), layer=int(l), head=int(h), prefix=int(t))
                    for r, l, h, t in zip(*(saved[k] for k in ("row", "layer", "head", "prefix")))]
    if args.calibrated:
        rows = sorted(set(m["row"] for m in metadata))
        folds = {row: i % 3 for i, row in enumerate(rows)}
        jobs = []
        for layer in (0, 5, 11):
            for head in (0, 6):
                group = [i for i, m in enumerate(metadata) if (m["layer"], m["head"]) == (layer, head)]
                for fold in range(3):
                    train = [i for i in group if folds[metadata[i]["row"]] != fold]
                    bases = calibration_bases(delta[train].astype(float), queries[train].astype(float))
                    jobs.extend((metadata[i], delta[i].astype(float), queries[i].astype(float), bases,
                                 args.iterations) for i in group if folds[metadata[i]["row"]] == fold)
        with ProcessPoolExecutor(max_workers=args.workers) as pool:
            fits = list(pool.map(calibrated_fit, jobs, chunksize=2))
        write_csv(args.output / "ttt_calibrated.csv", fits)
        write_csv(args.output / "ttt_calibrated_summary.csv", summarize(
            fits, ("prefix",), ("hankel_native_weighted", "hankel_aligned_weighted",
                                "hankel_aligned_matrix", "rank16_weighted", "fixed32_read")))
        return
    spectral_records = []
    for meta, change in zip(metadata, delta):
        full = change + priors[meta["layer"], meta["head"]]
        for component, target in (("change", change), ("full", full)):
            s, r95 = spectrum(target)
            d, _ = spectrum(target[1:, 1:] - target[:-1, :-1])
            norm = np.linalg.norm(target)
            spectral_records.append({**meta, "component": component, "r95": r95,
                                     "rank16_matrix": np.linalg.norm(s[16:]) / norm,
                                     **{f"floor_{b}pair": np.linalg.norm(d[2*b:]) / (2*norm)
                                        for b in BUDGETS}})
    write_csv(args.output / "ttt_spectra.csv", summarize(
        spectral_records, ("prefix", "component"),
        ("r95", "rank16_matrix", *(f"floor_{b}pair" for b in BUDGETS))))
    if args.spectra_only:
        return
    jobs = [(m, delta[i], queries[i], args.iterations) for i, m in enumerate(metadata)
            if m["layer"] in (0, 5, 11) and m["head"] in (0, 6)]
    print(f"Fitting {len(jobs)} recurrent changes, {args.iterations} iterations per start.", flush=True)
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        fits = list(pool.map(score_one, jobs, chunksize=2))
    write_csv(args.output / "ttt_fits.csv", fits)
    write_csv(args.output / "ttt_fit_summary.csv", summarize(
        fits, ("prefix",), ("hankel_lower", "hankel_matrix", "rank16_matrix", "hankel_read", "rank16_read")))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("states", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--iterations", type=int, default=180)
    parser.add_argument("--calibrated", action="store_true",
                        help="Evaluate query-weighted fits with three-fold, article-disjoint calibration.")
    parser.add_argument("--spectra-only", action="store_true", help="Write the spectra summary alone.")
    parser.add_argument("--priors", action="store_true",
                        help="Fit each pretrained W0 by eight pairs and save them for the online ablation.")
    parser.add_argument("--prior-metrics", type=Path,
                        help="Fit W0 using head-specific validation key/query covariances.")
    parser.add_argument("--frames", type=Path, help="Apply fixed input and output frames in a calibrated prior fit.")
    options = parser.parse_args()
    if options.frames and not options.prior_metrics:
        parser.error("--frames requires --prior-metrics.")
    main(options)
