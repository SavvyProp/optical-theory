"""Single-pass delta writes, with both full-band Hankel factors adaptive.

Examples have unit-norm keys. The dense and factor learners see exactly the
same observations, in the same order. A separate set of episodes selects one
scalar step size for each learner/task. No fit uses test episodes.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
from scipy.fft import irfft, rfft


N, B = 64, 8
TEETH = 2 * N - 1
INDICES = np.add.outer(np.arange(N), np.arange(N))
DIAGONALS = np.subtract.outer(np.arange(N), np.arange(N)) + N - 1
CHECKPOINTS = (8, 16, 32, 64)
STEPS = (0.025, 0.05, 0.1, 0.2, 0.35, 0.5, 0.75, 1., 1.25)
MODELS = ("dense", "rank16", "toeplitz", "hankel8", "hankel8_shift_prior")
TASKS = ("associations", "overwrites", "toeplitz", "rotated_toeplitz", "eight_pair", "rotated_eight_pair")


def unit(vectors):
    return vectors / np.linalg.norm(vectors, axis=-1, keepdims=True)


def qr_basis(rng, shape):
    q, r = np.linalg.qr(rng.normal(size=shape))
    return q * np.sign(np.diagonal(r, axis1=-2, axis2=-1))[..., None, :]


def episodes(task, count, seed):
    rng = np.random.default_rng(seed)
    if task in ("associations", "overwrites"):
        keys = qr_basis(rng, (count, N, N))
        if task == "overwrites":
            keys[:, 32:] = keys[:, :32]
        values = unit(rng.normal(size=keys.shape))
        return keys, values, None
    if task.endswith("toeplitz"):
        teacher = rng.normal(size=(count, TEETH))[:, DIAGONALS]
    else:
        a, b = rng.normal(size=(2, count, B, TEETH))
        teacher = (b[:, :, INDICES] @ a[:, :, INDICES]).sum(axis=1)
    # The original and rotated tasks share singular values and key streams.
    left = qr_basis(rng, (count, N, N))
    right = qr_basis(rng, (count, N, N))
    if task.startswith("rotated"):
        teacher = left @ teacher @ right
    teacher *= (np.sqrt(N) / np.linalg.norm(teacher, axis=(1, 2)))[:, None, None]
    keys = unit(rng.normal(size=(count, N, N)))
    return keys, keys @ teacher.transpose(0, 2, 1), teacher


def hankel_read(weights, vectors):
    # H(h)x is the central n entries of conv(h, reverse(x)).
    return irfft(rfft(weights, 192) * rfft(vectors[..., ::-1], 192), 192)[..., N-1:2*N-1]


def convolution(x, y):
    return irfft(rfft(x, 128) * rfft(y, 128), 128)[..., :TEETH]


class Learner:
    def __init__(self, name, episodes_count):
        self.name = name
        if name == "dense":
            self.state = np.zeros((episodes_count, N, N))
        elif name == "toeplitz":
            self.state = np.zeros((episodes_count, TEETH))
        else:
            rng = np.random.default_rng(5151)
            branches = 16 if name == "rank16" else B
            width = N if name == "rank16" else TEETH
            a, b = rng.normal(size=(2, branches // 2, width)) / np.sqrt(N)
            scale = np.full((branches // 2, 1), 1 / np.sqrt(2 * branches))
            if name == "rank16":
                scale *= np.sqrt(N)
            if name == "hankel8_shift_prior":
                # One canceling pair of branches receives 75% of expected reference
                # energy. Its two matrices per branch initially reverse the comb.
                a[0] = b[0] = 0
                a[0, N-1] = b[0, N-1] = 1
                scale[0] = np.sqrt(0.75 / 4)
                scale[1:] = np.sqrt(0.25 / (2 * (branches - 2)))
            a, b = a * scale, b * scale
            self.a = np.tile(np.concatenate((a, a)), (episodes_count, 1, 1))
            self.b = np.tile(np.concatenate((b, -b)), (episodes_count, 1, 1))

    def write(self, key, value, eta):
        if self.name == "dense":
            error = value - (self.state @ key[..., None])[..., 0]
            self.state += eta * error[:, :, None] * key[:, None, :]
        elif self.name == "toeplitz":
            error = value - hankel_read(self.state, key[:, ::-1])
            self.state += eta * convolution(error, key[:, ::-1])
        elif self.name == "rank16":
            z = (self.a * key[:, None]).sum(axis=-1)
            error = value - (self.b * z[..., None]).sum(axis=1)
            backward = (self.b * error[:, None]).sum(axis=-1)
            self.a += eta * backward[..., None] * key[:, None]
            self.b += eta * z[..., None] * error[:, None]
        else:
            z = hankel_read(self.a, key[:, None])
            error = value - hankel_read(self.b, z).sum(axis=1)
            backward = hankel_read(self.b, error[:, None])
            self.a += eta * convolution(backward, key[:, None])
            self.b += eta * convolution(error[:, None], z)

    def matrix(self):
        if self.name == "dense":
            return self.state
        if self.name == "toeplitz":
            return self.state[:, DIAGONALS]
        if self.name == "rank16":
            return self.b.transpose(0, 2, 1) @ self.a
        return (np.ascontiguousarray(self.b[:, :, INDICES]) @
                np.ascontiguousarray(self.a[:, :, INDICES])).sum(axis=1)


def run(data, name, task, eta):
    keys, values, teacher = data
    learner = Learner(name, len(keys))
    results = []
    for t in range(N):
        learner.write(keys[:, t], values[:, t], eta)
        if t+1 not in CHECKPOINTS:
            continue
        current = learner.matrix()
        if teacher is None:
            start = 32 if task == "overwrites" and t+1 == 64 else 0
            target = values[:, start:t+1]
            prediction = keys[:, start:t+1] @ current.transpose(0, 2, 1)
            choices = np.argmax(prediction @ target.transpose(0, 2, 1), axis=-1)
            top1 = (choices == np.arange(target.shape[1])).mean(axis=-1)
        else:
            prediction, target = current, teacher
            top1 = np.full(len(keys), np.nan)
        error = np.linalg.norm(prediction-target, axis=(1, 2)) / np.linalg.norm(target, axis=(1, 2))
        results.append((error, top1))
    return results


def benchmark(output):
    with output.open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("task", "model", "writes", "step", "episodes", "failures", "p10_error", "median_error",
                         "p90_error", "mean_top1"))
        for task in TASKS:
            calibration = episodes(task, 12, 101)
            evaluation = episodes(task, 32, 107)
            for name in MODELS:
                with np.errstate(over="ignore", invalid="ignore"):
                    scores = []
                    for eta in STEPS:
                        outcomes = run(calibration, name, task, eta)
                        score = np.mean([out[0].mean() for out in outcomes])
                        scores.append(score if np.isfinite(score) else np.inf)
                eta = STEPS[np.argmin(scores)]
                with np.errstate(over="ignore", invalid="ignore"):
                    results = run(evaluation, name, task, eta)
                for writes, (err, top1) in zip(CHECKPOINTS, results):
                    failed = ~np.isfinite(err)
                    err[failed] = np.inf
                    # Count divergent runs at worst error, including them in the denominator.
                    quantiles = np.quantile(err, (.1,.5,.9), method="nearest" if failed.any() else "linear")
                    accuracy = "" if np.isnan(top1).all() else np.where(failed, 0, top1).mean()
                    writer.writerow((task, name, writes, eta, len(err), failed.sum(), *quantiles, accuracy))
                file.flush()
            print(f"Finished {task}", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("output", type=Path)
    benchmark(parser.parse_args().output)
