"""Summarize paired evaluation articles and plot the synthetic and LM tests."""

import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def records(path):
    with path.open() as file:
        return list(csv.DictReader(file))


def language_results(directory):
    dense = sorted((r for r in records(directory / "lm_dense.csv") if r["split"] == "test"),
                   key=lambda r: int(r["row"]))
    ids = [r["row"] for r in dense]
    reference = np.array([float(r["nll"]) for r in dense])
    bootstrap = np.random.default_rng(134).integers(len(ids), size=(10000, len(ids)))
    results = []
    for file in sorted(directory.glob("lm_*.csv")):
        if file.name == "lm_summary.csv":
            continue
        rows = records(file)
        for split in ("test", "test_frozen"):
            test = sorted((r for r in rows if r["split"] == split), key=lambda r: int(r["row"]))
            if not test:
                continue
            assert [r["row"] for r in test] == ids and len(test) == 12
            assert all(r["predictions"] == "1087" for r in test)
            nll = np.array([float(r["nll"]) for r in test])
            excess = nll - reference
            lower, upper = np.quantile(excess[bootstrap].mean(axis=1), (.025, .975))
            results.append(dict(mode=test[0]["mode"], split=split, step=test[0]["step"],
                                articles=len(test), predictions=1087*len(test), nll=nll.mean(),
                                perplexity=np.exp(nll.mean()), excess_nll=excess.mean(),
                                paired_p025=lower, paired_p975=upper,
                                late_nll=np.mean([float(r["nll_after_511"]) for r in test])))
    with (directory / "lm_summary.csv").open("w", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    return {r["mode"]: r for r in results if r["split"] == "test"}


def plot(directory, language):
    plt.rcParams.update({"font.size": 10, "axes.spines.right": False, "axes.spines.top": False,
                         "svg.fonttype": "none", "font.family": "DejaVu Sans"})
    fig, (left, right) = plt.subplots(1, 2, figsize=(11, 3.75), layout="constrained",
                                      gridspec_kw={"width_ratios": (1.08, 1.)})
    measurements = {(r["task"], r["model"]): r for r in records(directory / "online_recall.csv")
                    if r["writes"] == "64"}
    x = np.arange(3)
    tasks = ("toeplitz", "eight_pair", "rotated_eight_pair")
    for method, color, offset, label in (("dense", "#485b75", -.15, "Dense delta"),
                                          ("hankel8", "#08796e", .15, "Eight Hankel pairs")):
        rows = [measurements[t, method] for t in tasks]
        mid = 100 * np.array([float(r["median_error"]) for r in rows])
        low = 100 * np.array([float(r["p10_error"]) for r in rows])
        high = 100 * np.array([float(r["p90_error"]) for r in rows])
        left.bar(x+offset, mid, width=.27, color=color, label=label)
        left.errorbar(x+offset, mid, yerr=[mid-low, high-mid], fmt="none", color="#161c24", capsize=3)
    left.set(xticks=x, xticklabels=("Toeplitz", "Eight pairs", "Rotated\neight pairs"),
             ylabel="Relative field error (%)", ylim=(0, 110))
    left.legend(loc="upper left", frameon=False)
    left.set_title("Learning a new map from 64 writes", loc="left", fontsize=12, weight="bold")
    modes = (("dense", "Dense TTT"), ("projected32_only", "32 key coordinates"),
             ("hankel_only", "Eight pairs"), ("hankel_calibrated", "Eight pairs, calibrated"),
             ("hankel_aligned", "Eight pairs, rotated"),
             ("bias_only", "Frozen dense matrix"))
    for i, (mode, label) in enumerate(modes):
        r = language[mode]
        color = "#08796e" if mode.startswith("hankel") else "#485b75"
        right.errorbar(r["excess_nll"], i,
                      xerr=[[r["excess_nll"]-r["paired_p025"]], [r["paired_p975"]-r["excess_nll"]]],
                      fmt="o", color=color, capsize=3, markersize=7)
    right.set(yticks=range(len(modes)), yticklabels=[l for _, l in modes],
              xlabel="Extra next-token loss vs dense TTT (nats/token)", ylim=(len(modes)-.5, -.5))
    right.axvline(0, color="#cccccc", linewidth=.8, zorder=0)
    right.set_title("Pretrained TTT on WikiText-2", loc="left", fontsize=12, weight="bold")
    for ax, axis in ((left, "y"), (right, "x")):
        ax.grid(axis=axis, alpha=.19)
        ax.set_axisbelow(True)
    fig.savefig(directory / "memory_comparison.svg", metadata={"Date": None})
    plt.close(fig)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    directory = parser.parse_args().directory
    plot(directory, language_results(directory))
