"""Score the pretrained TTT language model with noise on its fast-matrix query read.

Run python notes/ttt_read_noise.py DATA, where DATA has the pinned model and
passages.npz made by ttt_state_probe.py. The 16-token causal update is exact.
Each real noise component is independent Gaussian with standard deviation
equal to the specified fraction of the clean 64-dimensional read's RMS.
Use --prediction-noise to independently perturb the matrix-key product that
supplies the inner update, while keeping weight writes and storage noiseless.
"""

import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from ttt_online_eval import losses
from ttt_state_probe import load_model


def perturb(read, fraction, generator):
    if fraction == 0:
        return read
    sigma = fraction * read.square().mean(dim=-1, keepdim=True).sqrt()
    return read + sigma * torch.randn(read.shape, generator=generator, dtype=read.dtype)


def read_with_noise(reference, fraction, generator, prediction_generator=None):
    def run(layer, inputs, mini_batch_size, last_mini_batch_params_dict, cache_params=None):
        if cache_params is not None or last_mini_batch_params_dict is not None:
            raise ValueError("This probe runs each article from its initial state.")
        # Reference dual form for an entire article with causal 16-token minibatches.
        keys, queries, values = (inputs[name][0] for name in ("XK", "XQ", "XV"))
        weights, bias = layer.W1, layer.b1
        gamma, offset = layer.ttt_norm_weight[:, None], layer.ttt_norm_bias[:, None]
        result = []
        for chunk in range(keys.shape[1]):
            k, q, v = keys[:, chunk], queries[:, chunk], values[:, chunk]
            eta = inputs["eta"][0, :, chunk].tril()
            prediction = k @ weights
            if prediction_generator is not None:
                prediction = perturb(prediction, fraction, prediction_generator)
            grad = reference.ln_fused_l2_bwd(prediction + bias, v - k, gamma, offset)
            # Each token reads the state after its own causal prefix of updates.
            read = q @ weights - (eta * (q @ k.transpose(-2, -1))) @ grad
            read = perturb(read, fraction, generator)
            bias_at_token = bias - eta @ grad
            result.append(q + reference.ln_fwd(read + bias_at_token, gamma, offset))
            last_eta = eta[:, -1, :, None]
            weights = weights - (last_eta * k).transpose(-2, -1) @ grad
            bias = bias - (last_eta * grad).sum(dim=-2, keepdim=True)
        fields = torch.cat(result, dim=1).transpose(0, 1).reshape(1, -1, layer.width)
        return fields, None
    return run


@torch.inference_mode()
def main(directory, fraction, seeds, prediction_noise):
    reference, model = load_model(directory)
    with np.load(directory / "passages.npz") as passages:
        native_ttt = reference.TTTLinear.ttt
        try:
            validation_ids = torch.from_numpy(passages["ids"][:1, :128]).long()
            expected = model(input_ids=validation_ids, use_cache=False).last_hidden_state
            reference.TTTLinear.ttt = read_with_noise(reference, 0, None)
            actual = model(input_ids=validation_ids, use_cache=False).last_hidden_state
            torch.testing.assert_close(actual, expected, atol=8e-5, rtol=1e-4)
            print("Max clean hidden-state difference:", (expected - actual).abs().max().item(), flush=True)

            baseline = np.array(list(losses(model, passages)))
            # Independently reproduce the per-article reference scores.
            with (Path(__file__).parent / "hankel_results/lm_dense.csv").open() as f:
                native = [r for r in csv.DictReader(f) if r["split"] == "test"]
            np.testing.assert_array_equal(baseline[:, 0], [int(r["row"]) for r in native])
            np.testing.assert_allclose(baseline[:, 2], [float(r["nll"]) for r in native], atol=1e-5)
            print(f"Baseline {len(baseline)} articles, {int(baseline[:, 1].sum())} predictions:",
                  baseline[:, 2].mean(), np.exp(baseline[:, 2].mean()), flush=True)

            deltas = []
            for seed in seeds:
                generator = torch.Generator().manual_seed(seed)
                prediction_generator = torch.Generator().manual_seed(seed + 1000000) if prediction_noise else None
                reference.TTTLinear.ttt = read_with_noise(reference, fraction, generator, prediction_generator)
                scored = np.array(list(losses(model, passages)))
                np.testing.assert_array_equal(scored[:, :2], baseline[:, :2])
                differences = scored[:, 2] - baseline[:, 2]
                deltas.append(differences)
                print(f"Read RMS {fraction:.3f}, noisy key product {prediction_noise},"
                      f" seed {seed}, NLL {scored[:, 2].mean():.6f},"
                      f" PPL {np.exp(scored[:, 2].mean()):.4f}, mean NLL difference"
                      f" {differences.mean():+.6f}", flush=True)
            by_article = np.mean(deltas, axis=0)
            rng = np.random.default_rng(943)
            boot = rng.choice(by_article, (20000, len(by_article))).mean(axis=1)
            print("Seed-averaged article NLL differences:", np.round(by_article, 5))
            print("95% paired article bootstrap interval for delta NLL:", np.quantile(boot, [.025, .975]))
        finally:
            reference.TTTLinear.ttt = native_ttt


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory", type=Path)
    parser.add_argument("--fraction", type=float, default=0.10)
    parser.add_argument("--seeds", type=int, nargs="+", default=[1729, 2718])
    parser.add_argument("--prediction-noise", action="store_true")
    args = parser.parse_args()
    main(args.directory, args.fraction, args.seeds, args.prediction_noise)
