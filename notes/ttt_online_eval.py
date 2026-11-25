"""Replace the inner fast matrix in every layer of a pretrained 125M TTT-LM.

Keep the checkpoint's causal 16-token minibatches, bias, layer normalization,
learned gates and outer weights. Tune a single matrix-write multiplier on four
validation articles, then evaluate all 12 disjoint test passages.
"""

import argparse
import csv
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ttt_state_probe import load_model


MODES = ("dense", "bias_only", "projected32", "projected32_only", "rank16_residual",
         "hankel_residual", "hankel_only", "hankel_calibrated", "hankel_aligned")
RATES = {
    "dense": (1.,), "bias_only": (0.,), "projected32": (0.5, 1., 2.),
    "projected32_only": (0.5, 1., 2.),
    "rank16_residual": (0., 0.0625, 0.25, 1., 2.),
    "hankel_residual": (0., 0.0625, 0.25, 1., 2.),
    "hankel_only": (0., 0.015625, 0.0625, 0.25, 1.),
    "hankel_calibrated": (0., 0.015625, 0.0625, 0.25, 1.),
    "hankel_aligned": (0., 0.015625, 0.0625, 0.25, 1.),
}


def hankel_read(fields, vectors):
    result = torch.fft.irfft(torch.fft.rfft(fields, 192) *
                             torch.fft.rfft(vectors.flip(-1), 192), 192)
    return result[..., 63:127]


def convolve(left, right):
    return torch.fft.irfft(torch.fft.rfft(left, 128) * torch.fft.rfft(right, 128), 128)[..., :127]


def starting_factors(layer, mode, priors):
    if mode in ("hankel_only", "hankel_calibrated", "hankel_aligned"):
        pairs = priors[layer.layer_idx]
        return pairs[:, 0], pairs[:, 1]
    heads, width, _ = layer.W1.shape
    rank = 16 if mode == "rank16_residual" else 8
    length = width if mode == "rank16_residual" else 2 * width - 1
    rng = torch.Generator().manual_seed(700 + layer.layer_idx)
    a, b = torch.randn(2, heads, rank // 2, length, generator=rng) / np.sqrt(2 * rank)
    if mode == "hankel_residual":
        a, b = a / np.sqrt(width), b / np.sqrt(width)
    return torch.cat((a, a), dim=1), torch.cat((b, -b), dim=1)


def prefix_update(state, gradients, lr, token_scale, multiplier):
    return state[:, None] - multiplier * token_scale[..., None] * (gradients * lr[..., None]).cumsum(dim=1)


def fast_layer(reference, layer, inputs, mode, step, bases, priors):
    # The checkpoint uses row vectors: x @ W1. The Hankel routines implement
    # the equivalent column map sum H(b)H(a)x.
    q_all, k_all, v_all = (inputs[k][0] for k in ("XQ", "XK", "XV"))
    token_all = inputs["token_eta"][0]
    lr_all = inputs["ttt_lr_eta"][0, :, :, 0, :, None]
    heads, chunks, _, width = q_all.shape
    aligned = mode == "hankel_aligned"
    factor_only = mode in ("hankel_only", "hankel_calibrated", "hankel_aligned")
    if aligned:
        key_frame, value_frame = bases[layer.layer_idx].unbind(dim=1)
    base = torch.zeros_like(layer.W1) if factor_only or mode == "projected32_only" else layer.W1
    bias = layer.b1
    gamma, offset = layer.ttt_norm_weight[:, None], layer.ttt_norm_bias[:, None]
    projected = mode in ("projected32", "projected32_only", "dense_check")
    rank16 = mode == "rank16_residual"
    factored = factor_only or mode in ("rank16_residual", "hankel_residual")
    if projected:
        basis = (torch.eye(width).expand(heads, -1, -1) if mode == "dense_check" else bases[layer.layer_idx])
        memory = (basis.transpose(1, 2) @ layer.W1 if mode == "projected32_only" else
                  torch.zeros(heads, basis.shape[-1], width))
    if factored:
        a, b = starting_factors(layer, mode, priors)
        if step == 0:
            if rank16:
                base = base + a.transpose(1, 2) @ b
            else:
                teeth = torch.arange(width)[:, None] + torch.arange(width)[None, :]
                initial = (b[:, :, teeth] @ a[:, :, teeth]).sum(dim=1).transpose(1, 2)
                base = base + (key_frame @ initial @ value_frame.transpose(1, 2) if aligned else initial)
            factored = False
    outputs = []
    for t in range(chunks):
        q, k, v = q_all[:, t], k_all[:, t], v_all[:, t]
        lr, token = lr_all[:, t], token_all[:, t]
        prediction, read = k @ base + bias, q @ base
        if projected:
            key, query = k @ basis, q @ basis
            prediction = prediction + key @ memory
        if factored:
            if rank16:
                z = k @ a.transpose(1, 2)
                prediction = prediction + z @ b
            else:
                key = k @ key_frame if aligned else k
                query = q @ key_frame if aligned else q
                z = hankel_read(a[:, None], key[:, :, None])
                branch_sum = hankel_read(b[:, None], z).sum(dim=2)
                prediction = prediction + (branch_sum @ value_frame.transpose(1, 2) if aligned else branch_sum)
        grad = reference.ln_fused_l2_bwd(prediction, v-k, gamma, offset)
        weighted = lr * grad
        biases = bias - token * weighted.cumsum(dim=1)
        if projected:
            read = read + query @ memory - step * token * (torch.tril(query @ key.transpose(1, 2)) @ weighted)
            memory = memory - step * token[:, -1:] * (key.transpose(1, 2) @ weighted)
        if factored:
            if rank16:
                ga = (grad @ b.transpose(1, 2))[..., None] * k[:, :, None]
                gb = z[..., None] * grad[:, :, None]
            else:
                branch_grad = grad @ value_frame if aligned else grad
                ga = convolve(hankel_read(b[:, None], branch_grad[:, :, None]), key[:, :, None])
                gb = convolve(z, branch_grad[:, :, None])
            prefix_a = prefix_update(a, ga, lr, token, step)
            prefix_b = prefix_update(b, gb, lr, token, step)
            if rank16:
                scores = (prefix_a * q[:, :, None]).sum(dim=-1)
                read = read + (scores[..., None] * prefix_b).sum(dim=2)
            else:
                branch_read = hankel_read(prefix_b, hankel_read(prefix_a, query[:, :, None])).sum(dim=2)
                read = read + (branch_read @ value_frame.transpose(1, 2) if aligned else branch_read)
            a, b = prefix_a[:, -1], prefix_b[:, -1]
        outputs.append(q + reference.ln_fwd(read + biases, gamma, offset))
        bias = biases[:, -1:]
    fields = torch.cat(outputs, dim=1).transpose(0, 1).reshape(1, -1, heads * width)
    return fields


def losses(model, passages):
    for row, ids in zip(passages["rows"], passages["ids"]):
        ids = torch.from_numpy(ids).long()
        states = model(input_ids=ids[None], use_cache=False).last_hidden_state[0, :-1]
        # This checkpoint ties the output vocabulary weights to the embedding.
        logits = F.linear(states, model.embed_tokens.weight)
        values = F.cross_entropy(logits, ids[1:], reduction="none").double()
        yield int(row), len(values), float(values.mean()), float(values[511:].mean())


def make_bases(reference, model, directory, validation):
    method = reference.TTTLinear.ttt
    q_covariances = [torch.zeros(12, 64, 64, dtype=torch.float64) for _ in model.layers]
    k_covariances = [torch.zeros_like(c) for c in q_covariances]
    v_covariances = [torch.zeros_like(c) for c in q_covariances]

    def collect(layer, inputs, *args, **kwargs):
        q = inputs["XQ"][0].flatten(1, 2).double()
        k = inputs["XK"][0].flatten(1, 2).double()
        target = inputs["XV"][0].flatten(1, 2).double() - k
        q_covariances[layer.layer_idx] += q.transpose(1, 2) @ q
        k_covariances[layer.layer_idx] += k.transpose(1, 2) @ k
        v_covariances[layer.layer_idx] += target.transpose(1, 2) @ target
        return method(layer, inputs, *args, **kwargs)

    reference.TTTLinear.ttt = collect
    try:
        list(losses(model, validation))
    finally:
        reference.TTTLinear.ttt = method
    bases = torch.stack([torch.linalg.eigh(c)[1][:, :, -32:] for c in q_covariances]).float()
    np.save(directory / "key_bases.npy", bases.numpy())
    def unit_trace(matrices):
        cov = torch.stack(matrices)
        return 64 * cov / cov.diagonal(dim1=-2, dim2=-1).sum(dim=-1)[:, :, None, None]
    metric = 0.475 * (unit_trace(q_covariances) + unit_trace(k_covariances)) + 0.05 * torch.eye(64)
    np.save(directory / "prior_metrics.npy", metric.numpy())
    def coordinates(cov):
        vectors = torch.linalg.eigh(cov)[1].flip(-1)
        pivot = vectors.abs().argmax(dim=-2, keepdim=True)
        return vectors * vectors.gather(-2, pivot).sign()
    frames = torch.stack((coordinates(metric), coordinates(torch.stack(v_covariances))), dim=2)
    np.save(directory / "frames.npy", frames.float().numpy())


def verify(reference, model, validation):
    # Differentiate through a dense explicit pair product and LayerNorm.
    # The implementation under test uses FFT-based reads and FWM convolutions.
    with torch.inference_mode(False), torch.enable_grad():
        rng = torch.Generator().manual_seed(334)
        a, b = [(torch.randn(8, 127, generator=rng, dtype=torch.float64)/16).requires_grad_()
                for _ in range(2)]
        k, v = torch.randn(2, 64, generator=rng, dtype=torch.float64)
        index = torch.arange(64)[:, None] + torch.arange(64)[None, :]
        z = a[:, index] @ k
        p = (b[:, index] @ z[..., None]).sum(dim=0).flatten()
        target = v-k
        loss = 0.5 * (F.layer_norm(p, (64,), eps=1e-6) - target).square().sum()
        ga, gb = torch.autograd.grad(loss, (a, b))
        g = reference.ln_fused_l2_bwd(p, target, torch.ones_like(k), torch.zeros_like(k))
        torch.testing.assert_close(convolve(hankel_read(b, g), k), ga)
        torch.testing.assert_close(convolve(hankel_read(a, k), g), gb)
    method = reference.TTTLinear.ttt
    max_error = []

    def compare(layer, inputs, *args, **kwargs):
        output, state = method(layer, inputs, *args, **kwargs)
        direct = fast_layer(reference, layer, inputs, "dense_check", 1., None, None)
        torch.testing.assert_close(direct, output, atol=5e-5, rtol=1e-4)
        max_error.append(float((direct-output).abs().max()))
        return output, state

    reference.TTTLinear.ttt = compare
    try:
        model(input_ids=torch.from_numpy(validation["ids"][:1, :128]).long(), use_cache=False)
    finally:
        reference.TTTLinear.ttt = method
    print(f"Full-width update versus reference on all 12 layers, max field difference {max(max_error):.3g}")


def main(args):
    reference, model = load_model(args.directory)
    torch.set_num_threads(args.threads)
    validation = np.load(args.directory / "validation.npz")
    test = np.load(args.directory / "passages.npz")
    if args.mode == "verify":
        verify(reference, model, validation)
        make_bases(reference, model, args.directory, validation)
        return
    mode = args.mode
    bases = torch.from_numpy(np.load(args.directory / "key_bases.npy")) if mode.startswith("projected32") else None
    if mode == "hankel_aligned":
        bases = torch.from_numpy(np.load(args.directory / "frames.npy"))
    priors = None
    prior_files = {"hankel_only": "initial_factors.npy", "hankel_calibrated": "initial_calibrated.npy",
                   "hankel_aligned": "initial_aligned.npy"}
    if mode in prior_files:
        priors = torch.from_numpy(np.load(args.directory / prior_files[mode]))
    original = reference.TTTLinear.ttt
    if mode != "dense":
        def replaced(layer, inputs, *unused, **kwargs):
            if kwargs.get("cache_params") is not None or inputs["XQ"].shape[0] != 1:
                raise ValueError("This evaluation uses a single passage per forward and no cache.")
            return fast_layer(reference, layer, inputs, mode, step, bases, priors), None
        reference.TTTLinear.ttt = replaced
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / f"lm_{mode}.csv").open("w", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(("mode", "split", "step", "row", "predictions", "nll", "nll_after_511"))
        def score(split, data, step):
            start = time.monotonic()
            samples = list(losses(model, data))
            writer.writerows((mode, split, step, *s) for s in samples)
            file.flush()
            mean = np.mean([r[2] for r in samples])
            print(f"{mode} {split} step {step:g}, NLL {mean:.4f} ({time.monotonic()-start:.1f}s)", flush=True)
            return mean if np.isfinite(mean) else np.inf
        try:
            validation_scores = {}
            for step in RATES[mode]:
                validation_scores[step] = score("validation", validation, step)
            step = min(validation_scores, key=validation_scores.get)
            score("test", test, step)
            if mode in ("hankel_calibrated", "hankel_aligned") and step != 0.:
                step = 0.
                score("test_frozen", test, step)
        finally:
            reference.TTTLinear.ttt = original


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=(*MODES, "verify"))
    parser.add_argument("directory", type=Path)
    parser.add_argument("output", type=Path)
    parser.add_argument("--threads", type=int, default=3)
    with torch.inference_mode():
        main(parser.parse_args())
