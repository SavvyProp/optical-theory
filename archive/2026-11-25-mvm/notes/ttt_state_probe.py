"""Extract native TTT-Linear states and future queries for a Hankel memory probe.

Use the pinned files and commands in the repository research note. 'prepare'
uses tokenizers and pyarrow; 'extract' uses the local PyTorch model implementation.
Large public weights, passages, and intermediate tensors stay outside the repo.
"""

import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np


HASHES = {
    "config.json": "6d8080c8c86755d69fe74eb966f1c95951e0353fcf41774936cc9f90ed9f30b8",
    "model.safetensors": "80145a2d6fe137802d6142b870e67a9bf078a817641817abe3b930ac3b983775",
    "modeling_ttt.py": "7afecb505c070014247d37ff2519cf04942ebbfcdea6770218117ea3d1023357",
    "tokenizer.json": "c88bda6bdd84543eadebdf4bd2ec325ae43b71a3ff1fead6a766562c00b29bd8",
    "wiki-test.parquet": "5f1bea067869d04849c0f975a2b29c4ff47d867f484f5010ea5e861eab246d91",
    "wiki-validation.parquet": "204929b7ff9d6184953f867dedb860e40aa69c078fc1e54b3baaa8fb28511c4c",
}
PREFIXES = (64, 256, 1024)
NEXT_QUERIES = 64
PASSAGE_COUNT = 12


def checked(directory, name):
    path = directory / name
    with path.open("rb") as stream:
        digest = hashlib.sha256()
        for block in iter(lambda: stream.read(2**20), b""):
            digest.update(block)
    if digest.hexdigest() != HASHES[name]:
        raise ValueError(f"Unexpected SHA-256 for {path}. Check the dataset/checkpoint revision.")
    return path


def prepare(directory, validation=False):
    import pyarrow.parquet as parquet
    from tokenizers import Tokenizer

    name = "wiki-validation.parquet" if validation else "wiki-test.parquet"
    lines = parquet.read_table(checked(directory, name))["text"].to_pylist()
    tokenizer = Tokenizer.from_file(str(checked(directory, "tokenizer.json")))
    articles = []
    for row, line in enumerate(lines):
        line = line.strip()
        if line.startswith("= ") and line.endswith(" =") and not line.startswith("= ="):
            articles.append((row, []))
        if line and articles:
            articles[-1][1].append(line)
    eligible = []
    length = max(PREFIXES) + NEXT_QUERIES
    for row, paragraphs in articles:
        ids = tokenizer.encode("\n\n".join(paragraphs), add_special_tokens=True).ids
        if len(ids) >= length:
            eligible.append((row, ids[:length]))
    count = 4 if validation else PASSAGE_COUNT
    if len(eligible) < count:
        raise ValueError("Too few disjoint articles for the prespecified sample")
    selected = [eligible[i] for i in np.linspace(0, len(eligible) - 1, count, dtype=int)]
    output = "validation.npz" if validation else "passages.npz"
    np.savez(directory / output, rows=[x[0] for x in selected], ids=[x[1] for x in selected])
    print(f"{len(eligible)} eligible articles, {len(selected)} selected, {length} tokens each.")


def load_model(directory):
    import torch
    from safetensors.torch import load_file

    checked(directory, "modeling_ttt.py")
    sys.path.insert(0, str(directory))
    import modeling_ttt as reference

    torch.set_num_threads(4)
    config = reference.TTTConfig(**json.loads(checked(directory, "config.json").read_text()))
    model = reference.TTTModel(config).float().eval()
    tensors = load_file(str(checked(directory, "model.safetensors")))
    model.load_state_dict({k.removeprefix("model."): v for k, v in tensors.items()}, strict=True)
    del tensors
    # Use the reference implementation's CPU convolution.
    reference.causal_conv1d_fn = reference.causal_conv1d_update = None
    return reference, model


def extract(directory):
    import torch

    reference, model = load_model(directory)
    passages = np.load(directory / "passages.npz")
    records = {"row": [], "layer": [], "head": [], "prefix": [], "delta": [], "queries": []}
    priors = np.stack([block.seq_modeling_block.W1.detach().numpy().transpose(0, 2, 1)
                       for block in model.layers])
    current_row = None
    original = reference.TTTLinear.ttt

    def inspect(layer, inputs, mini_batch_size, last_mini_batch_params_dict=None, cache_params=None):
        if last_mini_batch_params_dict is not None or cache_params is not None:
            raise ValueError("The extractor expects one full divisible prefix and use_cache=False")
        query = inputs["XQ"][0].flatten(1, 2)
        outputs, previous, state = [], 0, None
        for end in (*PREFIXES, query.shape[1]):
            chunk = {k: v[:, :, previous // mini_batch_size:end // mini_batch_size] for k, v in inputs.items()}
            output, state = original(layer, chunk, mini_batch_size, state, None)
            outputs.append(output)
            if end in PREFIXES:
                matrices = state["W1_states"][0].transpose(1, 2).numpy()
                for head, matrix in enumerate(matrices):
                    records["row"].append(current_row)
                    records["layer"].append(layer.layer_idx)
                    records["head"].append(head)
                    records["prefix"].append(end)
                    records["delta"].append(matrix - priors[layer.layer_idx, head])
                    records["queries"].append(query[head, end:end + NEXT_QUERIES].numpy().copy())
            previous = end
        output = torch.cat(outputs, dim=1)
        if current_row == passages["rows"][0]:
            baseline, final = original(layer, inputs, mini_batch_size, None, None)
            torch.testing.assert_close(output, baseline, atol=1e-5, rtol=5e-5)
            for name in ("W1_states", "b1_states"):
                torch.testing.assert_close(state[name], final[name], atol=1e-5, rtol=5e-5)
        return output, state

    reference.TTTLinear.ttt = inspect
    try:
        with torch.inference_mode():
            for current_row, ids in zip(passages["rows"], passages["ids"]):
                model(input_ids=torch.from_numpy(ids)[None], use_cache=False)
                print(f"Extracted row {current_row}", flush=True)
    finally:
        reference.TTTLinear.ttt = original
    np.savez(directory / "states.npz", priors=priors, **records)
    print(f'{len(records["delta"])} head snapshots. Split and unsplit reference forwards agree.')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("stage", choices=("prepare", "prepare-validation", "extract"))
    parser.add_argument("directory", type=Path)
    args = parser.parse_args()
    if args.stage == "extract":
        extract(args.directory.resolve())
    else:
        prepare(args.directory.resolve(), validation=args.stage == "prepare-validation")
