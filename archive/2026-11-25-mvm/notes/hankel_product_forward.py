"""Forward stress test for a dense matrix encoded as a product of Hankel factors.

Run OPENBLAS_NUM_THREADS=1 python3 notes/hankel_product_forward.py.
For the 128-tooth case add --width 128 --depth 66 --probes-per-split 6.
Four fields, equal group velocity, ideal filters and an ideal scalar loop
amplifier. ASE, SPM/XPM and physical routing loss are omitted.
"""
import argparse

import numpy as np

from hankel_fwm_stress import ETA, PUMP_W, phase_matched_output
from hankel_product_viability import initial_factors


def propagate(factors, vectors, samples, steps, packet_pj, reference_w=PUMP_W):
    # The hardware conjugates the input field each lap. After an even number
    # of real factors the logical output is real for the signed inputs here.
    input_field = np.sqrt(packet_pj / 20)  # pJ/ps = W
    if not np.isfinite(reference_w) or reference_w <= 0:
        raise ValueError("Reference power must be positive and finite.")
    weight_w = PUMP_W**2 / reference_w  # retain the calibrated weak conversion
    reference_field = np.sqrt(reference_w)
    width = vectors.shape[1]
    indices = np.add.outer(np.arange(width), np.arange(width))
    if factors.shape[1] != 2 * width - 1 or samples < 2 * width - 1:
        raise ValueError("Each factor needs 2 × width − 1 taps and enough Fourier samples for all taps.")
    target = vectors.copy()
    weak = vectors.astype(complex) * input_field
    actual = weak.copy()
    pump_norms = np.linalg.norm(factors, axis=1)
    # One constant amplifier gain gives unity end-to-end scale in the
    # weak linear model, including small factor-to-factor norm differences.
    gain = np.exp(np.log(pump_norms).mean()) / np.sqrt(ETA)
    for factor, norm in zip(factors, pump_norms):
        weight = np.fft.ifft(factor / norm, n=samples) * samples * np.sqrt(weight_w)
        data = np.fft.ifft(actual, n=samples, axis=-1) * samples
        ideal_data = np.fft.ifft(weak, n=samples, axis=-1) * samples
        idler, invariant = phase_matched_output(np.broadcast_to(weight, data.shape),
                                                data, reference_field, steps)
        assert invariant < 1e-5
        weak_idler = weight * ideal_data.conj() * reference_field * (np.sqrt(ETA) / PUMP_W)
        weak = np.fft.fft(weak_idler, axis=-1)[:, :width] * (gain / samples)
        actual = np.fft.fft(idler, axis=-1)[:, :width] * (gain / samples)
        target = target @ factor[indices].T
    np.testing.assert_allclose(weak / input_field, target, atol=2e-12)
    return actual / input_field, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=64)
    parser.add_argument("--depth", type=int, help="Number of factors; defaults to width // 2 + 2.")
    parser.add_argument("--packet-pj", type=float, default=.512)
    parser.add_argument("--reference-mw", type=float, default=1000 * PUMP_W,
                        help="In-guide reference pump power. Total weight-comb power changes inversely"
                             " to keep the product of the two pump powers fixed.")
    parser.add_argument("--probes-per-split", type=int, default=10)
    args = parser.parse_args()
    depth = args.width // 2 + 2 if args.depth is None else args.depth
    split = args.probes_per_split
    if min(args.width, depth, split) < 1 or not np.isfinite(args.packet_pj) or args.packet_pj <= 0:
        parser.error("Width, depth, packet energy and number of probes must be positive.")
    if not np.isfinite(args.reference_mw) or args.reference_mw <= 0:
        parser.error("Reference power must be positive and finite.")
    reference_w = args.reference_mw / 1000
    print(f"Reference {args.reference_mw:.2f} mW, total weight comb {1000 * PUMP_W**2 / reference_w:.2f} mW")
    factors = initial_factors(width=args.width, depth=depth)['perturbed']
    vectors = np.random.default_rng(2167).normal(size=(2 * split, args.width))
    vectors /= np.linalg.norm(vectors, axis=1, keepdims=True)
    samples = max(1024, 16 * args.width)
    actual, target = propagate(factors, vectors, samples, 32, args.packet_pj, reference_w)
    refined, _ = propagate(factors, vectors[-1:], 2 * samples, 64, args.packet_pj, reference_w)
    convergence = np.linalg.norm(refined - actual[-1:]) / np.linalg.norm(refined)
    print(f'{args.width} teeth, {depth} passes, {args.packet_pj:g} pJ, '
          f'{split} calibration / {split} test probes; refined last-probe relative change {convergence:.3g}')
    np.testing.assert_allclose(actual[-1:], refined, rtol=3e-5, atol=1e-7)
    # Freeze one fitted complex scalar before scoring independent vectors.
    correction = np.vdot(actual[:split], target[:split]) / np.vdot(actual[:split], actual[:split])
    for scale in (1, correction):
        rms = np.linalg.norm(scale * actual[split:] - target[split:]) / np.linalg.norm(target[split:])
        print(f'Output scale {scale.real:.4f}, held-out deterministic field RMS {100*rms:.2f}%')


if __name__ == '__main__':
    main()
