"""Stress-test one budgeted read through a pump-depleted displacement loop.

The pulse is one periodic 20 ps window on a 50 GHz grid. This uses the
repository's four-field solver and an ideal query shifter, noiseless scalar
gain, exact carrier selection and flat phase matching. It scores only the
selected answer teeth. It does not simulate the weight writer or ASE.
"""

import argparse

import numpy as np

from hankel_displacement_memory import encode
from hankel_fwm_stress import ETA, PUMP_W, phase_matched_output


def propagate(matrix, queries, reference_mw, pilot_mw, input_pj, samples, steps):
    d = len(matrix)
    modes = 2 * d
    center = modes - 1
    offsets = center - np.arange(d)
    reference = np.sqrt(reference_mw / 1000)
    pilot = np.sqrt(pilot_mw / 1000)
    gain = PUMP_W / (np.sqrt(ETA) * reference * pilot)
    input_field = np.sqrt(input_pj / 20)
    initial = np.zeros((len(queries), modes), complex)
    initial[:, d:] = input_field * queries
    weak = initial.copy()
    actual = initial.copy()
    for lap, side in enumerate(encode(matrix)):
        pumps = np.zeros(2 * modes - 1, complex)
        pumps[center] = pilot
        pump_indices = center + offsets if lap % 2 == 0 else center - offsets
        pumps[pump_indices] = pilot * (side.conj() if lap % 2 == 0 else side)
        pump_wave = samples * np.fft.ifft(pumps, n=samples)
        data = samples * np.fft.ifft(actual, n=samples)
        weak_data = samples * np.fft.ifft(weak, n=samples)
        idler, invariants = phase_matched_output(
            np.broadcast_to(pump_wave, data.shape), data, reference, steps)
        if invariants > 1e-5:
            raise ArithmeticError(f"Refine integration step, invariant error {invariants:g}")
        weak_idler = pump_wave * weak_data.conj() * (np.sqrt(ETA) / PUMP_W) * reference
        actual = np.fft.fft(idler)[:, :modes] * (gain / samples)
        weak = np.fft.fft(weak_idler)[:, :modes] * (gain / samples)
        for field in (actual, weak):
            if lap % 2 == 0:
                field[:, :d-1] = field[:, 1:d]
                field[:, d-1] = 0
            else:
                field[:, d+1:] = field[:, d:-1]
                field[:, d] = 0
    assert d % 2 == 0
    target = queries @ matrix.T
    np.testing.assert_allclose(weak[:, :d] / input_field, target, atol=1e-11)
    return actual[:, :d] / input_field, target


def report(d, reference, pilot, energy, samples, steps, seed):
    rng = np.random.default_rng(seed)
    matrix, triangular = np.linalg.qr(rng.normal(size=(d, d)))
    matrix *= np.sign(np.diag(triangular))
    inputs = rng.normal(size=(12, d))
    inputs /= np.linalg.norm(inputs, axis=1, keepdims=True)
    output, target = propagate(matrix, inputs, reference, pilot, energy, samples, steps)
    scalar = np.vdot(output[:6], target[:6]) / np.vdot(output[:6], output[:6])
    for label, scale in (("programmed gain", 1.), ("held-out calibrated gain", scalar)):
        error = np.linalg.norm(scale * output[6:] - target[6:]) / np.linalg.norm(target[6:])
        print(f"d={d}, seed={seed}, {label}: {error:.3%} deterministic RMS", flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--fine", action="store_true", help="double waveform and integration resolution")
    parser.add_argument("--matrices", type=int, default=3, help="number of held-out dense matrices per width")
    parser.add_argument("--case", type=float, nargs=4, metavar=("D", "REF_MW", "PILOT_MW", "INPUT_PJ"))
    args = parser.parse_args()
    cases = [args.case] if args.case else [(32, 71.08, 9.22, .3957), (64, 100, 15.64, .7113)]
    for width, reference, pilot, energy in cases:
        for repeat in range(args.matrices):
            report(int(width), reference, pilot, energy, 1024 if args.fine else 512,
                   48 if args.fine else 24, 1874 + int(width) + repeat)
