"""Three-wave DFG stress test for one signed-real 128-wide, 66-factor Hankel MVM.

Run OPENBLAS_NUM_THREADS=1 python3 notes/hankel_product_dfg.py.
eta (1/W) is a hypothetical low-power slope P_idler/(P_pump * P_seed)
for one matched triplet. The weight comb is the pump near twice the data
frequency. Fields are in sqrt(W). Assume a flat, lossless, phase-matched guide
with common group velocity, fresh weights every visit, ideal output filters
and constant scalar loop gain. The propagation omits stochastic noise. A
separate balanced-factor budget estimates electrical energy and added noise.
"""
import argparse

import numpy as np

from hankel_displacement_energy import DC, MEMORY_COST, NF, PHOTON_PJ, SLOT_PS, SOA
from hankel_product_viability import initial_factors


def dfg_output(pump, data, eta, steps):
    # With no incident idler the phase of each field is fixed up to a sign.
    # This frame removes the common factor i in the generated idler.
    p0, s0 = np.broadcast_arrays(abs(pump), abs(data))
    fields = np.stack((p0, s0, np.zeros_like(s0)))
    dz = np.sqrt(eta) / steps

    def rhs(y):
        p, s, i = y
        return np.stack((-2*s*i, p*i, p*s))

    for _ in range(steps):
        a = rhs(fields)
        b = rhs(fields + dz*a/2)
        c = rhs(fields + dz*b/2)
        d = rhs(fields + dz*c)
        fields += dz*(a + 2*b + 2*c + d)/6
    p, s, i = fields
    # nu_p = nu_s + nu_i; approximate nu_s = nu_i for photon-flux factors only.
    np.testing.assert_allclose(p*p + 2*i*i, p0*p0, rtol=2e-5, atol=1e-8)
    np.testing.assert_allclose(s*s - i*i, s0*s0, rtol=2e-5, atol=1e-8)
    return i * np.exp(1j * (np.angle(pump) - np.angle(data)))


def propagate(factors, queries, samples, steps, eta, pump_w, packet_pj):
    n = queries.shape[1]
    idx = np.add.outer(np.arange(n), np.arange(n))
    # Constant scalar amplitude gain matches the ideal product normalization.
    norms = np.linalg.norm(factors, axis=1)
    gain = np.exp(np.log(norms).mean()) / np.sqrt(eta*pump_w)
    scale = np.sqrt(packet_pj/SLOT_PS)
    target = queries
    weak = queries * scale
    actual = weak.copy()
    for taps, norm in zip(factors, norms):
        pump = samples*np.fft.ifft(taps/norm, n=samples) * np.sqrt(pump_w)
        data = samples*np.fft.ifft(actual, n=samples, axis=-1)
        idler = dfg_output(pump, data, eta, steps)
        actual = np.fft.fft(idler, axis=-1)[:, :n] * (gain/samples)
        linear = np.sqrt(eta)*pump*(samples*np.fft.ifft(weak, n=samples, axis=-1)).conj()
        weak = np.fft.fft(linear, axis=-1)[:, :n] * (gain/samples)
        target = target @ taps[idx].T
    np.testing.assert_allclose(weak/scale, target, atol=2e-12)
    return actual/scale, target


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--eta", type=float, default=2., help="low-power conversion slope in 1/W")
    parser.add_argument("--pump-mw", type=float, default=100/3, help="total 775-nm weight power")
    parser.add_argument("--packet-pj", type=float, default=1.024)
    args = parser.parse_args()
    if not all(np.isfinite(v) and v > 0 for v in vars(args).values()):
        parser.error("powers, packet energy, and slope must be finite and positive")
    n, depth, split = 128, 66, 6
    factors = initial_factors(width=n, depth=depth)["perturbed"]
    queries = np.random.default_rng(2167).normal(size=(2*split, n))
    queries /= np.linalg.norm(queries, axis=1, keepdims=True)
    params = (args.eta, args.pump_mw/1000, args.packet_pj)
    actual, target = propagate(factors, queries, 2048, 32, *params)
    fine, _ = propagate(factors, queries[-1:], 4096, 64, *params)
    refinement = np.linalg.norm(actual[-1:]-fine) / np.linalg.norm(fine)
    assert refinement < 1e-5
    calibration = np.vdot(actual[:split], target[:split]) / np.vdot(actual[:split], actual[:split])
    q = args.eta*args.pump_mw/1000*n/(2*n-1)
    print(f"Balanced weak q {q:.4f}, last-probe refinement {refinement:.2g}")
    external_input_pj = 8.
    if q <= 1 and args.packet_pj <= external_input_pj:
        output_pj = external_input_pj * 10**0.1
        weights_pj = depth * params[1] * SLOT_PS * MEMORY_COST
        # These rails assume a balanced computation and the same net weight
        # delivery efficiency near 775 nm as on the FWM telecom weight rail.
        signal_pj = (depth * (1-q) * args.packet_pj + output_pj - args.packet_pj) / SOA
        wall_pj = (weights_pj + signal_pj) / DC
        noise_photons = depth * (1 + (NF/q - 1)/2)
        noise_rms = np.sqrt(noise_photons * n * PHOTON_PJ / args.packet_pj)
        print(f"Balanced loop gain {-10*np.log10(q):.2f} dB, {wall_pj/1000:.3f} nJ/read, "
              f"{1000*wall_pj/n**2:.2f} fJ/local MAC, {100*noise_rms:.2f}% added field RMS")
    for k in (1, calibration):
        err = np.linalg.norm(k*actual[split:]-target[split:]) / np.linalg.norm(target[split:])
        print(f"Scalar {k.real:.4f}, held-out deterministic field RMS {100*err:.2f}%")


if __name__ == "__main__":
    main()
