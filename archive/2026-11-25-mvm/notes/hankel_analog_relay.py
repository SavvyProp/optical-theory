"""Signed forward read through an analog receiver in place of the final amplifier.

Run OPENBLAS_NUM_THREADS=1 python3 -B notes/hankel_analog_relay.py.
The model uses the original dense matrices and G1. Efficiencies, receiver
charge noise, circuit power and optical phases are design assumptions.
"""
import argparse

import numpy as np
from scipy.constants import c, h
from scipy.optimize import brentq
from scipy.special import erf, roots_hermitenorm

from hankel_amplifier_sweep import N, TARGET
from hankel_fwm_stress import optical_program
from hankel_port_recovery import forward_reads
from hankel_power_budget import DC_EFFICIENCY, budget
from hankel_single_read_energy import waveguide_field_coefficient, workload


READ_NS = 3.493
INPUT_PJ = 4
OUTPUT_PJ = INPUT_PJ * TARGET
PHOTON_J = h * c / 1.55e-6
QE = 0.8
LO_FJ = 10  # At each detector pair during the optical integration window.
ENC_ELECTRONS = 150  # Total rms equivalent differential charge, including reset.


def receiver_window(samples, bandwidth_ghz, window_ps):
    """Overlap with a rectangular coherent reference after a Gaussian AWG.

    The signal gate is 20 ps. Normalize the reference to unit energy over the
    longer receive window. Convolution with the field impulse response reduces
    to an erf difference, for zero residual group delay and spectral phase.
    """
    t_ps = 20 * (np.arange(samples) / samples - 0.5)
    rate = np.pi * bandwidth_ghz * 0.001 / np.sqrt(2 * np.log(2))
    return np.sqrt(20 / window_ps) * (
        erf(rate * (t_ps + window_ps / 2)) - erf(rate * (t_ps - window_ps / 2))
    ) / 2


def run(pump, integration_ps, fine=False):
    programs, chirp = workload()
    scale = max(optical_program(weights, N)[2] for _, weights, _ in programs)
    coupling = waveguide_field_coefficient(4, .2) / waveguide_field_coefficient(2.12, .8)
    parts, _, _, _ = budget(READ_NS, gated_booster=True, pulsed_reference=True,
                            extra_booster_pj=50)
    balance = np.sqrt(parts["weight loop"] / parts["reference"])
    other_pj = (READ_NS * (parts["reference"] * pump * balance
                + parts["weight loop"] * pump / balance + parts["raw idler gain"]
                + parts["output gate"] + parts["clock, holding and trims"]) + 10) / DC_EFFICIENCY
    booster_pj = READ_NS * parts["output gain"] / DC_EFFICIENCY
    samples, steps = (1024, 64) if fine else (512, 32)
    window = receiver_window(samples, 27.2, integration_ps)
    coherent_capture = window.mean()**2
    excess_noise_factor = np.mean(window**2) / coherent_capture
    target_rows, sorted_rows, original_rows, excess_rows = [], [], [], []
    for target, wave, decoder, gain_db, _, terms in forward_reads(
            programs, chirp, scale, 2 * 10**(-.5/10), pump, samples=samples, steps=steps,
            coupling=coupling, reference_balance=balance, input_pj=INPUT_PJ):
        gain = 10**(gain_db / 10)
        photons = OUTPUT_PJ * 1e-12 / (gain * PHOTON_J)  # Full unit-norm packet
        target_rows.append(target)
        original_rows.append((np.fft.fft(wave)[:N] / decoder).real)
        sorted_rows.append((np.fft.fft(wave * window)[:N] / (decoder * window.mean())).real)
        # Normal-ordered excess photons per tooth before G2. Only one quadrature is read.
        excess_rows.append(terms / (2 * gain * photons))
    target = np.array(target_rows)
    measured = np.array(sorted_rows)
    excess = np.array(excess_rows)
    target_power = np.mean(target**2)
    base_sq = np.mean((np.array(original_rows) - target)**2) + excess.sum(axis=1).mean()
    print(f"Relative pump setting {pump}, integration {integration_ps:g} ps, "
          f"guide pumps {50*pump*balance:.2f} / {50*pump/balance:.2f} mW")
    print(f"G2 energy {booster_pj:.2f} pJ, other loads {other_pj:.2f} pJ, "
          f"original signed RMS {100*np.sqrt(base_sq/target_power):.2f}%")
    print(f"Pre-G2 photons per unit packet {photons:.0f}; filter mode capture "
          f"{coherent_capture:.3f} before AWG insertion and detector loss")
    nodes, weights = roots_hermitenorm(128 if fine else 64)
    weights /= weights.sum()
    edge_distance = np.abs(np.linspace(-1, 1, N))
    for label, loss_db in (("low", np.full(N, 3.5)), ("illustrative center-to-edge",
                            3.5 + 2.9 * edge_distance), ("high", np.full(N, 6.4))):
        eta = coherent_capture * 10**(-loss_db/10) * QE
        lo_electrons = QE * LO_FJ * 1e-15 / PHOTON_J
        # Strong-LO balanced readout. Vacuum contributes once. A white excess noise
        # approximation accounts for unequal weights within the original signal gate.
        noise_variance = (excess[:, :2].sum(axis=1)[:, None] * excess_noise_factor
                          + (1 + ENC_ELECTRONS**2 / lo_electrons) / (4 * eta * photons))
        samples_at_driver = measured[..., None] + np.sqrt(noise_variance)[..., None] * nodes
        def rms(output):
            mse = np.mean(((output - target[..., None])**2) @ weights)
            return 100 * np.sqrt(mse / target_power)
        print(f"AWG loss {label}, net efficiency {eta.min():.3f}–{eta.max():.3f}, signed receiver "
              f"RMS {rms(samples_at_driver):.2f}% including electronics")
        for headroom in (2.5, 2.8, 3.0, 3.5):
            limit = headroom / np.sqrt(N)
            linear = np.clip(samples_at_driver, -limit, limit)
            # Fixed small-signal drive calibration for a standard normal design
            # distribution. A voltage clamp holds the MZI at its first extremum.
            angle = brentq(lambda g: headroom*g*np.exp(-g*g/2) - 1, 0, 1)
            phase = np.clip(angle*np.sqrt(N)*samples_at_driver, -np.pi/2, np.pi/2)
            print(f"  ±{headroom:g} RMS: linearized/clipped {rms(linear):.2f}%, "
                  f"ordinary sine field transfer {rms(limit*np.sin(phase)):.2f}%")

    print("Headroom  TX efficiency  lane mW  relay pJ  whole-cell fJ/MAC")
    lo_pj = N * LO_FJ * .001 / .10  # 10% wall-to-reference efficiency
    for headroom, tx_eff, lane_mw in ((2.5,.20,.35), (2.8,.20,.35), (3,.20,.35), (3,.20,.70),
                                      (3,.10,.35), (2.5,.10,.35), (2.5,.20,.70)):
        carrier_pj = OUTPUT_PJ * headroom**2 / tx_eff
        relay_pj = carrier_pj + lo_pj + N * lane_mw * READ_NS
        print(f"{headroom:6.1f} {tx_eff:13.2f} {lane_mw:8.2f} {relay_pj:9.1f} "
              f"{1000*(other_pj+relay_pj)/N**2:16.1f}")
    print(f"Peak on workload: {np.max(np.abs(target))*np.sqrt(N):.3f} × nominal rms")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pump", type=float, default=.29, help="existing common pump scale")
    parser.add_argument("--integration-ps", type=float, default=40, help="receiver LO window")
    parser.add_argument("--fine", action="store_true", help="double waveform, solver and noise resolution")
    args = parser.parse_args()
    if args.pump <= 0 or args.integration_ps <= 0:
        parser.error("Pump scale and integration time must be positive.")
    run(args.pump, args.integration_ps, args.fine)
