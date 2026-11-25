"""Forward electrical energy for resident FWM Hankel products and optical shuffles.

Run python3 notes/hankel_product_energy.py. Energies are pJ per input vector.
One bank contains parallel cells. Between banks the router carries the circulating
signal level. A following gain section replaces the lost light. The final bank
alone boosts the output to the external interface level.
"""
import argparse
import math

from hankel_displacement_energy import (DC, ETA, LASER, MEMORY_COST, NF, PHOTON_PJ,
                                       REFERENCE_PS, SLOT_PS, SOA)
from hankel_fwm_stress import PUMP_W


def budget(width, visits, banks, lanes, circulating_pj, output_pj, router_loss_db,
           *, reference_w=PUMP_W, amplifier_efficiency=SOA, weight_delivery_efficiency=None):
    q = ETA * width / (2 * width - 1)
    transmission = 10 ** (-router_loss_db / 10)
    # Keep P_reference * P_weights fixed, so the modeled weak conversion is unchanged.
    weights_w = PUMP_W**2 / reference_w
    reference = visits * reference_w * REFERENCE_PS / LASER
    # MEMORY_COST includes the baseline SOA efficiency and the tap and delivery losses.
    if weight_delivery_efficiency is None:
        weight_delivery_efficiency = amplifier_efficiency / (MEMORY_COST * SOA)
    weights = visits * weights_w * SLOT_PS / weight_delivery_efficiency
    signal = visits * (1 - q) * circulating_pj / amplifier_efficiency
    # Bring all packets back to the circulating level each visit, then charge
    # the extra extraction gain only once. This equals a larger last-pass gain.
    extraction = (output_pj - circulating_pj) / amplifier_efficiency
    shuffle = (1 - transmission) * circulating_pj / amplifier_efficiency
    wall_pj = lanes * (banks * (reference + weights + signal)
                       + (banks - 1) * shuffle + extraction) / DC
    macs = banks * lanes * width**2

    # Balanced, unity-gain factors and independent spontaneous light in the
    # retained modes only. This estimate stops at the final circulating packet.
    per_visit_noise = 1 + (NF / q - 1) / 2
    per_shuffle_noise = (NF / transmission - 1) / 2 if router_loss_db else 0
    noise_photons = banks * visits * per_visit_noise + (banks - 1) * per_shuffle_noise
    field_rms = math.sqrt(noise_photons * width * PHOTON_PJ / circulating_pj)
    return wall_pj, 1000 * wall_pj / macs, field_rms, (reference, weights, signal, extraction, shuffle)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=128)
    parser.add_argument("--visits", type=int, default=66)
    parser.add_argument("--lanes", type=int, default=8)
    parser.add_argument("--banks", type=int, nargs="+", default=[1, 2, 3, 4])
    parser.add_argument("--circulating-pj", type=float, default=1.024)
    parser.add_argument("--input-pj", type=float, default=8.,
                        help="already encoded external energy per lane, before attenuation")
    parser.add_argument("--router-loss-db", type=float, default=3.)
    parser.add_argument("--reference-mw", type=float, default=1000 * PUMP_W,
                        help="reference power at the guide; the total weight power changes inversely")
    parser.add_argument("--amplifier-efficiency", type=float, default=SOA,
                        help="electrical-to-added-light fraction for weight and signal amplifiers")
    parser.add_argument("--weight-delivery-efficiency", type=float,
                        help="override the weight loop with net rail-to-programmed-weight efficiency, including shaping")
    args = parser.parse_args()
    if min(args.width, args.visits, args.lanes, *args.banks) < 1:
        parser.error("width, visits, banks and lanes must be positive")
    if not (0 < args.circulating_pj <= args.input_pj) or args.router_loss_db < 0:
        parser.error("use a positive circulating energy up to the input energy, and nonnegative loss")
    if not math.isfinite(args.reference_mw) or args.reference_mw <= 0 or not 0 < args.amplifier_efficiency <= 1:
        parser.error("use positive finite reference power and amplifier efficiency in (0, 1]")
    if args.weight_delivery_efficiency is not None and not 0 < args.weight_delivery_efficiency <= 1:
        parser.error("weight delivery efficiency must be in (0, 1]")
    output_pj = args.input_pj * 10**0.1  # +1 dB external energy gain

    def read(banks, lanes):
        return budget(args.width, args.visits, banks, lanes, args.circulating_pj,
                      output_pj, args.router_loss_db, reference_w=args.reference_mw / 1000,
                      amplifier_efficiency=args.amplifier_efficiency,
                      weight_delivery_efficiency=args.weight_delivery_efficiency)

    weight_mw = (1000 * PUMP_W)**2 / args.reference_mw
    print(f"In-guide reference {args.reference_mw:.2f} mW, total weights {weight_mw:.2f} mW, "
          f"amplifier efficiency {args.amplifier_efficiency:.1%}")
    if args.weight_delivery_efficiency is not None:
        print(f"Alternative weight delivery: {args.weight_delivery_efficiency:.1%} rail to programmed comb")
    cell_pj, cell_fj, rms, rails = read(1, 1)
    reference, weights, signal, extraction, shuffle = rails
    print(f"One cell: {cell_pj:.2f} pJ, {cell_fj:.2f} fJ/local MAC, "
          f"{100*rms:.2f}% added field RMS before extraction")
    print(f"One-lane device rails (pJ): reference {reference:.2f}, weights {weights:.2f}, "
          f"signal {signal:.2f}, extraction increment {extraction:.2f}, per shuffle {shuffle:.2f}")
    print("banks  shuffles  nJ/vector  fJ/local-MAC  fJ/dense-equivalent-MAC  added-field-RMS")
    for banks in args.banks:
        energy_pj, fj, rms, _ = read(banks, args.lanes)
        dense_fj = 1000 * energy_pj / (args.lanes * args.width)**2
        print(f"{banks:5d} {banks-1:9d} {energy_pj/1000:10.3f} {fj:13.2f} "
              f"{dense_fj:24.2f} {100*rms:15.2f}%")


if __name__ == "__main__":
    main()
