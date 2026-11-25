"""Electrical scenarios for one 64 x 64 forward read. Run with Python 3.

Powers are in mW, times in ns, and pulse energies in pJ (mW ns = pJ).
Device precedents and assumptions are in scattering_multipliers.md.
"""

N = 64
OPS = 2 * N * N
FRAME_NS = 2.88
DENSE_BANK_SCALE = 1.1576
REFERENCE_MW = 50
SLOT_WIDTH_NS = 0.020
REFERENCE_EFFICIENCY = 0.50 * 0.80  # Source and delivery to the guide
DC_EFFICIENCY = 0.95


def loop_power(scale):
    """Maintain one recirculating frame at its design bias at either read cadence."""
    tap_fraction = 0.25
    loop_transmission = 10 ** (-0.2 / 10)  # Other round-trip losses
    delivery = 10 ** (-0.5 / 10)  # From tapped port to nonlinear guide
    added_light_efficiency = 0.121
    # Unitary 64 x 64 W has unit mean squared slot norm before bank scaling.
    at_guide_pj = N * SLOT_WIDTH_NS * 50 / scale**2
    extracted_mw = at_guide_pj / (FRAME_NS * delivery)
    loss_factor = 1 + (1 - tap_fraction) * (1 - loop_transmission) / tap_fraction
    return extracted_mw * loss_factor / added_light_efficiency


def budget(interval_ns, *, gated_booster, pulsed_reference=False,
           raw_gain_mw=60, bank_scale=DENSE_BANK_SCALE, extra_booster_pj=0):
    reference_mw = REFERENCE_MW / REFERENCE_EFFICIENCY
    if pulsed_reference:
        # Stretch design: 24 ps equivalent source energy per 20 ps window.
        reference_mw *= N * 0.024 / interval_ns
    booster_mw = 150 * 1.5  # mA times V, continuous bias
    if gated_booster:
        # 1 ns of charge-equivalent drive including prebias/overdrive.
        booster_mw *= 1.0 / (0.80 * interval_ns)
    booster_mw += extra_booster_pj / interval_ns
    gate_pj = 1000 * 2.3**2 / 50 * SLOT_WIDTH_NS / 0.30
    components = {
        "reference": reference_mw, "weight loop": loop_power(bank_scale),
        "raw idler gain": raw_gain_mw, "output gain": booster_mw,
        "output gate": gate_pj / interval_ns, "clock, holding and trims": 5.0,
    }
    power_mw = sum(components.values()) / DC_EFFICIENCY
    energy_pj = power_mw * interval_ns
    return components, power_mw, energy_pj, OPS / energy_pj


def main():
    print("Component electrical allocations (mW), packet-biased G2, 2.88 ns")
    for name, value in budget(FRAME_NS, gated_booster=True)[0].items():
        print(f"{name:26} {value:7.2f}")
    print("Scenario                          interval ns     mW    nJ/MVM   TOPS/W")
    scenarios = [
        ("CW reference + CW output gain", dict(gated_booster=False)),
        ("packet-biased output gain", dict(gated_booster=True)),
        ("brighter output, 50 pJ reserve", dict(gated_booster=True, extra_booster_pj=50)),
        ("higher-current raw gain", dict(gated_booster=True, raw_gain_mw=450)),
        ("full nominal weight loading", dict(gated_booster=True, bank_scale=1)),
        ("efficient pulsed reference", dict(gated_booster=True, pulsed_reference=True)),
    ]
    for title, settings in scenarios:
        for interval_ns in (3.16, FRAME_NS):
            _, power, energy, efficiency = budget(interval_ns, **settings)
            print(f"{title:33} {interval_ns:7.2f} {power:9.2f} "
                  f"{energy / 1000:9.3f} {efficiency:8.3f}")
    # Vary one component at a time at the serial result interval.
    interval_ns = 3.16
    components, _, energy, _ = budget(interval_ns, gated_booster=True)
    headroom_pj = OPS / (2 * 2.48) - energy
    g1_max_mw = components["raw idler gain"] + DC_EFFICIENCY * headroom_pj / interval_ns
    g2_max_pj = components["output gain"] * interval_ns + DC_EFFICIENCY * headroom_pj
    print(f"Twice IBM four-phase efficiency, other assumptions fixed: "
          f"G1 up to {g1_max_mw:.1f} mW or G2 up to {g2_max_pj:.1f} pJ/read.")


if __name__ == "__main__":
    main()
