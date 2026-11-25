"""Planning area and forward throughput for the 128-wide recirculating cell.

Run python3 notes/hankel_product_area.py. Physical banks are distinct and
can pipeline independent vectors. The fiber storage length and the generator
of the already encoded optical weight packets are outside the compute tiles.
"""
import argparse


WIDTH = 128
VISITS = 66
LANES = 8
GUIDE_PS = 1e12 * .0212 * 4.04 / 299792458
FWM_MM2 = .28
MZI_MM2 = 1.00
SOA_MM2 = .75
DRIVER_MM2 = 2.31
BANK_DISTRIBUTION_MM2 = 2.00
ROUTER_SITE_MM2 = 2.00


def area(lanes, banks, local_routing_mm2, *, drivers=True):
    lane = FWM_MM2 + 4 * MZI_MM2 + 2 * SOA_MM2 + local_routing_mm2
    if drivers:
        lane += 4 * DRIVER_MM2
    if lanes == 1:
        return lane
    bank = lanes * lane + BANK_DISTRIBUTION_MM2
    boundary = ROUTER_SITE_MM2 + lanes * SOA_MM2
    return banks * bank + (banks - 1) * boundary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--lap-ps", type=float, default=400.)
    parser.add_argument("--packets", type=int, default=1,
                        help="independent packets launched as one group per lane")
    parser.add_argument("--slot-ps", type=float, default=100.,
                        help="minimum pitch of independent packets inside a group")
    parser.add_argument("--local-routing-mm2", type=float, default=2.)
    parser.add_argument("--local-only", action="store_true",
                        help="show throughput and both footprint conventions for one 128 x 128 cell")
    parser.add_argument("--banks", nargs="+", type=int, default=[3, 9, 17])
    args = parser.parse_args()
    if args.lap_ps < GUIDE_PS or args.packets < 1 or args.slot_ps < 20:
        parser.error("allow for the guide flight, at least one packet, and the 20 ps illuminated window")
    if args.packets * args.slot_ps > args.lap_ps:
        parser.error("the packet group including its last slot must fit within a data lap")
    if min(args.banks) < 1 or args.local_routing_mm2 < 0:
        parser.error("specify positive bank counts and nonnegative routing area")

    # One group's period is 66 laps. Dividing operations by ps yields TOPS.
    bank_time_ps = VISITS * args.lap_ps
    rate_per_ps = args.packets / bank_time_ps
    print(f"{bank_time_ps/1000:.2f} ns/bank, {rate_per_ps*1e6:.2f} Mvectors/s "
          f"with {args.packets} independent packet(s)/lane")
    if args.local_only:
        tops = 2 * WIDTH**2 * rate_per_ps
        print(f"One completed {WIDTH} x {WIDTH} MVM: {2 * WIDTH**2:,} real ops; {tops:.3f} TOPS")
        for label, drivers in [("Photonic including gain", False), ("Including driver dies", True)]:
            footprint = area(1, 1, args.local_routing_mm2, drivers=drivers)
            print(f"{label}: {footprint:.2f} mm2, {tops/footprint:.4f} local TOPS/mm2")
        return
    print("lanes banks  photonic_mm2  with_drivers_mm2  local_TOPS  local_TOPS/mm2  "
          "dense_TOPS  dense_TOPS/mm2  serial_dense_TOPS/mm2")
    for lanes, banks in [(1, 1)] + [(LANES, b) for b in args.banks]:
        photonic = area(lanes, banks, args.local_routing_mm2, drivers=False)
        total = area(lanes, banks, args.local_routing_mm2)
        local_ops = 2 * WIDTH**2 * lanes * banks
        dense_ops = 2 * (WIDTH * lanes)**2
        local_tops = local_ops * rate_per_ps
        dense_tops = dense_ops * rate_per_ps
        serial_density = dense_ops / (banks * bank_time_ps * total)
        print(f"{lanes:5d} {banks:5d} {photonic:13.2f} {total:17.2f} "
              f"{local_tops:11.2f} {local_tops/total:15.4f} {dense_tops:11.2f} "
              f"{dense_tops/total:15.4f} {serial_density:22.4f}")


if __name__ == "__main__":
    main()
