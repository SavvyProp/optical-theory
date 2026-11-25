"""Electrical sensitivity for one serial displacement-encoded Wq at unit signal gain.

Run OPENBLAS_NUM_THREADS=1 python3 notes/hankel_displacement_energy.py.
All reported powers are within a 20 ps active packet. References and the fixed
central weight tooth use synchronized sources. The variable side teeth are
replenished through the archived fiber-tap model. Costs exclude source
encoding, switch/bias/cooling loads, noisy-light amplifier loading, writeback
and finite-pulse distortion. RF is continuous across one serialized flight.
"""

import argparse
from dataclasses import dataclass

import numpy as np
from scipy.constants import c, h
from scipy.linalg import toeplitz
from scipy.optimize import minimize

from hankel_displacement_memory import encode


ETA = 0.468  # ideal one lit reference/weight pair at 50 mW each
NF = 10**0.5
SOA = 0.121
DC = 0.95
LASER = 0.40
PHOTON_PJ = h * c / 1550e-9 * 1e12
REFERENCE_PS, SLOT_PS = 24., 20.
TAP = 0.25
MEMORY_COST = 10**0.05 * (1 + (1-TAP) * (1-10**(-0.02)) / TAP) / SOA
MIN_VISIT_NS = 0.286  # guide flight alone; filters, switching and fiber add delay


@dataclass
class Profiles:
    side_norm: np.ndarray
    signal: np.ndarray
    amp_noise: np.ndarray
    fwm_noise: np.ndarray
    all_idlers: np.ndarray


def profiles(width, count=16, all_vacuum=False):
    rng = np.random.default_rng(7261 + width)
    eye = np.eye(width)
    shift = np.eye(width, k=-1)
    sites = np.arange(width)
    modes = 2*width
    all_rows = np.arange(1-modes, 2*modes-1)
    index = np.add.outer(all_rows, np.arange(modes))
    valid = (index >= 0) & (index < 2*modes-1)
    index = index.clip(0, 2*modes-2)

    cases = []
    for _ in range(count):
        w, r = np.linalg.qr(rng.normal(size=(width, width)))
        w *= np.sign(np.diag(r))
        taps = encode(w)
        transforms = [toeplitz(np.r_[v[-1], np.zeros(width-1)], v[::-1]) for v in taps]
        answer = np.zeros_like(w)
        query = eye.copy()
        signals, all_idlers = [], []
        couplings = []
        for v, transform in zip(taps, transforms):
            # Reflection of both mode indices on alternate visits preserves the
            # total idler energy. Calculate it on the entire intended family.
            pump = np.zeros(2*modes-1)
            pump[modes-1] = 1.
            pump[2*modes-2-sites] = v
            converter = pump[index] * valid
            idlers = converter @ np.vstack((answer, query))
            if all_vacuum:
                # The mixing matrix is Hankel. Exchanging input and output
                # indices gives the coupling of all vacuum inputs into the
                # selected outputs, now reversed into logical register order.
                couplings.append(converter.T[::-1])
            all_idlers.append(np.sum(idlers**2)/width)
            answer += transform @ query
            signals.append((np.sum(answer**2) + np.sum(query**2))/width)
            assert np.isclose(np.sum(idlers[modes-1:2*modes-1]**2)/width, signals[-1])
            query = shift @ query
        np.testing.assert_allclose(answer, w, atol=1e-12)

        # The suffix from a fresh per-mode noise field to the final answer is
        # [I,B]. An amplifier's noise enters before the query's zero-fill shift.
        # For vacuum fluctuations entering the conjugator also include the
        # current upper-triangular Toeplitz transform.
        suffix = np.zeros_like(w)
        ase, spontaneous = [], []
        for slot in reversed(range(width)):
            suffix = suffix @ shift
            ase.append(1 + np.sum(suffix**2)/width)
            if all_vacuum:
                vacuum_map = np.hstack((eye, suffix)) @ couplings[slot]
                spontaneous.append(np.sum(vacuum_map**2)/width)
            suffix += transforms[slot]
            if not all_vacuum:
                spontaneous.append(1 + np.sum(suffix**2)/width)
        cases.append([np.sum(taps**2, axis=1), signals, ase[::-1],
                      spontaneous[::-1], all_idlers])
    return Profiles(*np.asarray(cases).transpose(1, 0, 2))


def budget(prof, ref_mw, pilot_mw, loss_db, error, rf_mw):
    """Pooled isotropic noise target, real unitary targets, one polarization."""
    width = prof.signal.shape[1]
    transmission = 10**(-loss_db/10)
    g2 = ETA * ref_mw * pilot_mw / 50**2
    gain = 1 / (g2 * transmission)
    ase = transmission * (NF * gain - 1)/2
    photons = (prof.fwm_noise + ase*prof.amp_noise).sum(axis=1).mean()
    input_pj = width * PHOTON_PJ * photons/error**2

    # Amplifier precedes equal passive attenuation of both arms and query-edge
    # rejection. Deliver Wq at the launch energy; the last query is discarded.
    amp_pj = input_pj * (1/transmission - g2) * prof.signal.sum(axis=1).mean() / SOA
    side_mw = pilot_mw * prof.side_norm
    rails_pj = np.array([
        width * ref_mw * REFERENCE_PS / (1000*LASER),
        width * pilot_mw * REFERENCE_PS / (1000*LASER),
        width * side_mw.mean() * SLOT_PS/1000 * MEMORY_COST,
        amp_pj,
        width * rf_mw / 0.30 * MIN_VISIT_NS,
    ])
    all_idler_pj = g2 * input_pj * prof.all_idlers
    reference_ratio = all_idler_pj.max() / (ref_mw*SLOT_PS/1000)
    weight_ratio = np.max(all_idler_pj / ((pilot_mw+side_mw)*SLOT_PS/1000))
    return dict(rails_pj=rails_pj, fj=rails_pj.sum()/DC/width**2*1000,
                input_pj=input_pj, gain_db=10*np.log10(gain), photons=photons,
                reference_ratio=reference_ratio, weight_ratio=weight_ratio)


def optimized_pumps(prof, loss_db, error, rf_mw):
    # One fixed-gain program for all W in the ensemble. A cap on every slot
    # includes the central pilot and all variable side teeth.
    max_pilot = 100 / (1 + prof.side_norm.max())
    def objective(power):
        return budget(prof, *power, loss_db, error, rf_mw)['fj']
    result = minimize(objective, [50., 12.], bounds=[(20., 100.), (1., max_pilot)],
                      constraints=[dict(type='ineq', fun=lambda power: 1-ETA*np.prod(power)/50**2)],
                      method='SLSQP', tol=1e-8)
    if not result.success:
        raise RuntimeError(result.message)
    return result.x


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--error', type=float, default=0.10, help='RMS added field noise')
    parser.add_argument('--rf-mw', type=float, default=10., help='50 GHz RF power at shifter')
    parser.add_argument('--all-vacuum', action='store_true',
                        help='also couple vacuum bins outside the 2d-mode data aperture')
    args = parser.parse_args()
    for width in (32, 64):
        prof = profiles(width, all_vacuum=args.all_vacuum)
        print(f'width {width} mean/maximum side power in units of pilot '
              f'{prof.side_norm.mean():.3f}/{prof.side_norm.max():.3f}')
        for name, loss in [('50 mW peak weight', 0.), ('optimized', 0.),
                           ('optimized', 1.), ('optimized', 4.7)]:
            params = (50., 50/(1+prof.side_norm.max())) if name.startswith('50') else (
                optimized_pumps(prof, loss, args.error, args.rf_mw))
            row = budget(prof, *params, loss, args.error, args.rf_mw)
            r, p = params
            print(f"{name}, {loss:g} dB, R/pilot {r:.3f}/{p:.3f} mW, "
                  f"input {row['input_pj']:.3f} pJ, gain {row['gain_db']:.2f} dB, "
                  f"noise {row['photons']:.1f} ph/mode, {row['fj']:.1f} fJ/MAC")
            print('Pre-DC pJ [reference, pilot, weight loop, signal gain, RF]',
                  row['rails_pj'].round(2), 'max isotropic all-idler/input-pump energy ratios '
                  f"R {row['reference_ratio']:.3f}, weight {row['weight_ratio']:.3f}")


if __name__ == '__main__':
    main()
