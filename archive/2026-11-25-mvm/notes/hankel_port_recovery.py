"""Half-frame selector in a single 64 x 64 FWM cell, one input per read.

OPENBLAS_NUM_THREADS=1 python3 -B notes/hankel_port_recovery.py
Add --accuracy to run the single-read electrical energy/field-error tradeoff.
Add --ten-ps to check a faster frame and estimate forward field error.
At the guide end, the A/B bands use independent selector schedules. Fixed
extra optical loss and an example DC drive allocation are sensitivity inputs.
"""
import argparse

import numpy as np

from hankel_amplifier_sweep import Read
from hankel_comb_codes import (DATA_MODES, N, SAMPLES, SLOTS, WEIGHT_MODES, code,
                               masked_weights, mesh, passive_taps, two_copies)
from hankel_fwm_stress import (ETA, INPUT_LOSS_DB, OUTPUT_LOSS_DB, PACKET_POWER_W,
                              PUMP_W, field_gain, optical_program, read)
from hankel_power_budget import DC_EFFICIENCY, budget
from hankel_single_read_energy import workload, waveguide_field_coefficient
from hankel_sparse_read import sparse_workload


def terminal_rails():
    upper, lower = np.ones(1), np.zeros(1)
    for stage in range(6):
        short = (upper + lower) / np.sqrt(2)
        long = (upper - lower) / np.sqrt(2)
        upper = np.pad(short, (0, 2**stage))
        lower = np.pad(long, (2**stage, 0))
    np.testing.assert_equal(upper[N//2:], 0)
    np.testing.assert_equal(lower[:N//2], 0)
    np.testing.assert_allclose((upper + lower) / np.sqrt(2), passive_taps())
    return upper, lower


def check_timing(guard_ps, *, period_ps=20., slot_ps=45., reset_ps=None,
                 consecutive=False):
    upper, lower = terminal_rails()
    def samples(ps):
        index = round(ps * SAMPLES / period_ps)
        np.testing.assert_allclose(index * period_ps / SAMPLES, ps, atol=1e-10)
        return index

    pitch = samples(slot_ps)
    delta = SAMPLES // N
    period = np.arange(SAMPLES)
    guard = samples(guard_ps)
    reset = guard if reset_ps is None else samples(reset_ps)
    frame = N*pitch + guard + reset
    starts = SLOTS * pitch + (SLOTS >= N//2)*guard
    windows = starts[:, None] + period
    q = N - 1 - SLOTS
    # At half frame, all desired B-adjoint two-copy envelopes and windows of
    # continuous B-forward G1 noise are disjoint between the two rail settings.
    left = np.max((starts - q * delta + 2*SAMPLES)[:N//2])
    right = np.min((starts - q * delta)[N//2:])
    assert left < right
    edge = (left + right) // 2

    def through_rail(packet, stride, use_lower):
        impulse = lower if use_lower else upper
        field = mesh(packet, stride, impulse)
        return np.pad(field, (guard,0) if use_lower else (0,guard))

    def extend(packet, stride, advance):
        u, l = (through_rail(packet, stride, lo) for lo in (False, True))
        timeline = np.arange(len(u)) - advance
        return np.where(timeline < edge, u, l)[windows + advance]

    def collect(raw, stride):
        train = np.zeros(windows[-1, -1] + 1, complex)
        train[windows] = raw
        early = np.arange(len(train)) < edge
        return through_rail(early*train, stride, True) + through_rail((~early)*train, stride, False)

    rng = np.random.default_rng(261123)
    def random_complex(shape):
        return rng.standard_normal(shape) + 1j*rng.standard_normal(shape)

    x, e, query = random_complex((3, N))
    matrix = random_complex((N, N))
    masks = code(-q)
    def encode(w):
        return masked_weights(w.T, masks.conj()) @ WEIGHT_MODES
    taps = upper + lower
    signs = np.sign(taps * taps[::-1])[:, None]
    gain = 1 / (2*np.sqrt(N))
    def copies_a(x):
        return extend(x.conj() @ DATA_MODES, pitch, 0)
    def copies_b(e):
        return extend(two_copies(e @ DATA_MODES), pitch+delta, (N-1)*delta)
    def forward_wave(stored, x):
        return two_copies(collect(stored * copies_a(x).conj(), pitch+delta))
    def adjoint_wave(stored, e):
        return collect(stored * copies_b(e).conj(), pitch)
    adjoint_gate = (N-1)*pitch + guard + period
    forward_gate = adjoint_gate + SAMPLES
    def fwd(stored, x):
        return forward_wave(stored, x)[forward_gate] / gain
    def adj(stored, e):
        return adjoint_wave(stored, e)[adjoint_gate] / gain
    stored = signs * encode(matrix)
    increment = signs * copies_a(x) * copies_b(e) / gain
    updated = matrix + np.outer(e, x.conj())
    tests = [(fwd(stored,x), (matrix @ x) @ DATA_MODES),
             (adj(stored,e), (matrix.conj().T @ e).conj() @ DATA_MODES),
             (increment, signs * encode(np.outer(e, x.conj()))),
             (fwd(stored+increment,query), (updated @ query) @ DATA_MODES),
             (adj(stored+increment,e), (updated.conj().T @ e).conj() @ DATA_MODES)]
    errors = [np.linalg.norm(actual-target)/np.linalg.norm(target) for actual,target in tests]
    assert max(errors) < 1e-12
    print('Added midframe guard %.4f ps, B switching window %.4f ps, max ideal read/write errors %.2g'
          % (guard_ps, (right-left)*period_ps/SAMPLES, max(errors)))
    if guard_ps:
        next_start = frame - (N-1)*delta
        last_end = starts[-1] + 2*SAMPLES
        assert next_start > last_end
        print('B envelope gap at frame reset: %.4f ps' % ((next_start-last_end)*period_ps/SAMPLES))
    if consecutive:
        for wave, gate, ideal in (
                (forward_wave, forward_gate, lambda v: matrix @ v),
                (adjoint_wave, adjoint_gate, lambda v: (matrix.conj().T @ v).conj())):
            vectors = random_complex((3, N))
            waves = [wave(stored, v) for v in vectors]
            # Include all unselected pulses from earlier and later frames at each gate.
            train = np.zeros((len(waves)-1)*frame + len(waves[0]), complex)
            for i, packet in enumerate(waves):
                train[i*frame:i*frame+len(packet)] += packet
            for i, v in enumerate(vectors):
                target = ideal(v) @ DATA_MODES
                np.testing.assert_allclose(train[i*frame+gate]/gain, target, atol=1e-10)
        print('Three consecutive independent vectors passed the forward and adjoint gates.')


def forward_reads(programs, chirp, scale, traversal, pump, samples=512, steps=32,
                  *, coupling=1.0, reference_balance=1.0, early_db=10, input_pj=4,
                  period_ps=20.):
    """Yield target, normalized gated waveform, decoder and output noise per read."""
    # traversal is the power gain compared with one baseline mesh port (2 at
    # zero extra insertion loss); pump is *each* pump's relative power.
    field_ratio = np.sqrt(traversal)
    # Photon occupancy scales with energy. FWM loading scales with window-mean power.
    # The original 2 pJ reference packet occupies one 20 ps Fourier period.
    input_ratio = input_pj / 2
    seed_scale = np.sqrt(input_ratio * (20. / period_ps) * coupling)
    weak_gain = np.sqrt(ETA/N) * field_gain(-INPUT_LOSS_DB-OUTPUT_LOSS_DB) / (4*scale)
    for w, encoded, vectors in programs:
        weight, modes, peak = optical_program(encoded, samples)
        effective = coupling * pump
        weight *= (peak / scale) * np.sqrt(effective/reference_balance)
        receiver = Read(encoded, scale)
        receiver.signal *= (traversal*effective)**2
        receiver.raw *= traversal
        receiver.spontaneous *= traversal*effective**2
        g2, noise, noise_terms = receiver.noise('raw idler', early_db)
        for x in vectors:
            seed = seed_scale * field_ratio * chirp.conj() * x
            weak, actual, *_ = read(weight, modes, seed, 0, steps,
                                    reference_power_w=PUMP_W*effective*reference_balance)
            decoder = samples * seed_scale * np.sqrt(PACKET_POWER_W) * weak_gain * traversal*effective * chirp
            target = w @ x
            np.testing.assert_allclose(np.fft.fft(field_ratio*weak)[:N] / decoder,
                                       target, rtol=1e-10, atol=1e-12)
            yield target, field_ratio*actual, decoder, g2, noise/np.sqrt(input_ratio), noise_terms


def pool(*args, **kwargs):
    totals = np.zeros(3)
    for target, waveform, decoder, g2, noise, _ in forward_reads(*args, **kwargs):
        error = np.fft.fft(waveform)[:N] / decoder - target
        totals += [np.vdot(target,target).real, np.vdot(error,error).real, (noise/100)**2]
    signal, bias, ase = totals
    return g2, 100*np.sqrt(np.array([bias, ase, bias+ase])/signal)


def accuracy_tradeoff(programs, chirp, scale, traversal, coupling, interval, comps):
    ref, loop = comps['reference'], comps['weight loop']
    balance = np.sqrt(loop / ref)
    nominal = Read(programs[0][1], scale)
    nominal.signal *= (traversal * coupling)**2  # signal power transfer for pump = 1
    print('Input pJ, nominal output pJ, G1/G2 dB, pump ratio, fJ/MAC, deterministic/noise/total RMS %')
    for input_pj, early_db, late_db in (
            (4, 10, None), (4, 10, 30), (4, 10, 31), (4, 10, 33), (4, 10, 35),
            (4, 0, 31), (4, 0, 35), (2, 10, 31), (1, 10, 31)):
        # Weak useful power is proportional to the product of the pump powers.
        # With both scaled by 'pump', required G2 changes by -20 log10(pump).
        required_at_one, _, _ = nominal.noise('raw idler', early_db)
        pump = .29 if late_db is None else 10**((required_at_one - late_db) / 20)
        kwargs = dict(coupling=coupling, reference_balance=balance, early_db=early_db, input_pj=input_pj)
        actual_db, errors = pool(programs, chirp, scale, traversal, pump, **kwargs)
        if late_db is not None:
            np.testing.assert_allclose(actual_db, late_db, atol=1e-10)
        # G1 can be fully bypassed. No fractional current credit from reducing dB.
        # Keep 281.25 pJ G2 pulser charge in every case. The existing brighter-
        # output reserve of 50 pJ is specific to the ~4 pJ per-layer input.
        rails = interval * (sum(comps.values()) - ref - loop + pump*(ref*balance + loop/balance))
        if early_db == 0:
            rails -= interval * comps['raw idler gain']
        if input_pj <= 2:
            rails -= 50
        fj_per_mac = (rails + 10) / DC_EFFICIENCY * 1000 / N**2
        print(f'{input_pj} {input_pj*10**.1:.2f} {early_db:2} {actual_db:.2f} '
              f'{pump:.3f} {fj_per_mac:.2f} {errors.round(2)}', flush=True)
        if input_pj == 2:
            args = programs[:1], chirp, scale, traversal, pump
            _, coarse = pool(*args, **kwargs)
            _, fine = pool(*args, samples=1024, steps=64, **kwargs)
            np.testing.assert_allclose(coarse, fine, rtol=.001, atol=.002)


def combined(accuracy=False, ten_ps=False):
    programs, chirp = workload()
    scale = max(optical_program(encoded, N)[2] for _, encoded, _ in programs)
    k = waveguide_field_coefficient(4,.2)/waveguide_field_coefficient(2.12,.8)
    traversal = 2*10**(-.5/10)
    # 40 ps additional longest-arm delay and another 40 ps for frame reset.
    # As a conservative energy bound charge all continuous loads for both guards.
    interval = 3.413 + .080
    comps, _, _, _ = budget(interval, gated_booster=True, pulsed_reference=True, extra_booster_pj=50)
    if ten_ps:
        # 69 source clock ticks, 64 lit. Two missing ticks for switching and three for reset.
        slot_ps = 21.4453125
        check_timing(2*slot_ps, period_ps=10, slot_ps=slot_ps, reset_ps=3*slot_ps, consecutive=True)
        frame_ns = 69 * slot_ps / 1000
        flight_ns = .04 * 4.04 / 299792458 * 1e9
        first_ns = (63*slot_ps + 2*slot_ps + 20)/1000 + flight_ns
        print(f'10 ps frame {frame_ns:.5f} ns, {2*N**2/(frame_ns*1000):.3f} nominal TOPS, '
              f'first packet complete {first_ns:.4f} ns plus other common transit.')
        # Retain the two existing guide pump powers by reusing their original balance.
        # No energy budget for 100 GHz tooth spacing or the faster drivers is assumed.
        balance = np.sqrt(comps['weight loop'] / comps['reference'])
        for period_ps, input_pj in ((20,4), (10,4), (10,2)):
            g2, errors = pool(programs, chirp, scale, traversal, .29, coupling=k,
                              reference_balance=balance, input_pj=input_pj, period_ps=period_ps)
            print(f'{period_ps} ps, {input_pj} pJ, G2 {g2:.2f} dB, '
                  f'deterministic/noise/total field RMS % {errors.round(2)}', flush=True)
        return
    if accuracy:
        accuracy_tradeoff(programs, chirp, scale, traversal, k, interval, comps)
        return
    ref = comps['reference']
    loop = comps['weight loop']
    balance = np.sqrt(loop/ref)
    print('4 cm/0.2 dB per cm, pulsed reference, +0.5 dB/select, 3.493 ns charged, fully dense matrices.')
    print('guide reference and nominal weight (mW), G1 and G2 (dB), deterministic/noise/total %, fJ/MAC,'
          ' remaining pJ on selector rails at 200 fJ/MAC')
    for pump, early in [(.27,10),(.29,10),(.30,10),(.34,10),(.38,10),(.45,0),(.50,0),(.60,0)]:
        g2, errs = pool(programs,chirp,scale,traversal,pump,coupling=k,reference_balance=balance,early_db=early)
        rails = interval * (sum(comps.values()) - ref-loop + ref*pump*balance + loop*pump/balance)
        if early == 0:
            rails -= interval*comps['raw idler gain']
        e_pj = (rails + 10)/DC_EFFICIENCY
        print(f'{50*pump*balance:6.1f} {50*pump/balance:6.1f} {early:4} {g2:6.2f} '
              f'{errs.round(2)} {e_pj*1000/N**2:7.2f} {200*N**2*.001*DC_EFFICIENCY-rails:7.1f}',
              flush=True)
        if pump == .29:
            args = programs[:1],chirp,scale,traversal,pump
            kw = dict(coupling=k,reference_balance=balance,early_db=early)
            _, coarse = pool(*args,**kw)
            _, fine = pool(*args,samples=1024,steps=64,**kw)
            np.testing.assert_allclose(coarse, fine, rtol=.001, atol=.002)


def main():
    check_timing(0)
    check_timing(40)
    programs, chirp = sparse_workload()
    scale = max(optical_program(encoded, N)[2] for _, encoded, _ in programs)
    comps, _, _, _ = budget(3.16, gated_booster=True, extra_booster_pj=50)
    print('Per standalone read: nominal useful gain +1 dB, G1 +10 dB. Four-pJ input.')
    print('selector IL (dB/pass), both pump ratios, final gain dB, deterministic / noise / total %, fJ/MAC')
    for loss, pump in [(None,1), (0,1), (0,.5), (.5,1), (.5,2**-.5), (.5,.5)]:
        traversal = 1 if loss is None else 2 * 10**(-loss/10)
        g2, errors = pool(programs, chirp, scale, traversal, pump)
        # Allow 10 pJ/read on the rails for all switching. Hold both gain-die
        # electrical budgets fixed even when nominal G2 changes.
        drive_pj = 0 if loss is None else 10
        energy_pj = (3.16 * (sum(comps.values()) -
                    (1-pump)*(comps['reference']+comps['weight loop'])) + drive_pj) / DC_EFFICIENCY
        print(f'{str(loss):>6} {pump:9.4f} {g2:8.2f} {errors.round(2)} {1e3*energy_pj/N**2:8.2f}',
              flush=True)
    args = (programs[:1], chirp, scale, 2*10**(-.5/10), 2**-.5)
    _, coarse = pool(*args)
    _, fine = pool(*args, samples=1024, steps=64)
    np.testing.assert_allclose(coarse, fine, rtol=.001, atol=.002)
    combined()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument('--accuracy', action='store_true', help='run the single-read energy/error comparison')
    mode.add_argument('--ten-ps', action='store_true', help='check 10 ps timing and the forward error estimate')
    args = parser.parse_args()
    if args.ten_ps:
        combined(ten_ps=True)
    elif args.accuracy:
        combined(accuracy=True)
    else:
        main()
