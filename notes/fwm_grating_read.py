"""One isolated 64-row FWM read with a continuous chirped fiber grating.

Run with NumPy. Add --waveform to propagate the full field independently and
measure deterministic error in the light leaving the gate. See scattering_multipliers.md
for the component measurements, optical reference planes and noise assumptions.
"""

import argparse

import numpy as np


TEETH = 64
SPACING_GHZ = 50
REPLAY_TRANSMISSION = 10 ** (-5.2 / 10)
FWM_CONVERSION = 0.468
GRATING_TRANSMISSION = 10 ** (-3 / 10)
PREAMP_GAIN = 10.0
NOISE_FACTOR = 10 ** (5 / 10)
PHOTONS_PER_TOOTH = 2e-12 * 1550e-9 / (6.62607015e-34 * 299792458 * TEETH)


def pulse_spectrum(frequency, width, cosine):
    """Fourier transform of a unit-energy square or half-cosine field pulse."""
    offset = frequency * width
    if cosine:
        return np.sqrt(width / 2) * (np.sinc(offset - 0.5) + np.sinc(offset + 0.5))
    return np.sqrt(width) * np.sinc(offset)


def grating_phase(frequency, spacing, ripple_ps):
    """Quadratic phase plus optional sinusoidal delay ripple over four teeth."""
    period = 4 * spacing
    row_ps = 1000 * spacing / SPACING_GHZ
    return (np.pi * frequency**2 / spacing
            + ripple_ps / row_ps * period * np.cos(2 * np.pi * frequency / period))


def grating_kernel(spacing, cosine_input, gate_width, cosine_output, samples=256, ripple_ps=0):
    """Overlap from each (emission slot, generated idler) to each output mode.

    Time unit is one row slot. Frequency unit is its reciprocal. H(f) gives a
    delay of -j slots at tooth j after removing the common propagation delay.
    Integrate over the entire reflection band, including the tails of idlers
    centered outside it. The FFT evaluates all integer row shifts together.
    """
    teeth = np.arange(TEETH)
    idlers = np.arange(1 - TEETH, 2 * TEETH - 1)
    first = -spacing / 2 + 0.5 / samples
    frequency = first + np.arange(TEETH * spacing * samples) / samples
    reflection = np.exp(1j * grating_phase(frequency, spacing, ripple_ps))
    source = pulse_spectrum(frequency[:, None] - spacing * idlers, 1, cosine_input)
    row_phase = np.exp(-2j * np.pi * first * teeth) / samples
    kernel = np.empty((TEETH, TEETH, len(idlers)), complex)
    noise_bandwidth = np.empty(TEETH)
    for k in teeth:
        receive = pulse_spectrum(frequency - spacing * k, gate_width, cosine_output)
        noise_bandwidth[k] = np.sum(receive**2) / samples
        integrand = (receive * reflection)[:, None] * source
        folded = integrand.reshape(TEETH * spacing, samples, len(idlers)).sum(axis=0)
        phase_correction = np.exp(-1j * grating_phase(spacing * k, spacing, ripple_ps))
        kernel[:, k, :] = (phase_correction * row_phase[:, None]
                           * np.fft.fft(folded, axis=0)[:TEETH])
    return kernel, np.mean(noise_bandwidth)


def read_metrics(weights, kernel, noise_bandwidth):
    # Weight h in row r and input i generate idler j = r+h-i.
    teeth = np.arange(TEETH)
    offsets = teeth[:, None] - teeth[None, :] + (TEETH - 1)
    rows = np.empty((TEETH, TEETH, TEETH), complex)
    for r in teeth:
        rows[r] = np.einsum("khi,h->ki", kernel[r][:, r + offsets], weights[r])
    transfer = rows.sum(axis=0)
    scale = np.vdot(weights, transfer) / np.vdot(weights, weights)
    distortion = np.linalg.norm(transfer / scale - weights) / np.linalg.norm(weights)
    gain = REPLAY_TRANSMISSION * FWM_CONVERSION * GRATING_TRANSMISSION * abs(scale)**2 / TEETH
    postamp_gain = 1 / (gain * PREAMP_GAIN)

    # Independent vacuum inputs on 64 data modes per row. Products generated
    # from the same input mode in a row interfere before their norm is taken.
    pairs = FWM_CONVERSION * np.sum(abs(rows)**2) / TEETH
    preamp_noise = (NOISE_FACTOR * PREAMP_GAIN - 1) / 2
    postamp_noise = (NOISE_FACTOR * postamp_gain - 1) / 2
    photons = (postamp_gain * GRATING_TRANSMISSION
               * (PREAMP_GAIN * pairs + noise_bandwidth * preamp_noise) + postamp_noise)
    noise = np.sqrt(photons / PHOTONS_PER_TOOTH)
    return (10 * np.log10(gain), 100 * distortion, 100 * noise,
            100 * np.hypot(distortion, noise))


def waveform_error(weights, spacing, cosine_input, gate_width, cosine_output, ripple_ps):
    """Generate each basis input, reflect its continuous field, and gate in time."""
    teeth = np.arange(TEETH)
    points = 512 * spacing
    window = 256  # Slots in the padded FFT frame.
    first = (window // 2) * points - points // 2
    time = (np.arange(points) + 0.5) / points - 0.5
    carriers = np.exp(2j * np.pi * spacing * time[:, None] * teeth)
    pulse = np.sqrt(2) * np.cos(np.pi * time) if cosine_input else np.ones(points)
    field = np.zeros(window * points, complex)
    # For lit input i=0, form the weight comb in each slot, shifted by r teeth.
    field[first:first + TEETH * points] = ((pulse[:, None] * carriers)
                                           * (carriers @ weights.T)).T.ravel()
    frequency = np.fft.fftfreq(len(field), 1 / points)
    low, high = -0.5 * spacing, (TEETH - 0.5) * spacing
    band = ((frequency > low) & (frequency < high)).astype(float)
    band[(frequency == low) | (frequency == high)] = 0.5
    reflection = band * np.exp(1j * grating_phase(frequency, spacing, ripple_ps))
    spectrum = np.fft.fft(field)
    gate = np.flatnonzero(abs(time) < gate_width / 2)
    measured = np.empty((len(gate), TEETH), complex)
    for i in teeth:
        # Lighting input i shifts that same spectrum down by i teeth.
        shifted = np.roll(spectrum, -i * spacing * window)
        shifted *= np.exp(-1j * np.pi * spacing * i / points)
        measured[:, i] = np.fft.ifft(shifted * reflection)[first + gate]
    receive = np.ones(len(gate)) / np.sqrt(gate_width)
    if cosine_output:
        receive *= np.sqrt(2) * np.cos(np.pi * time[gate] / gate_width)
    expected_modes = (receive[:, None] * carriers[gate]
                      * np.exp(1j * grating_phase(spacing * teeth, spacing, ripple_ps)))
    expected = expected_modes @ weights
    scale = np.vdot(expected, measured) / np.vdot(expected, expected)
    projected = expected_modes.conj().T @ measured / points
    mode_error = np.linalg.norm(projected / scale - weights) / np.linalg.norm(weights)
    raw_error = np.linalg.norm(measured / scale - expected) / np.linalg.norm(expected)
    return 100 * mode_error, 100 * raw_error


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--waveform", action="store_true", help="also evaluate the full gated field")
    parser.add_argument("--ripple-ps", type=float, default=0.0,
                        help="peak sinusoidal group-delay error in ps, period 200 GHz")
    args = parser.parse_args()
    targets = [np.eye(TEETH)]
    for seed in range(10):
        sample = np.random.default_rng(seed).standard_normal((TEETH, TEETH))
        q, r = np.linalg.qr(sample)
        targets.append(q * np.sign(np.diag(r)))

    # Input and reference/weight overlap lasts a full row. The transmit and
    # receive tooth modes in each listed case are internally orthonormal.
    cases = [(20, False, 20, False), (60, True, 60, True),
             (60, True, 20, False), (80, True, 20, False)]
    for row_ps, cosine_input, gate_ps, cosine_output in cases:
        spacing = row_ps * SPACING_GHZ // 1000
        width = gate_ps / row_ps
        kernel, bandwidth = grating_kernel(spacing, cosine_input, width, cosine_output,
                                          ripple_ps=args.ripple_ps)
        results = []
        for weights in targets:
            metrics = read_metrics(weights, kernel, bandwidth)
            if args.waveform:
                projected, raw = waveform_error(weights, spacing, cosine_input, width,
                                                cosine_output, args.ripple_ps)
                if abs(projected - metrics[1]) > 0.05:  # Percentage points.
                    raise ValueError("Time propagation and overlap integral disagree")
                metrics += (raw,)
            results.append(metrics)
        dense = np.array(results[1:])
        print(f"\n{row_ps} ps rows / {gate_ps} ps gate")
        print("                 gain (dB), distortion (%), noise (%), total in modes (%)"
              + (", whole gated waveform distortion (%)" if args.waveform else ""))
        for label, values in [("identity", results[0]), ("dense min", dense.min(axis=0)),
                              ("dense max", dense.max(axis=0))]:
            print(f"{label:12}", "  ".join(f"{value:7.3f}" for value in values))


if __name__ == "__main__":
    main()
