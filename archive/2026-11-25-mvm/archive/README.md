# Archived optical multiplier investigations

- **Stimulated Raman scattering.** Gain measurements, candidate matrices and noise estimates, with an open question about using exponential gain for floating-point encoding. [Raman notes](srs.md)
- **SFG/DFG Hankel multiplier.** The dense-matrix power budget, reusable architecture and editable figures are documented below. [Power budget](#why-we-retired-the-multiplier) · [Figures](#editable-figures)

## Why we retired the multiplier

Two difference-frequency generation (DFG) stages and a coherent sum over 64 slots could represent any complex 64 × 64 local matrix in the ideal model. Sum-frequency generation (SFG) provided the weight update. Too little coherent light reached the next bank on the dense-matrix benchmark.

The simulation used one dense unitary, four signed inputs, 200 µW per input tooth, a 180 mW in-guide weight cap per slot, flat phase matching, and a 6 dB shuffle. The read window was 640 ps.

| Reader | Routed bank power loss | Mean coherent photons per tooth |
| --- | ---: | ---: |
| Hankel pair and temporal sum, 200 mW reference per guide | 58.3–58.4 dB | 1.45–1.48 |
| Single-DFG row dot products, ideal lossless retiming | 29.1–29.2 dB | ≈1,212 |

The row reader required an estimated 2.27 W circulating per lane at a 10% weight tap. Screen a replacement on dense matrices, including faint output coordinates, accumulated amplifier noise, and weight-delivery energy.

## What carries forward

- **Weight memory.** Inject, tap, and replenish time-slotted weights. The fixed-MVM design used a 680 ps loop (64 × 10 ps plus 40 ps dark) and 1.36 ns between inputs. A separate ~150 m loop was proposed for test-time training (TTT). Both bank gradients use the pre-update state, then the query reads the updated state.
- **Monarch shuffle.** Eight lanes × 64 teeth on a 100 GHz grid. Only data crosses between banks. Tooth group `r = k mod 8` moves from lane `j` to `(j+r) mod 8` at unchanged frequency. The resulting 512 × 512 map has rank at most eight per lane-pair block. The AWGR routing estimate was 3 dB. The later Mach–Zehnder interleaver budget was 6 dB from couplers alone.
- **Free-space activation.** Disperse wavelengths along one axis, image fibers along the other, focus onto 512 spots on a saturable absorber mirror, then recombine each comb. Model fluence, recovery, crosstalk, and round-trip loss.
- **Paper.** The Nature Communications plan paired noisy MLP simulations with offline language-model training and optical TTT updates during inference. The ~310M-real-parameter model was a sizing study. Task accuracy, latency, and energy per token remain open.

## Editable figures

- [1a · Hankel cell](figures/inkscape_diagrams/simplified_hankel_multiplier_cleaned.svg) · [1b · recirculating memory, forward read, and update](figures/inkscape_diagrams/1b_forward_backward_hankel_compact.svg)
- [2a · grouped shuffle, four-lane illustration](figures/inkscape_diagrams/2a_hankel_shuffle.svg) · [2b · reflective free-space activation](figures/inkscape_diagrams/2b_free_space_saturable_absorption.svg)
- [Component library](figures/self_diagram/components/) · [TTT drawing](figures/self_diagram/TTT_model.svg) · [System layouts and timing diagrams](figures/hankel_monarch_delta/figures/)

## Original notes, code, and results

All 39 SVGs from the Hankel investigation are directly editable. Its other 81 files are in the [source bundle](sources.tar.xz). Its entry points are `simulation_notes/matmul_power_budget.md` (calculations, alternatives, literature), `hankel_monarch/limitations.md` (expressivity), `hankel_monarch_delta/paper_design_point.md` (architecture and timing), and `paper_notes.md` (five-figure outline). Read one from the repo root with `tar -xJOf archive/sources.tar.xz paper_notes.md`.

Restore and verify all 120 originals in a fresh directory with

```sh
restored=$(mktemp -d)
tar -xJf archive/sources.tar.xz -C "$restored"
cp -a archive/figures/. "$restored/"
(cd "$restored" && sha256sum --quiet -c ARCHIVE-SHA256SUMS)
```
