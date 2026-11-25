# Energy-efficient optical matrix–vector multiplication

The goal is to minimize electrical energy for one forward multiply `y = Wx` with an independently programmable, signed-real 128 × 128 matrix. That is 16,384 useful multiply–accumulates (MACs) per vector. Intel’s measured INT8 compute-in-memory macro gives a reference of **13.6 fJ/MAC**, or **223 pJ per vector** at this size. The optical simulations target approximate inference. Compare designs at matched accuracy and include the cost of delivering each result to its consumer. [Intel measurement](https://ieeexplore.ieee.org/document/11409207/)

## Where low energy could come from

**For weights held across many vectors, start with spatial fan-out.** Encode an input once, distribute it across resident weights, and collect a full row sum at each detector. Programming energy can then be spread over the vectors that use the matrix. Bernstein et al. project **about 10 fJ/MAC at 1,000 × 1,000**, including sources, spatial modulation, converters and activation. The open question here is how much of that amortization survives at 128 × 128 and batch one. [Single-shot optical neural network](https://doi.org/10.1126/sciadv.adg7904)

**For weights supplied by another optical beam, start with weak sum-frequency generation (SFG).** Convert a small part of the input into each answer and pass the surviving input on to more rows. The table gives modeled **electrical energy for the optical sources**, including data, shaped weights and the coherent readout reference. All three designs implement the full dense matrix. [Model and assumptions](archive/2026-11-25-mvm/notes/scattering_multipliers.md#building-an-arbitrary-dense-mvm-from-two-useful-hankel-outputs)

| How the waveguide computes rows | fJ/MAC at 16 rows per input copy | Best modeled fJ/MAC (rows per copy) |
| --- | ---: | ---: |
| Select one weighted sum per pass | 5.39 | 4.74 (32) |
| Collect two Hankel outputs over two passes, flipping the phase of one output between passes | 3.86 | 3.40 (32) |
| Collect two independent rows in one pass using interleaved pump frequencies | 4.87 | 3.36 (64) |

The estimates assume 0.2 dB loss per transit, 40% data-source efficiency, 25% shaped-weight delivery, and 10% RMS error from terminal readout alone. The two-output designs extrapolate a measured single-output conversion efficiency. Receiver electronics, programming and any gain needed to feed another optical layer are additional costs. In the two-pass simulation at 16 rows per copy, deterministic field error was 0.65% for random dense rows and inputs and 8.06% for aligned positive rows and inputs, before readout noise. [Single-output experiment](https://doi.org/10.1103/PhysRevResearch.6.L022040) · [Propagation simulation](archive/2026-11-25-mvm/notes/paired_hankel_stress.py)

**The first comparison to run** is one row versus two coded rows with finite 20 ps pulses. At the assumed loss, the two-pass design saves source energy if its useful conversion per pump joule exceeds 51% of the single-row value. At 3.86 fJ/MAC, the remaining allowance under the silicon reference is **1.25 pJ per output** for all omitted costs. Verify pulse overlap and total field error, then price the receiver or the optical link to the next layer. [Energy calculation](archive/2026-11-25-mvm/notes/two_output_mvm.py)

The earlier recirculating four-wave-mixing design illustrates the cost of pumping and restoring a signal on 66 passes. It projected 81.1 fJ/MAC and gave 42.75% calibrated field error on one tested program. A single Hankel pass remains relevant for a layer that actually uses those shared coefficients. Compare that case with electronic convolution exploiting the same structure. [Earlier loop comparison](archive/2026-11-25-mvm/notes/scattering_multipliers.md#local-128--128-mvm-against-measured-intel-18a-digital-compute-in-memory)

## Archive

The snapshot preserves the simulations, results, editable SVGs, visualizations and earlier source bundle. Run historical commands from `archive/2026-11-25-mvm/`. [Previous project overview](archive/2026-11-25-mvm/README.md) · [Multiplier layouts](archive/2026-11-25-mvm/figures/) · [Earlier figures and source bundle](archive/2026-11-25-mvm/archive/README.md)
