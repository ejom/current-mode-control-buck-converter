# 11/13 simulations: open-loop distributed buck converter

These runs cover the open-loop converter from Röbenack & Gerbet (2024) at the paper's operating point:

- Supply and output: E = 12 V, R = 10 Ω, C_ext = 1 µF.
- Line: 6 m RG-58 with L' = 241 nH/m, C' = 100 pF/m, R' = 40 mΩ/m.
- Operating point: D = 0.512, which gives about 6 V out.

They compare three plants:

1. **Lumped (discrete L, R_L).** L = 1.446 µH, R_L = 0.24 Ω, C = 1.0006 µF (paper Eq. 22).
2. **Distributed.** The 6 m lossy line, i.e. the circuit in the LTspice schematics.
3. **Distributed + input element.** The line with L_in = 0.5 µH ∥ R_p = 47 Ω between the switch and the line, as in `input_filter_design.py`.

The half bridge is an ideal synchronous switch: a 0–12 V PWM with 1 ns edges and R_sw = 25 mΩ (the IRLML6244 R_DS(on)). This is the paper's E·d(t) model, without the AD8561 comparator and LTC4449 gate driver.

## Figures

| File | What it shows |
|---|---|
| `01_lumped_vs_line.png` | Start-up and steady state, lumped vs. line, at 2 MHz. Both models have the same output dynamics: they ring at ~134 kHz (ζ = 0.16) and settle to 5.99 V. The input current is very different. The line current climbs in steps every round trip of 2τ = 59 ns, and the first step is E/Z₀ = 0.24 A. In steady state it rings at ~17 MHz on top of the inductor triangle, so the RMS ripple is 639 mA vs. 301 mA. |
| `02_duty_cycle_sweep.png` | Duty-cycle sweep D = 0.4 / 0.5 / 0.512 / 0.6 (V_ref = 4.8 / 6 / 6.144 / 7.2 V in LTspice), with direct labels and a table. V_out = E·D·R/(R+R_L+R_sw) ≈ 11.70·D, and SPICE agrees to 0.01 %. |
| `03_ripple_vs_switching_frequency.png` | RMS input-current ripple vs. f_s from 2 to 10 MHz, with the five LTspice sweep frequencies marked. Every spike is a PWM harmonic k·f_s landing on a line series resonance n/(2τ) = n·16.98 MHz. |
| `04_input_current_waveforms_vs_fs.png` | Steady-state input current over 3 PWM periods at 2, 4, 6, 7.96 and 8.49 MHz, for all three plants. |
| `05_discrete_L_damping_and_ccm.png` | Lumped-model check of the hand calculations. (a) Step response for R = 10, 2, R_crit = 0.547 and 0.25 Ω: ζ = 1.00 exactly at R_crit, with no overshoot. (b) Inductor current at 2 MHz for R = 5, 10, R_crit = 11.85 and 20 Ω: i_L,min ≈ 0 at R_crit, and above it the synchronous switch drives the current negative. |
| `06_model_check_spice_vs_exact.png` | Validation of the models against each other at the five frequencies: exact line, 200-section ladder, ngspice transient and 25-section ladder. |
| `07_summary_table.png` | One table with L_crit, R_crit and ripple (exact and SPICE) for every f_s, with and without the input element. |

## Key results

- **The discrete L–R_L model is right for v_out but wrong for the input current.** Both models settle to the same output voltage with the same ringing. The input-current ripple, however, is 2–3× higher with the line, and up to 38× higher on a resonance.
- **Avoid 8.49 MHz (= f₁) without an input filter.** The fundamental sits on the line's antiresonance, but the 2nd harmonic (16.98 MHz) sits exactly on the first series resonance. At D = 0.512 ≠ 0.5 that gives 2.7 A RMS ripple (exact model). The paper's choice of **7.96 MHz** sits between resonances: 189 mA.
- **The L_in ∥ R_p element removes the resonances.** Ripple drops to 219 → 18 mA across 2–8.49 MHz, the worst case falls from ~12 A to 0.22 A, and all models agree to within a few mA.
- **Every case is in CCM** (L = 1.446 µH > L_crit = 1.22 µH at 2 MHz, R = 10 Ω < R_crit = 11.85 Ω). At 2 MHz the margin is only about 18 %.

## How it was computed

- **Transients ("SPICE")** use ngspice 42, with the line built from 200 RLC sections. LTspice's `LTRA` element was also tried in ngspice. It was slow and inaccurate there, settling to 6.08 V instead of 5.99 V, so the ladder is used instead.
- **"Exact"** is the periodic steady state of the paper's distributed model, Eq. (20). It comes from Fourier synthesis over the PWM harmonics, using the same method as `input_filter_design.py` with R_sw and the 1 ns edges included. It is exact for an ideal switch.
- **Caveat:** the line model has DC resistance only, with no skin effect. Real resonance peaks will be lower and wider, so the numbers on a resonance spike are qualitative. The trends and the numbers between spikes are solid.

## Re-running

```
cd images/1113/sim
python runs.py          # ngspice transients -> runs.pkl  (needs ngspice on PATH, ~6 min)
python make_figures.py  # writes the PNGs into images/1113
```

`analytic.py` imports `bode.py` from the repo root.
