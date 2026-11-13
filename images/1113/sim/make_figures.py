"""Figures for images/1113: open-loop distributed buck converter, Röbenack & Gerbet (2024).

Inputs: runs.pkl (ngspice transients from runs.py) + exact frequency-domain steady state
(analytic.py, built on bode.py). Run from this folder (needs ngspice on PATH):  python runs.py && python make_figures.py
"""

import pickle
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

import analytic as an
import spice as sp

OUT = str(Path(__file__).resolve().parent.parent)  # images/1113
SURF, INK, INK2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#8a8984", "#e4e3df"
BLUE, ORANGE, AQUA = "#2a78d6", "#eb6834", "#1baf7a"
RAMP = ["#6da7ec", "#2a78d6", "#1c5cab", "#0d366b"]  # ordinal blue for D

mpl.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "axes.spines.top": False, "axes.spines.right": False,
    "font.size": 10.5, "axes.titlesize": 11.5, "axes.titleweight": "bold",
    "axes.titlelocation": "left", "legend.frameon": False, "lines.linewidth": 2,
    "mathtext.fontset": "dejavusans",
})
TAU = sp.LEN * np.sqrt(sp.LP * sp.CP)  # one-way delay of the line
F_SERIES = 1 / (2 * TAU)               # first series resonance of the shorted line


def save(fig, name):
    fig.savefig(f"{OUT}/{name}", dpi=200, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


def note(fig, text):
    fig.text(0.0, -0.01, text, ha="left", va="top", fontsize=8.5, color=MUTED, wrap=True)


def table(ax, header, rows, col_w=None, highlight=None, fs=9.5):
    ax.axis("off")
    t = ax.table(cellText=rows, colLabels=header, loc="upper center", cellLoc="center",
                 colWidths=col_w)
    t.auto_set_font_size(False)
    t.set_fontsize(fs)
    t.scale(1, 1.55)
    for (r, c), cell in t.get_celld().items():
        cell.set_edgecolor(GRID)
        if r == 0:
            cell.set_height(cell.get_height() * 1.9)
            cell.set_facecolor("#eeede9")
            cell.set_text_props(weight="bold", color=INK)
        else:
            cell.set_facecolor(SURF if r % 2 else "#f5f5f2")
            if highlight and r - 1 in highlight:
                cell.set_text_props(weight="bold")
    return t


R = pickle.load(open("runs.pkl", "rb"))
us = 1e6


# ---------------------------------------------------------------- 1 lumped vs line
def fig_lumped_vs_line():
    lu, li = R["lumped_D0.512"], R["line_D0.512"]
    fs, T = 2e6, 0.5e-6
    fig, axs = plt.subplots(3, 1, figsize=(10, 11.5), gridspec_kw={"hspace": 0.5})

    ax = axs[0]
    ax.plot(lu["t"] * us, lu["v"], color=ORANGE, lw=3.2, label="lumped: discrete L = 1.446 µH, R$_L$ = 0.24 Ω")
    ax.plot(li["t"] * us, li["v"], color=BLUE, lw=1.4, label="distributed: 6 m lossy line")
    ax.axhline(an.vout_dc(0.512), color=MUTED, lw=1, ls="--")
    ax.text(60.5, an.vout_dc(0.512), f"{an.vout_dc(0.512):.2f} V", va="center", color=INK2, fontsize=9)
    ax.set(xlim=(0, 60), xlabel="time t in µs", ylabel="output voltage $v_{out}$ in V",
           title="(a) Start-up of the output voltage: both models ring at ~134 kHz and settle to the same value")
    ax.legend(loc="lower right")

    ax = axs[1]
    ax.plot(lu["t"] * us, lu["i"], color=ORANGE, label="lumped")
    ax.plot(li["t"] * us, li["i"], color=BLUE, lw=1.4, label="distributed (line)")
    for k in range(1, 9):
        ax.axvline(2 * k * TAU * us, color=GRID, lw=1, zorder=0)
    ax.text(0.99, 0.04, f"vertical lines: every 2τ = {2*TAU*1e9:.0f} ns (round trip on the line)\n"
            f"first step = E / Z$_0$ = 12 V / {np.sqrt(sp.LP/sp.CP):.0f} Ω = {12/np.sqrt(sp.LP/sp.CP):.2f} A",
            transform=ax.transAxes, fontsize=9, color=INK2, ha="right", va="bottom")
    ax.set(xlim=(0, 1.0), ylim=(-0.3, 4.5), xlabel="time t in µs", ylabel="input current $i(0,t)$ in A",
           title="(b) First microsecond: the line current climbs in steps every round trip, the inductor current ramps")
    ax.legend(loc="upper left")

    ax = axs[2]
    t = np.linspace(0, 2 * T, 4000)
    il, _ = an.steady_wave(t, fs, model="lumped")
    id_, _ = an.steady_wave(t, fs, model="line")
    m = sp.steady_metrics(li, fs, n_periods=8)
    sel = (li["t"] >= li["t"][-1] - 2 * T)
    tt = (li["t"][sel] - (li["t"][-1] - 2 * T))
    ax.axvspan(0, 0.512 * T * us, color="#f0f0ec", zorder=0)
    ax.axvspan(T * us, (1.512) * T * us, color="#f0f0ec", zorder=0)
    ax.plot(t * us, il, color=ORANGE, label=f"lumped, exact steady state (RMS ripple {il.std()*1e3:.0f} mA)")
    ax.plot(t * us, id_, color=BLUE, label=f"line, exact steady state (RMS ripple {id_.std()*1e3:.0f} mA)")
    ax.plot(tt * us, li["i"][sel], color=INK, lw=1, ls=(0, (3, 2)),
            label=f"line, ngspice 200-section ladder (RMS ripple {m['Irms_ripple']*1e3:.0f} mA)")
    ax.axhline(0, color=MUTED, lw=1)
    ax.text(0.02, 0.97, "shaded = switch on (D = 0.512)", transform=ax.transAxes,
            fontsize=8.5, color=MUTED, va="top")
    ax.set(xlim=(0, 2 * T * us), xlabel="time within two PWM periods in µs",
           ylabel="input current $i(0,t)$ in A",
           title="(c) Steady state at f$_s$ = 2 MHz: the line adds reflection ringing on top of the inductor triangle")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.17), ncol=1, fontsize=9.5)
    fig.suptitle("Discrete L–R$_L$ model vs. lossy transmission line  ·  open loop, D = 0.512, f$_s$ = 2 MHz, R = 10 Ω",
                 x=0.0, ha="left", fontsize=13, fontweight="bold", y=0.94)
    note(fig, "Ideal synchronous switch (E = 12 V, 1 ns edges, R$_{sw}$ = 25 mΩ). (a), (b) and the dashed trace in (c): "
              "ngspice transient with the line as 200 RLC sections. Solid traces in (c): exact periodic steady state "
              "from the paper's Eq. (20) line model (Fourier synthesis).")
    save(fig, "01_lumped_vs_line.png")


# ---------------------------------------------------------------- 2 duty sweep
def fig_duty():
    Ds = (0.4, 0.5, 0.512, 0.6)
    fig = plt.figure(figsize=(10, 8.6))
    gs = fig.add_gridspec(2, 1, height_ratios=[2.3, 0.75], hspace=0.3)
    ax = fig.add_subplot(gs[0])
    rows = []
    for D, c in zip(Ds, RAMP):
        r = R[f"line_D{D}"]
        m = sp.steady_metrics(r, 2e6, n_periods=8)
        ax.plot(r["t"] * us, r["v"], color=c, lw=1.6, label=f"D = {D}")
        ax.text(60.8, m["Vavg"], f"D = {D}  →  {m['Vavg']:.2f} V", va="center", fontsize=9.5,
                color=INK)
        th = an.vout_dc(D)
        t = np.linspace(0, 0.5e-6, 2000)
        i, v = an.steady_wave(t, 2e6, D=D)
        il, _ = an.steady_wave(t, 2e6, D=D, model="lumped")
        rows.append([f"{D}", f"{12*D:.3f}", f"{th:.3f}", f"{m['Vavg']:.3f}",
                     f"{(m['Vavg']-th)/th*100:+.2f} %", f"{(v.max()-v.min())*1e3:.0f}",
                     f"{il.std()*1e3:.0f}", f"{i.std()*1e3:.0f}"])
    # spread the D=0.5 / 0.512 labels apart
    for txt in ax.texts:
        if txt.get_text().startswith("D = 0.5 "):
            txt.set_y(txt.get_position()[1] - 0.13)
        if txt.get_text().startswith("D = 0.512"):
            txt.set_y(txt.get_position()[1] + 0.13)
    ax.set(xlim=(0, 60), ylim=(0, 12), xlabel="time t in µs", ylabel="output voltage $v_{out}$ in V",
           title="Start-up of $v_{out}$ for each duty cycle (6 m lossy line, f$_s$ = 2 MHz)")
    ax.legend(loc="lower right", ncol=4, bbox_to_anchor=(1, 0.02))
    fig.suptitle("Duty-cycle sweep: $V_{out} = E\\,D\\,R/(R+R_L+R_{sw})$ ≈ 11.70·D holds for the distributed converter",
                 x=0.0, ha="left", fontsize=13, fontweight="bold", y=0.97)
    tax = fig.add_subplot(gs[1])
    table(tax, ["D", "V$_{ref}$ in LTspice\n(= 12·D) in V", "V$_{out}$ theory\nin V",
                "V$_{out}$ SPICE\nin V", "error", "Δv$_{out}$ p-p\nin mV",
                "I$_{in}$ RMS ripple\nlumped in mA", "I$_{in}$ RMS ripple\nline in mA"],
          rows, highlight=[2])
    note(fig, "V$_{out}$ SPICE: mean over the last 8 PWM periods of a 60 µs ngspice run (200-section ladder line). "
              "Ripple columns: exact periodic steady state. Bold row = paper's operating point (6 V). "
              "The output ripple barely changes with D, but the line roughly doubles the input-current ripple "
              "compared with the discrete L.")
    save(fig, "02_duty_cycle_sweep.png")


# ---------------------------------------------------------------- 3 ripple vs fs
FS5 = (2e6, 4e6, 6e6, 7.96e6, 8.49e6)


def fig_ripple_fs():
    f = np.linspace(2e6, 10e6, 4001)
    r_line, r_lump, r_filt = an.ripple(f), an.ripple(f, model="lumped"), an.ripple(f, filt=True)
    fig, ax = plt.subplots(figsize=(11, 6.4))
    ax.semilogy(f / 1e6, r_line * 1e3, color=BLUE, lw=1.6, label="distributed (6 m line)")
    ax.semilogy(f / 1e6, r_lump * 1e3, color=ORANGE, label="lumped (discrete L, R$_L$)")
    ax.semilogy(f / 1e6, r_filt * 1e3, color=AQUA, label="line + L$_{in}$ = 0.5 µH $\\parallel$ R$_p$ = 47 Ω")
    for k, xy in ((2, (9.25, 9000)), (3, (6.4, 16000))):
        fk = F_SERIES / k
        ax.annotate(f"{k}·f$_s$ = {F_SERIES/1e6:.2f} MHz\n(series resonance)", (fk / 1e6, an.ripple(fk)[0] * 1e3),
                    xytext=xy, fontsize=8.5, color=INK2, ha="center",
                    arrowprops=dict(arrowstyle="-", color=MUTED, lw=0.8))
    for fs in (7.96e6, 8.49e6):
        ax.axvline(fs / 1e6, color=MUTED, lw=1, ls="--", zorder=0)
    for fs in FS5:
        rl, rf = an.ripple(fs)[0] * 1e3, an.ripple(fs, filt=True)[0] * 1e3
        ax.plot(fs / 1e6, rl, "o", ms=11, mfc=SURF, mec=INK, mew=2.0, zorder=5)
        ax.plot(fs / 1e6, rf, "o", ms=11, mfc=AQUA, mec=INK, mew=2.0, zorder=5)
    ann = {2e6: (24, 10), 4e6: (0, 12), 6e6: (0, 12), 7.96e6: (-30, 14), 8.49e6: (28, 6)}
    for fs in FS5:
        rl = an.ripple(fs)[0] * 1e3
        dx, dy = ann[fs]
        ax.annotate(f"{rl:,.0f} mA", (fs / 1e6, rl), xytext=(dx, dy), textcoords="offset points",
                    ha="center", fontsize=9, color=INK, fontweight="bold")
    ax.text(7.96, 3.2, "7.96 MHz\n(paper)", ha="right", fontsize=8.5, color=INK2)
    ax.text(8.49, 3.2, " 8.49 MHz\n (= f$_1$)", ha="left", fontsize=8.5, color=INK2)
    ax.set(xlim=(2, 10), ylim=(2, 3e4), xlabel="switching frequency f$_s$ in MHz",
           ylabel="RMS ripple of the line input current in mA (log)",
           title="Each spike = a PWM harmonic k·f$_s$ landing on a line series resonance n/(2τ) = "
                 f"n·{F_SERIES/1e6:.2f} MHz")
    ax.legend(loc="lower left", ncol=1)
    fig.suptitle("Switching-frequency sweep 2–10 MHz at D = 0.512: input-current ripple",
                 x=0.0, ha="left", fontsize=13, fontweight="bold", y=1.0)
    note(fig, "Exact periodic steady state (60 harmonics, ideal switch with 1 ns edges, R$_{sw}$ = 25 mΩ). "
              "Circles: the five frequencies swept in LTspice (white = no input element, aqua = with L$_{in}\\parallel$R$_p$). "
              "The model has DC line resistance only (no skin effect), so real resonance peaks will be lower and wider.")
    save(fig, "03_ripple_vs_switching_frequency.png")


# ---------------------------------------------------------------- 4 waveforms at the 5 fs
def fig_waveforms():
    fig, axs = plt.subplots(len(FS5), 1, figsize=(10, 13), gridspec_kw={"hspace": 0.62})
    for ax, fs in zip(axs, FS5):
        T = 1 / fs
        t = np.linspace(0, 3 * T, 3000)
        il, _ = an.steady_wave(t, fs, model="lumped")
        i0, _ = an.steady_wave(t, fs)
        ifl, _ = an.steady_wave(t, fs, filt=True)
        for k in range(3):
            ax.axvspan(k, k + 0.512, color="#f0f0ec", zorder=0)
        ax.plot(t / T, i0, color=BLUE, lw=1.4, label="line")
        ax.plot(t / T, il, color=ORANGE, lw=1.6, label="lumped")
        ax.plot(t / T, ifl, color=AQUA, lw=1.8, label="line + L$_{in}\\parallel$R$_p$")
        ax.set_xlim(0, 3)
        ax.set_ylabel("$i(0,t)$ in A")
        lab = {7.96e6: "  (paper's choice)", 8.49e6: "  (= f$_1$, first antiresonance)"}.get(fs, "")
        ax.set_title(f"f$_s$ = {fs/1e6:g} MHz{lab}     RMS ripple:  line {i0.std()*1e3:,.0f} mA  ·  "
                     f"lumped {il.std()*1e3:,.0f} mA  ·  line + L$_{{in}}\\parallel$R$_p$ {ifl.std()*1e3:,.0f} mA",
                     fontsize=10.5)
    axs[0].legend(loc="upper right", ncol=3, bbox_to_anchor=(1, 1.42))
    axs[-1].set_xlabel("time in PWM periods  t / T   (shaded = switch on)")
    fig.subplots_adjust(bottom=0.06)
    fig.suptitle("Steady-state input current at the five swept switching frequencies (D = 0.512)",
                 x=0.0, ha="left", fontsize=13, fontweight="bold", y=0.945)
    note(fig, "y-axes differ per row. At 8.49 MHz the fundamental sits on the line antiresonance, but the 2nd "
              f"harmonic (16.98 MHz) sits on the series resonance; D = 0.512 ≠ 0.5 leaves enough 2nd harmonic to "
              "drive a large 17 MHz current. The L$_{in}\\parallel$R$_p$ input element removes it.")
    save(fig, "04_input_current_waveforms_vs_fs.png")


# ---------------------------------------------------------------- 5 discrete-L checks
def fig_lumped_checks():
    sp_rsw = sp.R_SW
    sp.R_SW = 0.0  # match the hand calculation (only R_L = 0.24 Ohm in series)
    L, C, RL = sp.L_LUMP, sp.C_LUMP, sp.R_LUMP

    def zeta(Rl):
        return (RL * C + L / Rl) / (2 * np.sqrt(L * C * (1 + RL / Rl)))

    fig, axs = plt.subplots(1, 2, figsize=(13, 5.6), gridspec_kw={"wspace": 0.22})
    ax = axs[0]
    loads = (10.0, 2.0, 0.547, 0.25)
    cols = RAMP[::-1]
    for Rl, c in zip(loads, cols):
        r = sp.run(model="lumped", D=0.512, fs=2e6, R=Rl, step_input=True, tstop=40e-6, maxstep=5e-9)
        vf = 12 * 0.512 * Rl / (Rl + RL)
        name = "R$_{crit}$ = 0.547 Ω" if Rl == 0.547 else f"R = {Rl:g} Ω"
        ax.plot(r["t"] * us, r["v"] / vf, color=c, lw=1.8,
                label=f"{name}:  ζ = {zeta(Rl):.2f},  overshoot {max(0, r['v'].max()/vf-1)*100:.1f} %")
    ax.axhline(1, color=MUTED, lw=1, ls="--")
    ax.set(xlim=(0, 40), xlabel="time t in µs", ylabel="$v_{out}/V_{out,final}$",
           title="(a) Damping vs. load: step d = 0 → 0.512, averaged model")
    ax.legend(loc="lower right", fontsize=9)

    ax = axs[1]
    T = 0.5e-6
    for Rl, c in zip((5.0, 10.0, 11.85, 20.0), RAMP):
        r = sp.run(model="lumped", D=0.512, fs=2e6, R=Rl, tstop=100e-6, maxstep=0.5e-9)
        sel = r["t"] >= r["t"][-1] - 2 * T
        tt = (r["t"][sel] - r["t"][sel][0]) / T
        imin = r["i"][sel].min()
        name = "R$_{crit}$ = 11.85 Ω" if Rl == 11.85 else f"R = {Rl:g} Ω"
        ax.plot(tt, r["i"][sel], color=c, lw=1.8, label=f"{name}:  i$_{{L,min}}$ = {imin*1e3:+.0f} mA")
    ax.axhline(0, color=INK2, lw=1.2)
    for k in range(2):
        ax.axvspan(k, k + 0.512, color="#f0f0ec", zorder=0)
    ax.set(xlim=(0, 2), xlabel="time in PWM periods t / T  (shaded = switch on)",
           ylabel="inductor current $i_L$ in A",
           title="(b) CCM/DCM boundary at f$_s$ = 2 MHz: $R_{crit} = 2Lf_s/(1-D)$")
    ax.legend(loc="upper right", fontsize=9)
    fig.suptitle("Discrete L–R$_L$ checks of the hand calculations (L = 1.446 µH, R$_L$ = 0.24 Ω, C = 1.0006 µF)",
                 x=0.0, ha="left", fontsize=13, fontweight="bold", y=1.01)
    note(fig, "ngspice, lumped model, ideal source (R$_{sw}$ = 0) so the numbers match the KVL hand results. "
              "(a) ζ = (R$_L$C + L/R) / (2√(LC(1+R$_L$/R))); ζ = 1 at R$_{crit}$ = 0.547 Ω. "
              "(b) The switch is synchronous, so above R$_{crit}$ the current goes negative instead of entering DCM.")
    save(fig, "05_discrete_L_damping_and_ccm.png")
    sp.R_SW = sp_rsw


# ---------------------------------------------------------------- 6 model check
def z_ladder(s, N):
    dl, dr, dc = sp.LP * sp.LEN / N, sp.RLP * sp.LEN / N, sp.CP * sp.LEN / N
    Z = 1 / (s * (sp.C_EXT + dc) + 1 / sp.R_LOAD) + s * dl + dr
    for _ in range(N - 1):
        Z = 1 / (s * dc + 1 / Z) + s * dl + dr
    return Z


def ripple_ladder(f, N, filt=False):
    exact = an.z_line
    an.z_line = lambda s: z_ladder(s, N)
    try:
        return an.ripple(f, filt=filt)
    finally:
        an.z_line = exact


def fig_model_check():
    fig, axs = plt.subplots(1, 2, figsize=(13, 5.4), sharey=True, gridspec_kw={"wspace": 0.06})
    out_rows = []
    x = np.arange(len(FS5))
    series = [("exact line, Eq. (20)", BLUE, "o", -0.24),
              ("200-section ladder (analytic)", ORANGE, "s", -0.08),
              ("ngspice transient, 200 sections", INK, "D", 0.08),
              ("25-section ladder (paper's Scilab N)", AQUA, "^", 0.24)]
    for ax, filt, ttl in ((axs[0], False, "(a) line only"), (axs[1], True, "(b) line + L$_{in}\\parallel$R$_p$")):
        vals = {n: [] for n, *_ in series}
        for fs in FS5:
            key = ("filt_fs%g" if filt else "line_fs%g") % (fs / 1e6)
            if not filt and fs == 2e6:
                key = "line_D0.512"
            m = sp.steady_metrics(R[key], fs, window=2e-6)
            ex, l200, l25 = an.ripple(fs, filt=filt)[0], ripple_ladder(fs, 200, filt)[0], ripple_ladder(fs, 25, filt)[0]
            vals[series[0][0]].append(ex)
            vals[series[1][0]].append(l200)
            vals[series[2][0]].append(m["Irms_ripple"])
            vals[series[3][0]].append(l25)
            out_rows.append((filt, fs, m["Irms_ripple"], l200, ex, l25))
        for i in x:
            ys = [vals[n][i] * 1e3 for n, *_ in series]
            ax.plot([i - 0.3, i + 0.3], [min(ys), max(ys)], alpha=0)
            ax.axvspan(i - 0.36, i + 0.36, color="#f3f3f0", zorder=0)
        for n, c, mk, dx in series:
            face = "none" if c == INK else c
            ax.semilogy(x + dx, np.array(vals[n]) * 1e3, mk, ms=9, mfc=face, mec=c if c != INK else INK,
                        mew=1.6, label=n, ls="none", zorder=4)
        for i in x:
            ax.text(i, 1.01, f"{vals[series[0][0]][i]*1e3:,.0f} mA", transform=ax.get_xaxis_transform(),
                    ha="center", fontsize=8.5, color=INK2)
        ax.set_xticks(x, [f"{f/1e6:g} MHz" for f in FS5])
        ax.set(xlim=(-0.5, len(FS5) - 0.5), ylim=(10, 5000), xlabel="switching frequency f$_s$")
        ax.set_title(ttl, pad=18)
        ax.grid(axis="x", visible=False)
    axs[0].set_ylabel("RMS ripple of line input current in mA (log)")
    axs[0].legend(loc="lower left", fontsize=9)
    fig.suptitle("Model check at the five swept frequencies: SPICE matches its ladder model; the ladder approaches "
                 "the exact line as sections increase", x=0.0, ha="left", fontsize=12.5, fontweight="bold", y=1.03)
    note(fig, "Numbers on top = exact line. Without the input element the ripple sits near sharp line resonances, so "
              "a coarse ladder (25 sections) can be off by 3× (7.96 and 8.49 MHz) and even SPICE with 200 sections "
              "is ~30 % low at 8.49 MHz. With L$_{in}\\parallel$R$_p$ the resonances are damped and every model agrees "
              "within a few mA.")
    save(fig, "06_model_check_spice_vs_exact.png")
    return out_rows


# ---------------------------------------------------------------- 7 summary table
def fig_summary(check_rows):
    rows = []
    for fs in FS5:
        rl = an.ripple(fs)[0] * 1e3
        ru = an.ripple(fs, model="lumped")[0] * 1e3
        rf = an.ripple(fs, filt=True)[0] * 1e3
        sp_l = [r for r in check_rows if not r[0] and r[1] == fs][0][2] * 1e3
        sp_f = [r for r in check_rows if r[0] and r[1] == fs][0][2] * 1e3
        lcrit = (1 - 0.512) * 10 / (2 * fs) * 1e6
        rcrit = 2 * sp.L_LUMP * fs / (1 - 0.512)
        rows.append([f"{fs/1e6:g}", f"{1e9/fs:.1f}", f"{lcrit:.3f}", f"{rcrit:.1f}",
                     f"{ru:,.0f}", f"{rl:,.0f}", f"{sp_l:,.0f}", f"{rf:,.0f}", f"{sp_f:,.0f}",
                     f"{rf/rl*100:.0f} %"])
    fig = plt.figure(figsize=(13, 3.4))
    ax = fig.add_axes([0, 0.1, 1, 0.66])
    table(ax, ["f$_s$\nin MHz", "T\nin ns", "L$_{crit}$\nin µH", "R$_{crit,CCM}$\nin Ω",
               "lumped\nexact", "line\nexact", "line\nSPICE", "line + L$_{in}\\parallel$R$_p$\nexact",
               "line + L$_{in}\\parallel$R$_p$\nSPICE", "filtered /\nunfiltered"],
          rows, highlight=[3], fs=10)
    fig.text(0.0, 0.97, "Summary at D = 0.512, R = 10 Ω: CCM limits and RMS input-current ripple (mA)",
             fontsize=13, fontweight="bold")
    fig.text(0.0, 0.88, "Ripple columns are RMS ripple of the line input current i(0,t) in mA.  "
             "L = 1.446 µH, so every row is in CCM (L > L$_{crit}$, R < R$_{crit}$).", fontsize=9.5, color=INK2)
    fig.text(0.0, 0.02, "SPICE = ngspice transient, line as 200 RLC sections. Exact = periodic steady state of the paper's "
             "distributed model. Bold row = paper's closed-loop f$_s$.", fontsize=8.5, color=MUTED)
    save(fig, "07_summary_table.png")


if __name__ == "__main__":
    import os
    os.makedirs(OUT, exist_ok=True)
    fig_lumped_vs_line()
    fig_duty()
    fig_ripple_fs()
    fig_waveforms()
    fig_lumped_checks()
    rows = fig_model_check()
    fig_summary(rows)
    for r in rows:
        print("filt" if r[0] else "line", r[1] / 1e6, "spice %.0f lad200 %.0f exact %.0f lad25 %.0f" %
              tuple(x * 1e3 for x in r[2:]))
