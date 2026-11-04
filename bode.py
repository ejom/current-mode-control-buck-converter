"""Bode plot of the lumped (standard) and distributed buck converter.

Reproduces the "standard converter" and "distributed converter" curves of
Figure 10 in Röbenack & Gerbet, "Current-Mode Control of a Distributed Buck
Converter with a Lossy Transmission Line", Electronics 2024, 13, 3565.
Also plots two extensions with a discrete element at the line input
(derivations in transfer_functions.md).

Run with:  uv run bode.py
"""

import matplotlib.pyplot as plt
import numpy as np

# --- Parameters (paper Section 3.1; fork-buck_transmission_line/src/context.sce)
E = 12.0  # supply voltage [V]
R = 10.0  # load resistance [Ohm]
Cext = 1e-6  # external capacitor [F]

# primary line constants (per metre), RG 58 C/U coax
Lp = 241e-9  # [H/m]
Cp = 100e-12  # [F/m]
RLp = 40e-3  # [Ohm/m]
GCp = 0.2e-12  # [S/m]
l = 6.0  # line length [m]

# lumped-equivalent parameters, Eq. (22)
L = l * Lp
C = l * Cp + Cext
RL = l * RLp
GC = l * GCp

# discrete element at the line input (transfer_functions.md, Sections 3-4).
# Values chosen in transfer_functions.md Section 7 (see input_filter_design.py):
# Rp ~ Z0 = sqrt(L'/C') = 49 Ohm back-terminates the line at high frequency.
Lin = 0.5e-6  # series input inductor [H]
Rp = 47.0  # resistor in parallel with Lin [Ohm]


def P_lumped(s):
    """Duty ratio -> inductor current of the standard converter, Eqs. (4)/(32).

    From the state-space model (23)-(24):
        P(s) = E (sC + GC + 1/R) / ((sL + RL)(sC + GC + 1/R) + 1)
    """
    Y = s * C + GC + 1.0 / R
    return E * Y / ((s * L + RL) * Y + 1.0)


def _line_num_den(s):
    """Numerator N and denominator Delta of Eq. (20), so that P = E N / Delta
    and the line input impedance is Z_line = Delta / N.

        N     = (sC'+G')R sinh(gl) + g(1+sRCext) cosh(gl)
        Delta = (sL'+R')(1+sRCext) sinh(gl) + gR cosh(gl)
    with g = sqrt((sL'+R')(sC'+G')), Eq. (12). P is even in g, so the branch
    of the square root does not matter.
    """
    zL = s * Lp + RLp  # series impedance per length
    yC = s * Cp + GCp  # shunt admittance per length
    g = np.sqrt(zL * yC)
    sh, ch = np.sinh(g * l), np.cosh(g * l)
    num = yC * R * sh + g * (1 + s * R * Cext) * ch
    den = zL * (1 + s * R * Cext) * sh + g * R * ch
    return num, den


def P_series_input(s, Zin):
    """Distributed converter with impedance Zin in series at the line input:
    P = E / (Zin + Z_line) = E N / (Zin N + Delta)."""
    num, den = _line_num_den(s)
    return E * num / (Zin * num + den)


def Z_L_par_R(s, L_in, R_p):
    """Impedance of L_in in parallel with R_p."""
    return s * L_in * R_p / (R_p + s * L_in)


def P_distributed(s):
    """Duty ratio -> line input current of the distributed converter, Eq. (20)."""
    return P_series_input(s, 0.0)


def P_distributed_L(s, L_in=Lin):
    """Distributed converter with a discrete inductor L_in at the line input."""
    return P_series_input(s, s * L_in)


def P_distributed_LR(s, L_in=Lin, R_p=Rp):
    """Distributed converter with L_in || R_p in series at the line input.

    Returns the total switch current (= line input current); the current in
    the L_in branch alone is this times R_p / (R_p + s L_in).
    """
    return P_series_input(s, Z_L_par_R(s, L_in, R_p))


def main():
    w = np.logspace(3, 9, 100_000)  # angular frequency [rad/s], as in bode1.sce
    s = 1j * w

    fig = bode_figure(w, s, logx=True)
    fig.axes[-1].set_xlabel(r"Angular frequency $\omega$ in rad/s")
    fig.tight_layout()
    fig.savefig("bode.png", dpi=150)
    print("saved bode.png")

    # zoom on the paper's PWM frequency range, f_PWM in 2..10 MHz (Section 3.4),
    # which contains the first antiresonance f1 ~ 8.49 MHz
    f = np.linspace(2e6, 10e6, 20_000)
    fig = bode_figure(f / 1e6, 2j * np.pi * f, logx=False)
    fig.axes[-1].set_xlabel("Frequency $f$ in MHz")
    fig.axes[0].set_xlim(2, 10)
    fig.tight_layout()
    fig.savefig("bode_zoom.png", dpi=150)
    print("saved bode_zoom.png")
    plt.show()


def bode_figure(x, s, logx):
    """Magnitude/phase figure of all four plants evaluated at s, plotted vs x."""
    curves = {
        "standard converter": P_lumped(s),
        "distributed converter": P_distributed(s),
        rf"distributed + $L_{{in}}$ = {Lin * 1e6:g} µH": P_distributed_L(s),
        rf"distributed + $L_{{in}} \parallel R_p$ ({Rp:g} Ω)": P_distributed_LR(s),
    }

    fig, (ax_mag, ax_ph) = plt.subplots(2, 1, sharex=True, figsize=(9, 9))
    # dashed standard curve drawn on top so it stays visible where they overlap
    styles = ["--", "-", "-", "-"]
    zorders = [5, 2, 3, 4]
    for (label, P), style, z in zip(curves.items(), styles, zorders):
        kw = dict(lw=2, label=label, zorder=z)
        ax_mag.plot(x, 20 * np.log10(np.abs(P)), style, **kw)
        ax_ph.plot(x, np.degrees(np.angle(P)), style, **kw)

    ax_mag.set_ylabel("Amplitude response in dB")
    ax_ph.set_ylabel("Phase in deg")
    for ax in (ax_mag, ax_ph):
        if logx:
            ax.set_xscale("log")
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(loc="lower left")
    return fig


if __name__ == "__main__":
    main()
