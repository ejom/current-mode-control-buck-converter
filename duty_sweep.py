"""Input-current ripple of the four plants over PWM frequency AND duty ratio
(transfer_functions.md, Section 8).

input_filter_design.py fixes D0 = 0.512. Here D is swept as well, because the
duty ratio sets the harmonic weights a_k = 2|sin(pi k D)| / (pi k): a harmonic
that sits on a line resonance can be strong at one D and vanish at another
(e.g. all even harmonics vanish at D = 0.5). The ripple (= standard deviation of
the line input current in steady state) is computed exactly as in
input_filter_design.py:

    I_rms,ripple(f_PWM, D) = sqrt( sum_k (a_k(D) |P(j k 2 pi f_PWM)|)^2 / 2 )

Run with:  uv run duty_sweep.py
"""

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

import bode as b

K = 40  # number of PWM harmonics
D0 = 0.512  # paper's operating point, Eq. (26)

k = np.arange(1, K + 1)
f_pwm = np.linspace(2e6, 10e6, 1601)
D = np.linspace(0.02, 0.98, 193)
a2 = (2 * np.sin(np.pi * np.outer(D, k)) / (np.pi * k)) ** 2  # (D, k)


def ripple_map(P_fn):
    """RMS ripple [A] on the (D, f_PWM) grid."""
    P2 = np.abs(P_fn(2j * np.pi * np.outer(f_pwm, k))) ** 2  # (f, k)
    return np.sqrt(a2 @ P2.T / 2)  # (D, f)


def main():
    plants = {
        "standard converter": b.P_lumped,
        "distributed converter": b.P_distributed,
        rf"distributed + $L_{{in}}$ = {b.Lin * 1e6:g} µH": b.P_distributed_L,
        rf"distributed + $L_{{in}} \parallel R_p$ ({b.Rp:g} Ω)": b.P_distributed_LR,
    }
    maps = {name: ripple_map(fn) * 1e3 for name, fn in plants.items()}  # mA

    norm = LogNorm(vmin=10, vmax=max(m.max() for m in maps.values()))
    fig, axes = plt.subplots(2, 2, sharex=True, sharey=True, figsize=(12, 9))
    for ax, (name, r) in zip(axes.flat, maps.items()):
        mesh = ax.pcolormesh(f_pwm / 1e6, D, r, norm=norm, cmap="Blues", shading="auto")
        i, j = np.unravel_index(r.argmax(), r.shape)
        ax.plot(f_pwm[j] / 1e6, D[i], "o", mfc="none", mec="#d62728", mew=2, ms=10, clip_on=False)
        ax.axhline(D0, color="0.4", ls=":", lw=1)
        ax.set_title(
            f"{name}\nworst {r.max():,.0f} mA at "
            f"$f_{{PWM}}$ = {f_pwm[j] / 1e6:.2f} MHz, D = {D[i]:.2f}",
            fontsize=10,
        )
        print(
            f"{name:45s} worst {r.max():8.0f} mA at f = {f_pwm[j] / 1e6:6.3f} MHz,"
            f" D = {D[i]:.2f};  ripple at D0 worst {r[np.abs(D - D0).argmin()].max():6.0f} mA"
        )
    for ax in axes[1]:
        ax.set_xlabel(r"$f_{PWM}$ in MHz")
    for ax in axes[:, 0]:
        ax.set_ylabel("duty ratio D")
    fig.colorbar(mesh, ax=axes, label="RMS ripple of line input current in mA (log)")
    fig.suptitle(
        "Input-current ripple over PWM frequency and duty ratio "
        rf"(dotted: $D_0$ = {D0}; circle: worst case)"
    )
    fig.savefig("duty_sweep.png", dpi=150, bbox_inches="tight")
    print("saved duty_sweep.png")
    plt.show()


if __name__ == "__main__":
    main()
