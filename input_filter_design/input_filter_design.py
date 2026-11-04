"""Light design study for the input element L_in || R_p of the distributed buck
converter (transfer_functions.md, Section 7).

Figure of merit: the RMS ripple of the line input current under PWM with
duty ratio D0 (paper Section 3.4), worst case over f_PWM in 2..10 MHz. The
ideal-switch model is linear in d(t), so the ripple follows exactly from the
Fourier series of d(t) pushed through P(jw):

    d(t) = D0 + sum_k a_k cos(k w_PWM t + phi_k),  a_k = 2|sin(pi k D0)| / (pi k)
    I_rms,ripple = sqrt( sum_k (a_k |P(j k w_PWM)|)^2 / 2 )

Run with:  uv run input_filter_design.py
"""

import matplotlib.pyplot as plt
import numpy as np

import bode as b

D0 = 0.512  # duty ratio for v0 = 6 V, paper Eq. (26)
K = 40  # number of PWM harmonics
Z0 = np.sqrt(b.Lp / b.Cp)  # line characteristic impedance (lossless), ~49 Ohm

k = np.arange(1, K + 1)
a_k = 2 * np.abs(np.sin(np.pi * k * D0)) / (np.pi * k)
f_pwm = np.linspace(2e6, 10e6, 801)
S = 2j * np.pi * np.outer(f_pwm, k)  # s at every harmonic of every f_PWM


def ripple(P_fn):
    """RMS ripple current [A] for each f_PWM, given P_fn(s)."""
    P = P_fn(S)
    return np.sqrt(np.sum((np.abs(P) * a_k) ** 2, axis=1) / 2)


def hf_peak_db(L_in, R_p):
    """Largest gain of the L_in || R_p plant above 10^7 rad/s (line resonances)."""
    w = np.logspace(7, 9.5, 20_000)
    return 20 * np.log10(np.abs(b.P_distributed_LR(1j * w, L_in, R_p)).max())


def main_resonance(L_in, R_p):
    """(frequency in kHz, gain in dB) of the main LC-like resonance."""
    w = np.logspace(4, 7, 4000)
    P = np.abs(b.P_distributed_LR(1j * w, L_in, R_p))
    i = P.argmax()
    return w[i] / (2 * np.pi * 1e3), 20 * np.log10(P[i])


def main():
    r_lumped = ripple(b.P_lumped).max()
    r_dist = ripple(b.P_distributed).max()
    print(f"Z0 = {Z0:.1f} Ohm")
    print(f"worst-case ripple, standard converter:            {r_lumped * 1e3:7.0f} mA")
    print(f"worst-case ripple, distributed (no input element): {r_dist * 1e3:7.0f} mA\n")

    Rps = np.logspace(0.5, 3, 101)
    Lins = [0.1e-6, 0.2e-6, 0.3e-6, 0.5e-6, 0.75e-6, 1e-6, 1.5e-6, 2e-6]
    sweeps = {}
    print(" L_in    best R_p  (within 10%)   ripple   HF peak   main res.  | R_p = Z0: ripple  HF peak")
    for L_in in Lins:
        r = np.array([ripple(lambda s: b.P_distributed_LR(s, L_in, R)).max() for R in Rps])
        sweeps[L_in] = r
        i = r.argmin()
        ok = Rps[r < 1.1 * r[i]]
        f0, g0 = main_resonance(L_in, Rps[i])
        rz = ripple(lambda s: b.P_distributed_LR(s, L_in, Z0)).max()
        print(
            f"{L_in * 1e6:4.2f} µH  {Rps[i]:6.0f} Ω  ({ok.min():3.0f}–{ok.max():4.0f} Ω)"
            f"  {r[i] * 1e3:5.0f} mA  {hf_peak_db(L_in, Rps[i]):5.1f} dB"
            f"  {f0:4.0f} kHz {g0:4.1f} dB"
            f"  |  {rz * 1e3:5.0f} mA  {hf_peak_db(L_in, Z0):5.1f} dB"
        )

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(13, 5))
    for L_in, r in sweeps.items():
        ax1.loglog(Rps, r * 1e3, lw=2, label=rf"$L_{{in}}$ = {L_in * 1e6:g} µH")
    ax1.axhline(r_lumped * 1e3, color="k", ls="--", label="standard converter")
    ax1.axvline(Z0, color="gray", ls=":", label=r"$Z_0=\sqrt{L'/C'}$")
    ax1.set_xlabel(r"$R_p$ in Ω")
    ax1.set_ylabel(r"worst-case RMS ripple over $f_{PWM}$ = 2–10 MHz in mA")
    ax1.set_title(rf"Ripple vs $R_p$ (no input element: {r_dist:.1f} A)")

    curves = {
        "standard converter": b.P_lumped,
        "distributed converter": b.P_distributed,
        rf"distributed + $L_{{in}}$ = {b.Lin * 1e6:g} µH": b.P_distributed_L,
        rf"distributed + $L_{{in}} \parallel R_p$ ({b.Rp:g} Ω)": b.P_distributed_LR,
    }
    for (label, P_fn), style in zip(curves.items(), ["--", "-", "-", "-"]):
        ax2.semilogy(f_pwm / 1e6, ripple(P_fn) * 1e3, style, lw=2, label=label)
    ax2.set_xlabel(r"$f_{PWM}$ in MHz")
    ax2.set_ylabel("RMS ripple of line input current in mA")
    ax2.set_title(rf"Ripple vs PWM frequency ($D_0$ = {D0})")
    for ax in (ax1, ax2):
        ax.grid(True, which="both", alpha=0.4)
        ax.legend(fontsize=8)
    fig.tight_layout()
    fig.savefig("input_filter_design.png", dpi=150)
    print("\nsaved input_filter_design.png")
    plt.show()


if __name__ == "__main__":
    main()
