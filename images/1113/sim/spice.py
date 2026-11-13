"""ngspice runner for the open-loop buck converter of Röbenack & Gerbet (2024).

Two plant models feed the same R || C_ext load:
  * "line"   : 6 m lossy line, LTRA model (same .model line as the LTspice schematics)
  * "lumped" : discrete L = l*L', R_L = l*R_L', C = l*C' + C_ext   (paper Eq. 22)
The half bridge is an ideal synchronous switch: a 0..E PWM source with 1 ns edges
and R_sw = 25 mOhm (IRLML6244 R_DS(on)), so it matches the E*d(t) model of the paper.
Optional input element: L_in || R_p between the switch node and the line input.
"""

import os
import subprocess
import tempfile

import numpy as np

E = 12.0
R_LOAD = 10.0
C_EXT = 1e-6
LP, CP, RLP, LEN = 241e-9, 100e-12, 40e-3, 6.0
L_LUMP, R_LUMP, C_LUMP = LEN * LP, LEN * RLP, LEN * CP + C_EXT
R_SW = 0.025
T_EDGE = 1e-9
L_IN, R_P = 0.5e-6, 47.0
N_SEC = 60  # ladder sections for the 6 m line


def netlist(model="line", D=0.512, fs=2e6, R=R_LOAD, filt=False, tstop=60e-6,
            maxstep=None, step_input=False):
    T = 1.0 / fs
    if maxstep is None:
        maxstep = min(0.5e-9, T / 100)
    lines = [f"* buck {model} D={D} fs={fs} R={R} filt={filt}"]
    if step_input:  # averaged model: d(t) = D step at t = 0
        lines.append(f"Vsw sw 0 PWL(0 0 {T_EDGE} {E * D})")
    else:
        ton = D * T - T_EDGE  # trapezoid area = D*T*E
        lines.append(f"Vsw sw 0 PULSE(0 {E} 0 {T_EDGE} {T_EDGE} {ton:.6e} {T:.6e})")
    lines.append(f"Rsw sw a {R_SW}")
    lines.append("Vsense a b 0")
    if filt:
        lines += [f"Lin b c {L_IN}", f"Rp b c {R_P}"]
        nin = "c"
    else:
        nin = "b"
    if model == "ltra":
        lines += [f"O1 {nin} 0 out 0 LOSSY",
                  f".model LOSSY LTRA(L={LP} C={CP} R={RLP} G=0 LEN={LEN})"]
    elif model == "line":  # N-section RLC ladder, as the paper's context.sce
        dl, dr, dc = LP * LEN / N_SEC, RLP * LEN / N_SEC, CP * LEN / N_SEC
        prev = nin
        for k in range(1, N_SEC + 1):
            nxt = "out" if k == N_SEC else f"n{k}"
            lines += [f"Ls{k} {prev} r{k} {dl:.6e}", f"Rs{k} r{k} {nxt} {dr:.6e}"]
            if k < N_SEC:
                lines.append(f"Cs{k} {nxt} 0 {dc:.6e}")
            prev = nxt
    else:
        lines += [f"L1 {nin} m {L_LUMP}", f"RL m out {R_LUMP}"]
    cload = {"ltra": C_EXT, "line": C_EXT + CP * LEN / N_SEC}.get(model, C_LUMP)
    lines += [f"C1 out 0 {cload}", f"R1 out 0 {R}"]
    lines.append(".save v(out) i(vsense)")
    lines.append(f".tran {maxstep:.4e} {tstop:.4e} 0 {maxstep:.4e}")
    return "\n".join(lines)


def run(**kw):
    """Run one transient; returns dict with t, v (output), i (switch/line input current)."""
    with tempfile.TemporaryDirectory() as d:
        cir, dat = os.path.join(d, "c.cir"), os.path.join(d, "c.dat")
        ctl = (f"\n.control\nset filetype=ascii\nrun\nwrdata {dat} v(out) i(vsense)\n"
               ".endc\n.end\n")
        with open(cir, "w") as f:
            f.write(netlist(**kw) + ctl)
        p = subprocess.run(["ngspice", "-b", cir], capture_output=True, text=True)
        if not os.path.exists(dat):
            raise RuntimeError(p.stdout[-2000:] + p.stderr[-2000:])
        a = np.loadtxt(dat)
    return {"t": a[:, 0], "v": a[:, 1], "i": a[:, 3]}


def steady_metrics(res, fs, n_periods=None, window=3e-6):
    """Metrics over the last whole number of PWM periods (uniformly resampled)."""
    T = 1.0 / fs
    n = n_periods or max(1, int(round(window / T)))
    t_end = res["t"][-1]
    tt = np.linspace(t_end - n * T, t_end, n * 400, endpoint=False)
    v = np.interp(tt, res["t"], res["v"])
    i = np.interp(tt, res["t"], res["i"])
    return {
        "Vavg": v.mean(), "Vpp": v.max() - v.min(),
        "Iavg": i.mean(), "Irms_ripple": i.std(), "Ipp": i.max() - i.min(),
        "Imin": i.min(), "Imax": i.max(),
    }
