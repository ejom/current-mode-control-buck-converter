"""Exact periodic steady state of the ideal-switch converter (frequency domain).

d(t) has Fourier weights a_k = 2|sin(pi k D)|/(pi k) (times a sinc for the 1 ns edges),
so the RMS ripple of the line input current is
    I_rms,ripple = sqrt( sum_k (E a_k |Y(j k w)|)^2 / 2 ),  Y = 1/(Z_in + Z_plant)
Z_plant is the input impedance of the line (Eq. 20 of the paper, via bode.py) or of the
lumped L-R_L-C network. This is the same method as input_filter_design.py / duty_sweep.py.
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))  # repo root, for bode.py
import bode as b  # noqa: E402
import spice as sp

K = 60


def z_line(s):
    num, den = b._line_num_den(s)
    return den / num


def z_lumped(s):
    Y = s * sp.C_LUMP + 1.0 / sp.R_LOAD
    return s * sp.L_LUMP + sp.R_LUMP + 1.0 / Y


def ripple(f, D=0.512, model="line", filt=False):
    f = np.atleast_1d(f)
    k = np.arange(1, K + 1)
    s = 2j * np.pi * np.outer(f, k)
    Zin = sp.R_SW + 0j
    if filt:
        Zin = Zin + s * sp.L_IN * sp.R_P / (sp.R_P + s * sp.L_IN)
    Zp = z_line(s) if model == "line" else z_lumped(s)
    a = 2 * np.abs(np.sin(np.pi * k * D)) / (np.pi * k)
    a = a * np.abs(np.sinc(np.outer(f, k) * sp.T_EDGE))
    I = sp.E * a * np.abs(1.0 / (Zin + Zp))
    return np.sqrt(np.sum(I**2, axis=1) / 2)


def vout_dc(D, R=sp.R_LOAD):
    return sp.E * D * R / (R + sp.R_LUMP + sp.R_SW)


def _plant(s, model, filt):
    Zin = sp.R_SW + 0j
    if filt:
        Zin = Zin + s * sp.L_IN * sp.R_P / (sp.R_P + s * sp.L_IN)
    if model == "line":
        zL = s * b.Lp + b.RLp
        yC = s * b.Cp + b.GCp
        g = np.sqrt(zL * yC)
        sh, ch = np.sinh(g * b.l), np.cosh(g * b.l)
        Z = 1.0 / (s * sp.C_EXT + 1.0 / sp.R_LOAD)
        m11, m12 = ch, yC / g * sh
        m21, m22 = zL / g * sh, ch
        Zl = (m21 + m22 * Z) / (m11 + m12 * Z)       # line input impedance
        Hv = Z / (m11 + m12 * Z)                     # v_out / i(0)
    else:
        Y = s * sp.C_LUMP + 1.0 / sp.R_LOAD
        Zl = s * sp.L_LUMP + sp.R_LUMP + 1.0 / Y
        Hv = 1.0 / Y
    Y0 = 1.0 / (Zin + Zl)
    return Y0, Hv


def steady_wave(t, fs, D=0.512, model="line", filt=False, K=None):
    """Exact periodic steady state i(0,t), v_out(t); pulse is on for 0 <= t mod T < D T."""
    K = K or int(min(4000, 2.0e9 / fs))
    k = np.arange(1, K + 1)
    w = 2 * np.pi * fs
    s = 1j * w * k
    Dk = (1 - np.exp(-2j * np.pi * k * D)) / (2j * np.pi * k)
    Dk = Dk * np.sinc(k * fs * sp.T_EDGE) * np.exp(-1j * w * k * sp.T_EDGE / 2)
    Y0, Hv = _plant(s, model, filt)
    Ik = sp.E * Dk * Y0
    Vk = Ik * Hv
    i_dc = sp.E * D / (sp.R_SW + sp.R_LUMP + sp.R_LOAD)
    v_dc = i_dc * sp.R_LOAD
    ph = np.exp(1j * np.outer(t, w * k))
    i = i_dc + 2 * np.real(ph @ Ik)
    v = v_dc + 2 * np.real(ph @ Vk)
    return i, v
