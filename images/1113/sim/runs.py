"""Run every ngspice transient used by the 11/13 figures and cache them in runs.pkl."""

import pickle
import sys
from multiprocessing import Pool

import numpy as np

import spice as sp

sp.N_SEC = 200  # ladder sections for the line in SPICE
LINE_STEP = 0.2e-9

JOBS = {}
# A + B: lumped vs line, duty sweep (fs = 2 MHz)
JOBS["lumped_D0.512"] = dict(model="lumped", D=0.512, fs=2e6, maxstep=0.5e-9)
for D in (0.4, 0.5, 0.512, 0.6):
    JOBS[f"line_D{D}"] = dict(model="line", D=D, fs=2e6, maxstep=LINE_STEP)
# validation points: fs sweep with and without the L_in || R_p input element
for fs in (4e6, 6e6, 7.96e6, 8.49e6):
    JOBS[f"line_fs{fs/1e6:g}"] = dict(model="line", D=0.512, fs=fs, maxstep=LINE_STEP)
for fs in (2e6, 4e6, 6e6, 7.96e6, 8.49e6):
    JOBS[f"filt_fs{fs/1e6:g}"] = dict(model="line", D=0.512, fs=fs, filt=True, maxstep=LINE_STEP)


def work(item):
    name, kw = item
    res = sp.run(**kw)
    # keep the full start-up for a few cases, otherwise only the last 4 us
    if not (name.startswith(("lumped", "line_D", "step"))):
        keep = res["t"] >= res["t"][-1] - 4e-6
        res = {k: v[keep] for k, v in res.items()}
    print("done", name, flush=True)
    return name, kw, res


if __name__ == "__main__":
    with Pool(2) as p:
        out = p.map(work, list(JOBS.items()), chunksize=1)
    with open("runs.pkl", "wb") as f:
        pickle.dump({n: {"kw": kw, **{k: np.asarray(v) for k, v in r.items()}}
                     for n, kw, r in out}, f)
    print("saved runs.pkl", file=sys.stderr)
