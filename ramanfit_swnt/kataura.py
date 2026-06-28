"""Candidate (n,m) chirality assignment from RBM frequency + laser energy.

IMPORTANT — accuracy caveat
---------------------------
The optical transition energies ``E_ii`` used here come from a *first-order
zone-folding tight-binding model*, NOT from measured Kataura data:

    E_ii(p) = p * (2 * a_cc * gamma0) / d        (eV, d in nm)

with ``p`` the transition branch index (p = 1,2,4,5,7,8,... for semiconducting
tubes where mod(n-m,3)!=0, and p = 3,6,... for metallic tubes where
mod(n-m,3)==0).  Trigonal-warping and many-body (excitonic) corrections are
*not* included, so transition energies are approximate (typ. +/-0.1-0.2 eV).

Assignments are therefore "candidates" intended to guide interpretation, not
definitive (n,m) identifications.  A measured Kataura table can be supplied via
``data/kataura.csv`` (columns: n,m,diameter_nm,type,E_ii_eV) to override the
computed values.
"""
import csv
import os
import numpy as np

A_CC = 0.142       # nearest-neighbour C-C distance (nm)
GAMMA0 = 3.0       # tight-binding overlap integral (eV); tuned so E22_S(1nm)~1.7 eV
_SLOPE = 2 * A_CC * GAMMA0   # eV*nm, per unit branch index

_CSV = os.path.join(os.path.dirname(__file__), "data", "kataura.csv")


def diameter(n, m):
    """Geometric diameter (nm) of an (n,m) tube."""
    return (np.sqrt(3.0) * A_CC / np.pi) * np.sqrt(n * n + n * m + m * m)


def is_metallic(n, m):
    return (n - m) % 3 == 0


def transition_energies(n, m, max_branch=6):
    """Approximate E_ii (eV) for tube (n,m), as a list of (label, energy)."""
    d = diameter(n, m)
    if is_metallic(n, m):
        branches = [(3, "E11M"), (6, "E22M")]
    else:
        branches = [(1, "E11S"), (2, "E22S"), (4, "E33S"), (5, "E44S")]
    return [(label, _SLOPE * p / d) for p, label in branches if p <= max_branch]


def build_table(max_n=25):
    """Generate the full candidate table of physically allowed (n,m) tubes."""
    rows = []
    for n in range(1, max_n + 1):
        for m in range(0, n + 1):
            if n == 0 and m == 0:
                continue
            d = diameter(n, m)
            typ = "M" if is_metallic(n, m) else "S"
            for label, e in transition_energies(n, m):
                rows.append({"n": n, "m": m, "diameter_nm": round(float(d), 4),
                             "type": typ, "transition": label,
                             "E_ii_eV": round(float(e), 4)})
    return rows


def load_table(max_n=25):
    """Load the bundled CSV if present, else compute the table on the fly."""
    if os.path.exists(_CSV):
        with open(_CSV, newline="") as f:
            rows = []
            for r in csv.DictReader(f):
                rows.append({"n": int(r["n"]), "m": int(r["m"]),
                             "diameter_nm": float(r["diameter_nm"]),
                             "type": r["type"], "transition": r.get("transition", ""),
                             "E_ii_eV": float(r["E_ii_eV"])})
            return rows
    return build_table(max_n)


def assign_nm(omega_or_diameter, laser_eV, *, is_diameter=False,
              diameter_tol=0.06, energy_window=0.20, relation="248/w",
              table=None, max_n=25):
    """Return candidate (n,m) assignments for a single RBM peak.

    Each candidate matches the measured diameter within ``diameter_tol`` (nm)
    AND has an optical transition within ``energy_window`` (eV) of the laser.
    Candidates are sorted by combined diameter+energy mismatch (best first).
    """
    if is_diameter:
        d_meas = float(omega_or_diameter)
    else:
        from .swnt import rbm_to_diameter
        d_meas = float(rbm_to_diameter(omega_or_diameter, relation))

    rows = table if table is not None else load_table(max_n)
    cands = []
    for r in rows:
        dd = abs(r["diameter_nm"] - d_meas)
        if dd > diameter_tol:
            continue
        de = abs(r["E_ii_eV"] - laser_eV)
        if de > energy_window:
            continue
        score = dd / diameter_tol + de / energy_window
        cands.append({**r, "d_diff": round(dd, 4), "E_diff": round(de, 4),
                      "score": round(score, 4)})
    cands.sort(key=lambda c: c["score"])
    return cands
