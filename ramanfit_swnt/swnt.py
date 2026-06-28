"""SWNT-specific band fits: RBM (-> diameter), G+/G- (-> metallic/semiconducting),
D and 2D (-> defect / quality metrics)."""
import numpy as np
from scipy.signal import find_peaks
from lmfit.models import LinearModel, LorentzianModel, BreitWignerModel

from .peaks import slice_region, fit_lorentzians, peak_record, r_squared


# ----------------------------- RBM -----------------------------
def rbm_to_diameter(omega, relation="248/w"):
    """Convert an RBM frequency (cm^-1) to tube diameter (nm)."""
    if relation == "248/w":
        return 248.0 / omega
    if relation == "araujo":
        # omega = 217.8/d + 15.7  ->  d = 217.8 / (omega - 15.7)
        return 217.8 / (omega - 15.7)
    raise ValueError(f"Unknown RBM relation: {relation!r}")


def seed_rbm_centers(x, y, min_prominence_frac=0.04):
    """Detect candidate RBM peak positions via prominence-based peak finding."""
    if len(x) < 5:
        return []
    baseline = np.percentile(y, 5)
    prom = max(20.0, min_prominence_frac * (y.max() - baseline))
    idx, _ = find_peaks(y, prominence=prom, distance=4)
    return [float(x[i]) for i in idx]


def fit_rbm(x, y, region=(100.0, 350.0), relation="248/w", min_prominence_frac=0.04):
    """Fit the RBM region with one Lorentzian per detected peak."""
    xr, yr = slice_region(x, y, *region)
    centers = seed_rbm_centers(xr, yr, min_prominence_frac)
    if not centers:
        return {"peaks": [], "r_squared": None, "result": None, "x": xr, "y": yr}
    seeds = [{"center": c, "prefix": f"r{i}_", "label": f"RBM{i+1}",
              "sigma": 5.0, "sigma_min": 1.5, "sigma_max": 15.0,
              "center_min": c - 8, "center_max": c + 8} for i, c in enumerate(centers)]
    out, records = fit_lorentzians(xr, yr, seeds)
    for rec in records:
        rec["diameter_nm"] = float(rbm_to_diameter(rec["center"], relation))
    return {"peaks": records, "r_squared": r_squared(out), "result": out,
            "x": xr, "y": yr}


# ----------------------------- G band -----------------------------
def fit_gband(x, y, region=(1500.0, 1650.0), gminus_lineshape="lorentzian"):
    """Fit the G band as G+ (Lorentzian) plus G- (Lorentzian or BWF).

    ``center(G+) > center(G-)`` is enforced via non-overlapping bounds to avoid
    the two components swapping roles.  Returns a dict with both peaks and a
    ``metallic`` verdict (BWF asymmetry / G- downshift heuristic).
    """
    xg, yg = slice_region(x, y, *region)
    bg = LinearModel(prefix="lin_")
    pars = bg.guess(yg, x=xg)

    gp = LorentzianModel(prefix="gp_")          # G+ : sharp, ~1590
    pars.update(gp.make_params())
    pars["gp_center"].set(value=1591, min=1585, max=1605)
    pars["gp_sigma"].set(value=10, min=3, max=30)
    pars["gp_amplitude"].set(value=max(1.0, yg.max() * 10), min=1.0)

    info = {"x": xg, "y": yg, "gminus_lineshape": gminus_lineshape}

    if gminus_lineshape == "bwf":
        gm = BreitWignerModel(prefix="gm_")     # metallic Fano line
        pars.update(gm.make_params())
        pars["gm_center"].set(value=1560, min=1530, max=1584)
        pars["gm_sigma"].set(value=30, min=5, max=90)
        pars["gm_amplitude"].set(value=max(1.0, yg.max()), min=0.1)
        pars["gm_q"].set(value=-0.2)            # 1/q < 0 -> downshifted Fano tail
        out = (bg + gp + gm).fit(yg, pars, x=xg)
        q = float(out.params["gm_q"].value)
        gm_center = float(out.params["gm_center"].value)
        gp_rec = peak_record(out, "gp_", "G+")
        gm_rec = {"label": "G-", "center": gm_center,
                  "center_stderr": (None if out.params["gm_center"].stderr is None
                                    else float(out.params["gm_center"].stderr)),
                  "sigma": float(out.params["gm_sigma"].value),
                  "amplitude": float(out.params["gm_amplitude"].value),
                  "q": q, "lineshape": "bwf", "at_bound": []}
        # metallic signature: strong Fano asymmetry (small |q|) or large downshift
        info["metallic"] = bool(abs(1.0 / q) > 0.1) if q != 0 else False
        info["bwf_q"] = q
    else:
        gm = LorentzianModel(prefix="gm_")      # semiconducting: symmetric
        pars.update(gm.make_params())
        pars["gm_center"].set(value=1568, min=1540, max=1584)
        pars["gm_sigma"].set(value=15, min=4, max=50)
        pars["gm_amplitude"].set(value=max(1.0, yg.max() * 3), min=1.0)
        out = (bg + gp + gm).fit(yg, pars, x=xg)
        gp_rec = peak_record(out, "gp_", "G+")
        gm_rec = peak_record(out, "gm_", "G-")
        gm_rec["lineshape"] = "lorentzian"
        info["metallic"] = False

    info["G_plus"] = gp_rec
    info["G_minus"] = gm_rec
    info["splitting"] = float(gp_rec["center"] - gm_rec["center"])
    info["r_squared"] = r_squared(out)
    info["result"] = out
    return info


# ----------------------------- D / 2D -----------------------------
def _single_lorentzian(x, y, region, center0, cwin=25, sigma0=20, sigma_max=80):
    xs, ys = slice_region(x, y, *region)
    if len(xs) < 8:
        return None, xs, ys
    seeds = [{"center": center0, "prefix": "p_", "sigma": sigma0,
              "sigma_max": sigma_max, "center_min": center0 - cwin,
              "center_max": center0 + cwin}]
    out, records = fit_lorentzians(xs, ys, seeds)
    rec = records[0]
    rec["r_squared"] = r_squared(out)
    return rec, xs, ys


def fit_d_band(x, y, region=(1250.0, 1450.0)):
    rec, xs, ys = _single_lorentzian(x, y, region, 1340, cwin=30)
    return {"peak": rec, "x": xs, "y": ys}


def fit_2d_band(x, y, region=(2500.0, 2800.0)):
    rec, xs, ys = _single_lorentzian(x, y, region, 2650, cwin=60, sigma0=40, sigma_max=150)
    return {"peak": rec, "x": xs, "y": ys}
