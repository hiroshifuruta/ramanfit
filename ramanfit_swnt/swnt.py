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


# ----------------------------- D + G band (joint) -----------------------------
def _gminus_record(out):
    """Build a G- peak record handling both Lorentzian and BWF lineshapes."""
    p = out.params
    if "gm_q" in p:  # BWF / Fano
        q = float(p["gm_q"].value)
        return {"label": "G-", "center": float(p["gm_center"].value),
                "center_stderr": (None if p["gm_center"].stderr is None
                                  else float(p["gm_center"].stderr)),
                "sigma": float(p["gm_sigma"].value),
                "amplitude": float(p["gm_amplitude"].value),
                # BWF peak height at resonance (a*(1 + 1/q)^2 approximation -> use max)
                "height": float(p["gm_amplitude"].value * (1.0 + 1.0 / q) ** 2)
                if q != 0 else float(p["gm_amplitude"].value),
                "q": q, "lineshape": "bwf", "at_bound": []}, (q != 0 and abs(1.0 / q) > 0.1)
    rec = peak_record(out, "gm_", "G-")
    rec["lineshape"] = "lorentzian"
    return rec, False


def fit_dg(x, y, region=(1000.0, 1700.0), gminus_lineshape="lorentzian"):
    """Jointly fit the whole D-G region with four Lorentzians + linear background:

    * **D**   (~1340) — defect band
    * **D''** (~1500-1535, broad) — disorder band that fills the D-to-G valley
    * **G-**  (~1567, Lorentzian or BWF) and **G+** (~1592)

    Fitting the region as a whole (default 1000-1700 cm^-1) avoids the railed,
    uncertainty-free fits produced by fitting D and G in separate narrow windows.
    ``center(G+) > center(G-)`` is enforced via non-overlapping bounds.

    Returns a dict with ``D``, ``Dpp``, ``G_plus``, ``G_minus`` records, the
    ``splitting`` and ``metallic`` verdict, ``r_squared`` and the lmfit result.
    """
    xg, yg = slice_region(x, y, *region)
    bg = LinearModel(prefix="lin_")
    pars = bg.guess(yg, x=xg)
    ymax = float(yg.max())

    d = LorentzianModel(prefix="d_")            # D band
    pars.update(d.make_params())
    pars["d_center"].set(value=1340, min=1310, max=1370)
    pars["d_sigma"].set(value=25, min=5, max=90)
    pars["d_amplitude"].set(value=max(1.0, ymax), min=1.0)

    dpp = LorentzianModel(prefix="dpp_")        # D'' broad disorder band
    pars.update(dpp.make_params())
    pars["dpp_center"].set(value=1510, min=1450, max=1560)
    pars["dpp_sigma"].set(value=70, min=20, max=150)
    pars["dpp_amplitude"].set(value=max(1.0, ymax * 2), min=1.0)

    gp = LorentzianModel(prefix="gp_")          # G+ : sharp, ~1592
    pars.update(gp.make_params())
    pars["gp_center"].set(value=1592, min=1585, max=1605)
    pars["gp_sigma"].set(value=10, min=3, max=30)
    pars["gp_amplitude"].set(value=max(1.0, ymax * 10), min=1.0)

    model = bg + d + dpp + gp
    if gminus_lineshape == "bwf":
        gm = BreitWignerModel(prefix="gm_")     # metallic Fano line
        pars.update(gm.make_params())
        pars["gm_center"].set(value=1565, min=1545, max=1584)
        pars["gm_sigma"].set(value=25, min=5, max=80)
        pars["gm_amplitude"].set(value=max(1.0, ymax), min=0.1)
        pars["gm_q"].set(value=-0.2)
    else:
        gm = LorentzianModel(prefix="gm_")      # semiconducting: symmetric
        pars.update(gm.make_params())
        pars["gm_center"].set(value=1567, min=1545, max=1584)
        pars["gm_sigma"].set(value=20, min=4, max=60)
        pars["gm_amplitude"].set(value=max(1.0, ymax * 3), min=1.0)
    model = model + gm

    out = model.fit(yg, pars, x=xg)
    gp_rec = peak_record(out, "gp_", "G+")
    gm_rec, metallic = _gminus_record(out)
    return {
        "x": xg, "y": yg, "result": out, "r_squared": r_squared(out),
        "gminus_lineshape": gminus_lineshape,
        "D": peak_record(out, "d_", "D"),
        "Dpp": peak_record(out, "dpp_", "D''"),
        "G_plus": gp_rec, "G_minus": gm_rec,
        "splitting": float(gp_rec["center"] - gm_rec["center"]),
        "metallic": bool(metallic),
    }


# ----------------------------- 2D -----------------------------
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


def fit_2d_band(x, y, region=(2500.0, 2800.0)):
    rec, xs, ys = _single_lorentzian(x, y, region, 2650, cwin=60, sigma0=40, sigma_max=150)
    return {"peak": rec, "x": xs, "y": ys}
