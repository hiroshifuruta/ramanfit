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


def fit_rbm(x, y, region=(180.0, 350.0), relation="248/w", min_prominence_frac=0.04,
            extra_centers=()):
    """Fit the RBM region with one Lorentzian per detected peak.

    A **constant** baseline is used (not linear): a notch filter blocks the
    Rayleigh line below ~200 cm^-1, so there is no genuine sloping background
    across the RBM window, and a linear baseline would spuriously tilt to chase
    the notch edge.

    ``extra_centers`` lets you seed Lorentzians at frequencies (cm^-1) the
    prominence-based detector misses -- e.g. a shoulder riding on a stronger
    neighbour.  They are merged with the auto-detected centers (anything within
    3 cm^-1 of an existing seed is treated as a duplicate and dropped).
    """
    xr, yr = slice_region(x, y, *region)
    centers = seed_rbm_centers(xr, yr, min_prominence_frac)
    for c in extra_centers:
        if region[0] <= c <= region[1] and all(abs(c - e) > 3.0 for e in centers):
            centers.append(float(c))
    centers.sort()
    if not centers:
        return {"peaks": [], "r_squared": None, "result": None, "x": xr, "y": yr}
    seeds = [{"center": c, "prefix": f"r{i}_", "label": f"RBM{i+1}",
              "sigma": 5.0, "sigma_min": 1.5, "sigma_max": 15.0,
              "center_min": c - 8, "center_max": c + 8} for i, c in enumerate(centers)]
    out, records = fit_lorentzians(xr, yr, seeds, background="constant")
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
    """Jointly fit the whole D-G region with five Lorentzians + linear background:

    * **D**   (~1340) — defect band
    * **D''** (~1500-1535, broad) — disorder band that fills the D-to-G valley
    * **G-**  (~1567, Lorentzian or BWF) and **G+** (~1592)
    * **D'**  (~1600-1620) — defect-activated band sitting just above G+

    Fitting the region as a whole (default 1000-1700 cm^-1) avoids the railed,
    uncertainty-free fits produced by fitting D and G in separate narrow windows.
    ``center(G+) > center(G-)`` is enforced via non-overlapping bounds.

    Without the D' band a single Lorentzian G+ cannot reproduce its upper
    shoulder, leaving a large oscillating residual across 1550-1630 cm^-1;
    adding D' roughly halves that residual and sharpens G+ to its true width.

    Returns a dict with ``D``, ``Dpp``, ``G_plus``, ``G_minus``, ``Dprime``
    records, the ``splitting`` and ``metallic`` verdict, ``r_squared`` and the
    lmfit result.
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

    dp = LorentzianModel(prefix="dp_")          # D' : shoulder just above G+
    pars.update(dp.make_params())
    pars["dp_center"].set(value=1610, min=1595, max=1630)
    pars["dp_sigma"].set(value=12, min=4, max=40)
    pars["dp_amplitude"].set(value=max(1.0, ymax), min=1.0)

    model = bg + d + dpp + gp + dp
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
        "Dprime": peak_record(out, "dp_", "D'"),
        "splitting": float(gp_rec["center"] - gm_rec["center"]),
        "metallic": bool(metallic),
    }


# ----------------------------- 2D -----------------------------
def _seed_2d_components(xs, ys, shoulder_offset=35.0, fit_gstar=True):
    """Build seed specs for the 2D region: a broad G* band plus the 2D triplet.

    The 2D band itself is a single tall peak with weaker shoulders, not several
    resolved maxima -- prominence-based peak finding either misses the shoulders
    or fragments the noisy summit.  So we anchor on the global maximum and add a
    shoulder seed ``shoulder_offset`` cm^-1 to either side.

    ``fit_gstar`` adds the **G\\*** (iTOLA) combination band near 2450 cm^-1 --
    a broad, weak feature physically distinct from the 2D overtone but living in
    the same window.  It gets its own (wider) sigma bounds and a center range
    fixed to ~2400-2490 so it cannot wander into the 2D triplet.

    Returns a list of seed dicts (the ``seeds`` format of ``fit_lorentzians``).
    """
    if len(xs) < 8:
        return []
    seeds = []
    lo, hi = float(xs.min()), float(xs.max())
    if fit_gstar and lo <= 2450 <= hi:
        seeds.append({"center": 2450.0, "label": "G*", "sigma": 60.0,
                      "sigma_min": 30.0, "sigma_max": 180.0,
                      "center_min": 2410.0, "center_max": 2490.0})
    # 2D triplet: main maximum (searched above the G* region) + a shoulder each side
    main_mask = xs >= 2520
    cmax = float(xs[main_mask][int(np.argmax(ys[main_mask]))]) if main_mask.any() \
        else float(xs[int(np.argmax(ys))])
    centers = [cmax]
    if cmax - shoulder_offset >= max(lo, 2520):
        centers.insert(0, cmax - shoulder_offset)
    if cmax + shoulder_offset <= hi:
        centers.append(cmax + shoulder_offset)
    for c in centers:
        seeds.append({"center": c, "label": "2D", "sigma": 30.0,
                      "sigma_min": 5.0, "sigma_max": 120.0,
                      "center_min": c - 40, "center_max": c + 40})
    return seeds


def fit_2d_band(x, y, region=(2380.0, 2820.0), shoulder_offset=35.0,
                fit_gstar=True, min_height_frac=0.02):
    """Fit the 2D region over a wide window as a multi-component profile.

    Two physically distinct families share this window and are fit jointly:

    * **G\\*** (~2450, iTOLA combination band) -- broad and weak, included when
      ``fit_gstar`` and the window covers 2450.
    * the **2D** overtone (~2655) -- a main peak with shoulders from the
      diameter distribution / different (n,m) species.

    After the joint fit, components weaker than ``min_height_frac`` of the
    tallest are dropped and the band refit, so the reported count reflects only
    real components.

    Returns ``peaks`` (components, by descending height; each has a ``label`` of
    ``"G*"`` or ``"2D"``), ``peak`` (the dominant 2D peak, kept for the
    I(2D)/I(G) metric), ``r_squared`` and the data.
    """
    xs, ys = slice_region(x, y, *region)
    seeds = _seed_2d_components(xs, ys, shoulder_offset, fit_gstar)
    if not seeds:
        return {"peaks": [], "peak": None, "r_squared": None, "x": xs, "y": ys}

    def _fit(seed_list):
        for i, s in enumerate(seed_list):
            s["prefix"] = f"t{i}_"
        out, records = fit_lorentzians(xs, ys, seed_list)
        for rec, s in zip(records, seed_list):
            rec["label"] = s["label"]
        return out, records

    out, records = _fit(seeds)
    hmax = max((r["height"] for r in records), default=0.0)
    kept = [s for s, r in zip(seeds, records)
            if r["height"] >= min_height_frac * hmax]
    if 0 < len(kept) < len(records):           # drop negligible components, refit
        out, records = _fit(kept)
    records.sort(key=lambda r: r["height"], reverse=True)
    # the I(2D)/I(G) metric must use the 2D overtone, not the G* band
    twod_peaks = [r for r in records if r["label"] == "2D"] or records
    return {"peaks": records, "peak": (twod_peaks[0] if twod_peaks else None),
            "r_squared": r_squared(out), "result": out, "x": xs, "y": ys}
