"""SWNT-specific band fits: RBM (-> diameter), G+/G- (-> metallic/semiconducting),
D and 2D (-> defect / quality metrics)."""
import numpy as np
from scipy.signal import find_peaks
from lmfit.models import ConstantModel, LorentzianModel, BreitWignerModel

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


def seed_rbm_centers(x, y, min_prominence_frac=0.03):
    """Detect candidate RBM peak positions via prominence-based peak finding.

    The signal is lightly smoothed first so noise spikes do not inflate the
    prominence of, or fragment, the real bands.  The prominence threshold scales
    with the band envelope (``min_prominence_frac`` of max-above-baseline) with a
    low absolute floor, so weak-signal spectra still yield their shoulder bands
    instead of collapsing to a few tall peaks.
    """
    if len(x) < 5:
        return []
    baseline = np.percentile(y, 5)
    ys = np.convolve(y, np.ones(5) / 5.0, mode="same") if len(y) >= 5 else y
    prom = max(8.0, min_prominence_frac * (y.max() - baseline))
    idx, _ = find_peaks(ys, prominence=prom, distance=3)
    return [float(x[i]) for i in idx]


def fit_rbm(x, y, region=(180.0, 350.0), relation="248/w", min_prominence_frac=0.03,
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


def fit_dg(x, y, region=(1000.0, 1800.0), gminus_lineshape="lorentzian"):
    """Jointly fit the whole D-G region with five Lorentzians + linear background:

    * **D**   (~1340) — defect band
    * **D''** (~1500-1535, broad) — disorder band that fills the D-to-G valley
    * **G-**  (~1567, Lorentzian or BWF) and **G+** (~1592)
    * **D'**  (~1600-1620) — defect-activated band sitting just above G+

    Fitting the region as a whole (default 1000-1800 cm^-1) avoids the railed,
    uncertainty-free fits produced by fitting D and G in separate narrow windows.
    ``center(G+) > center(G-)`` is enforced via non-overlapping bounds.

    Without the D' band a single Lorentzian G+ cannot reproduce its upper
    shoulder, leaving a large oscillating residual across 1550-1630 cm^-1;
    adding D' roughly halves that residual and sharpens G+ to its true width.

    The background is a flat **constant**: the D-G window's continuum is level
    across samples here, so a linear background only tilts to chase the strong
    G+ tail (pulling the right edge well below the data) without improving the
    fit -- the five bands already carry any genuine slope in their tails.

    Returns a dict with ``D``, ``Dpp``, ``G_plus``, ``G_minus``, ``Dprime``
    records, the ``splitting`` and ``metallic`` verdict, ``r_squared`` and the
    lmfit result.
    """
    xg, yg = slice_region(x, y, *region)
    bg = ConstantModel(prefix="lin_")
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
def _seed_2d_components(xs, ys, shoulder_offset=35.0, fit_gstar=True,
                        fit_2dprime=True):
    """Build seed specs for the full second-order region (matches a reference
    Igor multipeak fit of this sample): G*, the 2D triplet, two weak broad
    combination bands at ~2915 and ~3105, 2D', and a weak ~3510 band.

    The strong 2D band is a single tall peak with weaker shoulders, not several
    resolved maxima, so it is seeded as the maximum of the 2D window (~2520-2780)
    plus a shoulder ``shoulder_offset`` cm^-1 to each side.  The other bands sit
    at fixed centers (each added only if the window covers it):

    * **G\\*** (~2446, iTOLA combination band) -- ``fit_gstar``
    * **2D** triplet (~2615 / 2654 / 2685)
    * **D+D''** (~2915) and **2D''** (~3105) -- weak broad combination bands
    * **2D'** (~3185, overtone of D') -- ``fit_2dprime``
    * a weak band near ~3510

    Returns a list of seed dicts (the ``seeds`` format of ``fit_lorentzians``).
    """
    if len(xs) < 8:
        return []
    seeds = []
    lo, hi = float(xs.min()), float(xs.max())
    if fit_gstar and lo <= 2446 <= hi:
        seeds.append({"center": 2446.0, "label": "G*", "sigma": 55.0,
                      "sigma_min": 20.0, "sigma_max": 110.0,
                      "center_min": 2420.0, "center_max": 2470.0})
    # 2D triplet: main maximum (searched in the 2D window only) + a shoulder each side
    win = (xs >= 2520) & (xs <= 2780)
    cmax = float(xs[win][int(np.argmax(ys[win]))]) if win.any() \
        else float(xs[int(np.argmax(ys))])
    centers = [cmax]
    if cmax - shoulder_offset >= max(lo, 2520):
        centers.insert(0, cmax - shoulder_offset)
    if cmax + shoulder_offset <= min(hi, 2780):
        centers.append(cmax + shoulder_offset)
    for c in centers:
        seeds.append({"center": c, "label": "2D", "sigma": 30.0,
                      "sigma_min": 5.0, "sigma_max": 120.0,
                      "center_min": c - 40, "center_max": c + 40})
    # weak broad combination bands (D+D'' and 2D'')
    for center, lab, cwin in [(2915.0, "D+D''", 45.0), (3105.0, "2D''", 55.0)]:
        if lo <= center <= hi:
            seeds.append({"center": center, "label": lab, "sigma": 70.0,
                          "sigma_min": 20.0, "sigma_max": 130.0,
                          "center_min": center - cwin,
                          "center_max": center + cwin})
    if fit_2dprime and lo <= 3185 <= hi:
        seeds.append({"center": 3185.0, "label": "2D'", "sigma": 50.0,
                      "sigma_min": 8.0, "sigma_max": 90.0,
                      "center_min": 3150.0, "center_max": 3220.0})
    return seeds


def _valley_baseline(xs, ys):
    """Estimate a flat (constant) 2D-region baseline at the true continuum floor.

    With every band fit explicitly (the weak combination bands carry the
    inter-band intensity), the genuine baseline is the flat far plateau past the
    2D' band (~3400-3500 cm^-1), NOT the ~3050 dip -- that dip is propped up by
    the D+D'' / 2D'' band tails.  Use that plateau's level, falling back to a low
    percentile of the window if it is out of range.
    """
    sel = (xs >= 3400) & (xs <= 3500)
    level = float(np.median(ys[sel])) if sel.any() \
        else float(np.percentile(ys, 5))
    return np.full_like(xs, level)


def fit_2d_band(x, y, region=(2380.0, 3260.0), shoulder_offset=35.0,
                fit_gstar=True, fit_2dprime=True, min_height_frac=0.02):
    """Fit the 2D region over a wide window as a multi-component profile.

    Three physically distinct families share this window and are fit jointly:

    * **G\\*** (~2450, iTOLA combination band) -- broad and weak, included when
      ``fit_gstar`` and the window covers 2450.
    * the **2D** overtone (~2655) -- a main peak with shoulders from the
      diameter distribution / different (n,m) species.
    * **2D'** (~3180, overtone of the D' band) -- weak, included when
      ``fit_2dprime`` and the window reaches it.

    The window is deliberately wide (default 2380-3500) so the ~3050 cm^-1
    inter-band valley is included.  The baseline is a flat constant pinned to
    that valley level (see :func:`_valley_baseline`): fitting a constant/linear
    background jointly with the peaks pulls it below the true level because the
    peak tails steal area, so the level is set from the data first, subtracted,
    and the peaks fit on the flattened data with **no** further background.  The
    flat line is the reported ``baseline``.

    After the joint fit, components weaker than ``min_height_frac`` of the
    tallest are dropped and the band refit, so the reported count reflects only
    real components.

    Returns ``peaks`` (components, by descending height; each has a ``label`` of
    ``"G*"``, ``"2D"`` or ``"2D'"``), ``peak`` (the dominant 2D peak, kept for
    the I(2D)/I(G) metric), ``r_squared``, ``baseline`` and the data.
    """
    xs, ys = slice_region(x, y, *region)
    seeds = _seed_2d_components(xs, ys, shoulder_offset, fit_gstar, fit_2dprime)
    if not seeds:
        return {"peaks": [], "peak": None, "r_squared": None,
                "baseline": None, "x": xs, "y": ys}

    curve = _valley_baseline(xs, ys)
    yb = ys - curve

    def _fit(seed_list):
        for i, s in enumerate(seed_list):
            s["prefix"] = f"t{i}_"
        # no background -- the curved continuum is already removed
        out, records = fit_lorentzians(xs, yb, seed_list, background="none")
        for rec, s in zip(records, seed_list):
            rec["label"] = s["label"]
        return out, records

    out, records = _fit(seeds)
    hmax = max((r["height"] for r in records), default=0.0)
    # prune only negligible 2D *shoulders*; the named G* / 2D' bands are kept
    # even when weak because they are physically distinct features we sought.
    kept = [s for s, r in zip(seeds, records)
            if s["label"] != "2D" or r["height"] >= min_height_frac * hmax]
    if 0 < len(kept) < len(records):           # drop negligible shoulders, refit
        out, records = _fit(kept)
    records.sort(key=lambda r: r["height"], reverse=True)
    # baseline is the valley-anchored continuum itself (no offset)
    baseline = curve
    # R^2 against the *original* data (fit + curve vs ys)
    full_fit = out.best_fit + curve
    r2 = float(1.0 - np.var(ys - full_fit) / np.var(ys))
    # the I(2D)/I(G) metric must use the 2D overtone, not the G* band
    twod_peaks = [r for r in records if r["label"] == "2D"] or records
    return {"peaks": records, "peak": (twod_peaks[0] if twod_peaks else None),
            "r_squared": r2, "result": out, "baseline": baseline,
            "x": xs, "y": ys}
