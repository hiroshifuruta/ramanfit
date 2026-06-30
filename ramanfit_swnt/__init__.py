"""ramanfit_swnt - single-wall carbon nanotube Raman analysis.

Pipeline: load -> Si-calibrate -> fit RBM / G / D / 2D -> (n,m) assignment.

Example
-------
>>> from ramanfit_swnt import load_spectrum, analyze_swnt, SwntConfig
>>> x, y = load_spectrum("data/LA2026062501.tsv")
>>> res = analyze_swnt(x, y, SwntConfig(laser_nm=532))
>>> print(res["calibration"]["offset"])
"""
from .config import (SwntConfig, load_config, save_rbm_centers,
                     rbm_sidecar_path)
from .io import load_spectrum, choose_spectrum_file
from .calibrate import calibrate, find_si_peak
from .export import export_pdf
from . import swnt, kataura, report

__all__ = ["SwntConfig", "load_config", "save_rbm_centers", "rbm_sidecar_path",
           "load_spectrum", "choose_spectrum_file", "calibrate", "find_si_peak",
           "export_pdf", "analyze_swnt", "swnt", "kataura", "report"]


def analyze_swnt(x, y, config=None):
    """Run the full SWNT analysis pipeline and return a results dict.

    Keys: ``calibration``, ``rbm``, ``gband``, ``d_band``, ``twod_band``,
    ``quality`` (ID_IG, I2D_IG, twod_center) and ``config``.
    """
    cfg = config or SwntConfig()

    # 1. Si calibration (applied to x before every downstream fit)
    x_cal, cal = calibrate(x, y, nominal=cfg.si_nominal, region=cfg.si_region,
                           min_prominence=cfg.si_min_prominence)

    # 2. RBM -> diameters -> candidate (n,m)
    rbm = swnt.fit_rbm(x_cal, y, region=cfg.rbm_region, relation=cfg.rbm_relation,
                       min_prominence_frac=cfg.rbm_min_prominence_frac,
                       extra_centers=cfg.rbm_extra_centers)
    for pk in rbm["peaks"]:
        pk["nm_candidates"] = kataura.assign_nm(
            pk["center"], cfg.laser_eV, relation=cfg.rbm_relation,
            diameter_tol=cfg.nm_diameter_tol, energy_window=cfg.nm_energy_window,
            max_n=cfg.nm_max_n)

    # 3. Joint D-G fit (D + D'' + G-/G+ over the whole region)
    dg = swnt.fit_dg(x_cal, y, region=cfg.dg_region,
                     gminus_lineshape=cfg.gminus_lineshape)
    # Backward-compatible views onto the joint fit
    gband = dg
    d_band = {"peak": dg["D"], "x": dg["x"], "y": dg["y"]}

    # 4. 2D band
    twod_band = swnt.fit_2d_band(x_cal, y, region=cfg.twod_region,
                                 shoulder_offset=cfg.twod_shoulder_offset,
                                 fit_gstar=cfg.twod_fit_gstar,
                                 fit_2dprime=cfg.twod_fit_2dprime)

    # 5. Quality metrics
    gp_height = dg["G_plus"]["height"]
    d_height = dg["D"]["height"]
    twod_peak = twod_band["peak"]
    quality = {
        "ID_IG": (d_height / gp_height) if (d_height and gp_height) else None,
        "twod_center": (twod_peak["center"] if twod_peak else None),
        "I2D_IG": (twod_peak["height"] / gp_height) if (twod_peak and gp_height) else None,
    }

    return {"calibration": cal, "rbm": rbm, "dg": dg, "gband": gband,
            "d_band": d_band, "twod_band": twod_band, "quality": quality,
            "config": cfg}
