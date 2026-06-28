"""Spectral-shift calibration from the crystalline-Si substrate line (520.7 cm^-1)."""
import numpy as np
from lmfit.models import LinearModel, LorentzianModel

from .peaks import slice_region


def find_si_peak(x, y, region=(480.0, 560.0), min_prominence=200.0):
    """Fit a Lorentzian to the Si line and return ``(center, stderr, height)``.

    Returns ``None`` if the region is empty or no peak clears ``min_prominence``
    above the local baseline.
    """
    xs, ys = slice_region(x, y, *region)
    if len(xs) < 10:
        return None
    baseline = np.percentile(ys, 10)
    if (ys.max() - baseline) < min_prominence:
        return None

    bg = LinearModel(prefix="b_")
    pars = bg.guess(ys, x=xs)
    si = LorentzianModel(prefix="si_")
    pars.update(si.make_params())
    c0 = float(xs[np.argmax(ys)])
    pars["si_center"].set(value=c0, min=region[0], max=region[1])
    pars["si_sigma"].set(value=3.0, min=0.5, max=20.0)
    pars["si_amplitude"].set(value=max(1.0, (ys.max() - baseline) * 5), min=1.0)
    out = (bg + si).fit(ys, pars, x=xs)

    center = float(out.params["si_center"].value)
    stderr = out.params["si_center"].stderr
    height = float(out.params["si_height"].value)
    return center, (None if stderr is None else float(stderr)), height


def calibrate(x, y, nominal=520.7, region=(480.0, 560.0), min_prominence=200.0):
    """Return ``(x_calibrated, info)``.

    ``info`` is a dict with ``applied`` (bool), ``offset`` (cm^-1, observed-nominal),
    ``si_center``, ``si_center_stderr`` and ``nominal``.  When no Si line is found,
    the offset is 0 and ``x`` is returned unchanged with ``applied=False``.
    """
    found = find_si_peak(x, y, region=region, min_prominence=min_prominence)
    if found is None:
        return x, {"applied": False, "offset": 0.0, "si_center": None,
                   "si_center_stderr": None, "nominal": nominal,
                   "warning": "No Si peak found; calibration skipped."}
    center, stderr, _ = found
    offset = center - nominal
    return x - offset, {"applied": True, "offset": float(offset),
                        "si_center": float(center), "si_center_stderr": stderr,
                        "nominal": nominal}
