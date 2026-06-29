"""Shared multi-Lorentzian + linear-background fitting helpers.

This is the single place the peak bookkeeping (center / FWHM / height / area /
stderr / at-bound flag) lives, replacing the copy-pasted blocks scattered across
the original ``ramanfit.py`` and ``run_*.py`` scripts.
"""
import numpy as np
from lmfit.models import ConstantModel, LinearModel, LorentzianModel


def slice_region(x, y, lo, hi):
    """Return the (x, y) samples with ``lo <= x <= hi``."""
    m = (x >= lo) & (x <= hi)
    return x[m], y[m]


def _at_bound(par, rtol=1e-4):
    """True if a fitted parameter sits on (or against) one of its bounds."""
    flags = []
    if par.min is not None and np.isfinite(par.min):
        if abs(par.value - par.min) <= rtol * max(1.0, abs(par.min)):
            flags.append("min")
    if par.max is not None and np.isfinite(par.max):
        if abs(par.value - par.max) <= rtol * max(1.0, abs(par.max)):
            flags.append("max")
    return flags


def peak_record(out, prefix, label):
    """Extract a tidy dict for one Lorentzian component of an lmfit result."""
    p = out.params
    def g(name):
        return float(p[f"{prefix}{name}"].value)
    def err(name):
        e = p[f"{prefix}{name}"].stderr
        return None if e is None else float(e)
    at_bound = []
    for nm in ("center", "sigma", "amplitude"):
        for b in _at_bound(p[f"{prefix}{nm}"]):
            at_bound.append(f"{nm}@{b}")
    return {
        "label": label,
        "center": g("center"),
        "center_stderr": err("center"),
        "fwhm": g("fwhm"),
        "height": g("height"),
        "amplitude": g("amplitude"),
        "sigma": g("sigma"),
        "at_bound": at_bound,
    }


def fit_lorentzians(x, y, seeds, sigma0=10.0, sigma_bounds=(2.0, 80.0),
                    center_window=15.0, background="linear"):
    """Fit ``len(seeds)`` Lorentzians plus a background.

    ``seeds`` is a list of dicts; each may contain ``center`` (required) and
    optional ``prefix``, ``center_min``, ``center_max``, ``sigma``,
    ``sigma_min``, ``sigma_max``, ``amplitude``.

    ``background`` selects the baseline model: ``"linear"`` (default) or
    ``"constant"``.  A constant baseline is appropriate when a linear slope
    would be unphysical -- e.g. the RBM region, whose low-frequency edge is
    clipped by a notch filter below ~200 cm^-1.

    Returns ``(lmfit_result, [peak_record, ...])`` with records ordered as the
    seeds were given.
    """
    if background == "constant":
        bg = ConstantModel(prefix="lin_")
    elif background == "linear":
        bg = LinearModel(prefix="lin_")
    else:
        raise ValueError(f"Unknown background: {background!r}")
    pars = bg.guess(y, x=x)
    model = bg
    prefixes = []
    span = float(np.trapezoid(np.clip(y - np.min(y), 0, None), x)) if len(x) > 1 else 1.0
    for i, seed in enumerate(seeds):
        prefix = seed.get("prefix", f"p{i}_")
        prefixes.append((prefix, seed.get("label", prefix.rstrip("_"))))
        comp = LorentzianModel(prefix=prefix)
        pars.update(comp.make_params())
        c0 = seed["center"]
        pars[f"{prefix}center"].set(
            value=c0,
            min=seed.get("center_min", c0 - center_window),
            max=seed.get("center_max", c0 + center_window),
        )
        pars[f"{prefix}sigma"].set(
            value=seed.get("sigma", sigma0),
            min=seed.get("sigma_min", sigma_bounds[0]),
            max=seed.get("sigma_max", sigma_bounds[1]),
        )
        amp0 = seed.get("amplitude", max(1.0, span / max(1, len(seeds))))
        pars[f"{prefix}amplitude"].set(value=amp0, min=1.0)
        model = model + comp

    out = model.fit(y, pars, x=x)
    records = [peak_record(out, prefix, label) for prefix, label in prefixes]
    return out, records


def r_squared(out):
    """Coefficient of determination of an lmfit result against its own data."""
    return float(1.0 - out.residual.var() / np.var(out.data))
