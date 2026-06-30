"""Configuration for SWNT Raman analysis."""
import json
import os
from dataclasses import dataclass, fields
from typing import Tuple

# Photon energy (eV) = 1239.84193 (eV*nm) / wavelength (nm)
HC_EV_NM = 1239.84193


@dataclass
class SwntConfig:
    """Tunable parameters for :func:`ramanfit_swnt.analyze_swnt`.

    All region bounds are in cm^-1.  Defaults target a 532 nm measurement of
    arc/HiPco-class SWNT (~0.8-1.4 nm diameter) on a silicon substrate.
    """
    laser_nm: float = 532.0

    # --- Si calibration ---
    si_nominal: float = 520.7          # crystalline-Si reference line (cm^-1)
    si_region: Tuple[float, float] = (480.0, 560.0)
    si_min_prominence: float = 200.0   # min peak prominence to accept a Si line

    # --- RBM ---
    # Lower bound starts at 180 (not 100): a notch filter blocks the Rayleigh
    # line below ~200 cm^-1, so 150-180 is dominated by the filter's transmission
    # edge.  Excluding that edge leaves a flat background that fit_rbm models with
    # a *constant* baseline (a linear one would chase the notch slope).
    rbm_region: Tuple[float, float] = (180.0, 350.0)
    rbm_relation: str = "248/w"        # "248/w" (Jorio) or "araujo"
    rbm_min_prominence_frac: float = 0.03  # fraction of (max-baseline)
    # Manually-seeded RBM centers (cm^-1) for shoulders the detector still
    # misses on a given sample.  Empty by default -- the smoothed, low-floor
    # detector now resolves the shoulder bands on its own; add entries only for
    # a specific spectrum that needs them.
    rbm_extra_centers: Tuple[float, ...] = ()

    # --- D / G band (fit jointly over the whole D-G region) ---
    # D, D'' (broad disorder band), G- and G+ are fit simultaneously so the
    # D-to-G valley is modelled instead of forced onto a straight baseline.
    dg_region: Tuple[float, float] = (1000.0, 1800.0)
    gminus_lineshape: str = "lorentzian"  # "lorentzian" (semicond.) or "bwf" (metallic)

    # --- 2D ---
    # Wide window covering the whole second-order region so every band a
    # reference Igor multipeak fit resolves is captured: G* (~2446), the 2D
    # triplet (~2655), the weak D+D'' (~2915) and 2D'' (~3105) combination
    # bands, 2D' (~3185) and a weak ~3510 band.  A flat (constant) baseline is
    # used; the broad combination bands carry the inter-band intensity that a
    # curved background would otherwise have to chase.
    twod_region: Tuple[float, float] = (2380.0, 3560.0)
    twod_shoulder_offset: float = 35.0      # cm^-1, 2D shoulder seed spacing
    twod_fit_gstar: bool = True             # include the ~2446 G* band
    twod_fit_2dprime: bool = True           # include the ~3185 2D' band

    # --- (n,m) assignment ---
    nm_diameter_tol: float = 0.06      # nm; |d_candidate - d_measured| tolerance
    nm_energy_window: float = 0.20     # eV; |E_ii - E_laser| resonance window
    nm_max_n: int = 25

    @property
    def laser_eV(self) -> float:
        return HC_EV_NM / self.laser_nm


# --------------------------- config persistence ---------------------------
# Two-layer scheme:
#  * shared parameters (laser_nm, regions, lineshape, ...) live in one
#    project-level file, ``data/_config.json`` by default;
#  * the per-spectrum RBM shoulder seeds, which differ from file to file, live
#    in a sidecar next to each spectrum: ``data/<name>.txt`` -> ``data/<name>.rbm.json``.
_SHARED_CONFIG = os.path.join("data", "_config.json")

# tuple-typed fields need list<->tuple conversion when (de)serialising JSON
_TUPLE_FIELDS = {f.name for f in fields(SwntConfig)
                 if getattr(f.type, "__origin__", None) is tuple
                 or str(f.type).startswith("typing.Tuple")}


def rbm_sidecar_path(infile):
    """Return the RBM-seed sidecar path for a spectrum file.

    ``data/LA.._X0Y0.txt`` -> ``data/LA.._X0Y0.rbm.json``.
    """
    return os.path.splitext(infile)[0] + ".rbm.json"


def save_rbm_centers(infile, centers):
    """Write the manual RBM extra-centers for ``infile`` to its sidecar.

    Pass an empty/falsy ``centers`` to remove the sidecar (back to auto-detect).
    """
    path = rbm_sidecar_path(infile)
    if not centers:
        if os.path.exists(path):
            os.remove(path)
        return path
    with open(path, "w", encoding="utf-8") as fh:
        json.dump([float(c) for c in centers], fh)
    return path


def load_config(infile=None, shared_path=_SHARED_CONFIG, **overrides):
    """Build a :class:`SwntConfig`, layering shared file + per-file RBM seeds.

    1. start from ``SwntConfig`` defaults;
    2. apply the shared JSON (``data/_config.json``) if present -- only keys
       that are real ``SwntConfig`` fields are used, others are ignored;
    3. if ``infile`` is given and its ``.rbm.json`` sidecar exists, use it for
       ``rbm_extra_centers``;
    4. apply any explicit keyword ``overrides`` last (highest priority).

    Missing files are simply skipped, so this is safe to call unconditionally.
    """
    valid = {f.name for f in fields(SwntConfig)}
    values = {}

    if shared_path and os.path.exists(shared_path):
        with open(shared_path, encoding="utf-8") as fh:
            for k, v in json.load(fh).items():
                if k in valid:
                    values[k] = tuple(v) if k in _TUPLE_FIELDS and isinstance(v, list) else v

    if infile:
        sidecar = rbm_sidecar_path(infile)
        if os.path.exists(sidecar):
            with open(sidecar, encoding="utf-8") as fh:
                values["rbm_extra_centers"] = tuple(json.load(fh))

    for k, v in overrides.items():
        if k in valid:
            values[k] = v

    return SwntConfig(**values)
