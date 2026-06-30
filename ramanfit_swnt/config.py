"""Configuration for SWNT Raman analysis."""
from dataclasses import dataclass, field
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
    rbm_min_prominence_frac: float = 0.04  # fraction of (max-baseline)
    # Manually-seeded RBM centers (cm^-1) for shoulders the detector misses,
    # e.g. the ~240 cm^-1 peak hidden on the flank of the 233 line.
    rbm_extra_centers: Tuple[float, ...] = (240.0,)

    # --- D / G band (fit jointly over the whole D-G region) ---
    # D, D'' (broad disorder band), G- and G+ are fit simultaneously so the
    # D-to-G valley is modelled instead of forced onto a straight baseline.
    dg_region: Tuple[float, float] = (1000.0, 1700.0)
    gminus_lineshape: str = "lorentzian"  # "lorentzian" (semicond.) or "bwf" (metallic)

    # --- 2D ---
    # Wide window so both the broad G* (~2450, iTOLA) band and the multi-
    # component 2D overtone (~2655, diameter-distribution split) are captured;
    # G* + the 2D main peak + a shoulder each side are fit jointly and the
    # negligible ones pruned.
    twod_region: Tuple[float, float] = (2380.0, 2820.0)
    twod_shoulder_offset: float = 35.0      # cm^-1, shoulder seed spacing
    twod_fit_gstar: bool = True             # include the ~2450 G* band

    # --- (n,m) assignment ---
    nm_diameter_tol: float = 0.06      # nm; |d_candidate - d_measured| tolerance
    nm_energy_window: float = 0.20     # eV; |E_ii - E_laser| resonance window
    nm_max_n: int = 25

    @property
    def laser_eV(self) -> float:
        return HC_EV_NM / self.laser_nm
