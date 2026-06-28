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
    rbm_region: Tuple[float, float] = (100.0, 350.0)
    rbm_relation: str = "248/w"        # "248/w" (Jorio) or "araujo"
    rbm_min_prominence_frac: float = 0.04  # fraction of (max-baseline)

    # --- G band ---
    gband_region: Tuple[float, float] = (1500.0, 1650.0)
    gminus_lineshape: str = "lorentzian"  # "lorentzian" (semicond.) or "bwf" (metallic)

    # --- D / 2D ---
    d_region: Tuple[float, float] = (1250.0, 1450.0)
    twod_region: Tuple[float, float] = (2500.0, 2800.0)

    # --- (n,m) assignment ---
    nm_diameter_tol: float = 0.06      # nm; |d_candidate - d_measured| tolerance
    nm_energy_window: float = 0.20     # eV; |E_ii - E_laser| resonance window
    nm_max_n: int = 25

    @property
    def laser_eV(self) -> float:
        return HC_EV_NM / self.laser_nm
