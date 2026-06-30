"""Tests for the SWNT analysis pipeline.

Uses the real SWNT spectrum data/LA2026062501.tsv (532 nm, on Si) as a fixture
and locks in the numbers established during development.
"""
import os
import numpy as np
import pytest

from ramanfit_swnt import load_spectrum, analyze_swnt, SwntConfig
from ramanfit_swnt import kataura

DATA = os.path.join(os.path.dirname(__file__), "..", "data", "LA2026062501.tsv")
# The sample spectrum lives under data/ (gitignored), so the full-pipeline tests
# are skipped where it is absent (e.g. CI).  The data-independent unit tests
# below always run.
needs_data = pytest.mark.skipif(not os.path.exists(DATA),
                                reason="sample SWNT spectrum not present")


@pytest.fixture(scope="module")
def result():
    if not os.path.exists(DATA):
        pytest.skip("sample SWNT spectrum not present")
    x, y = load_spectrum(DATA)
    return analyze_swnt(x, y, SwntConfig(laser_nm=532))


@needs_data
def test_loader_sorted_two_columns():
    x, y = load_spectrum(DATA)
    assert x.ndim == y.ndim == 1
    assert len(x) == len(y) > 1000
    assert np.all(np.diff(x) >= 0)          # ascending


def test_si_calibration_offset(result):
    cal = result["calibration"]
    assert cal["applied"] is True
    assert cal["si_center"] == pytest.approx(519.76, abs=0.3)
    assert cal["offset"] == pytest.approx(-0.94, abs=0.2)


def test_rbm_peaks_and_diameters(result):
    peaks = result["rbm"]["peaks"]
    assert len(peaks) >= 4
    assert result["rbm"]["r_squared"] > 0.95
    for pk in peaks:
        assert 0.8 <= pk["diameter_nm"] <= 1.4


def test_dg_joint_fit(result):
    g = result["dg"]
    # joint D + D'' + G-/G+ fit should be excellent and well-determined
    assert g["r_squared"] > 0.99
    assert g["D"]["center"] == pytest.approx(1338, abs=8)
    assert 1450 <= g["Dpp"]["center"] <= 1560          # broad disorder band
    assert g["G_plus"]["center"] == pytest.approx(1592, abs=6)
    # G+ must sit above G- (the ordering bug we explicitly guard against)
    assert g["G_plus"]["center"] > g["G_minus"]["center"]
    assert 1540 <= g["G_minus"]["center"] <= 1585
    # D' sits just above G+ and absorbs its upper shoulder
    assert g["Dprime"]["center"] > g["G_plus"]["center"]
    assert 1595 <= g["Dprime"]["center"] <= 1630
    # uncertainties must be finite (railed fits return None)
    assert g["G_plus"]["center_stderr"] is not None
    assert g["D"]["center_stderr"] is not None


def test_low_defect_quality(result):
    q = result["quality"]
    assert q["ID_IG"] is not None and q["ID_IG"] < 0.2
    assert q["twod_center"] == pytest.approx(2655, abs=10)


def test_2d_multicomponent(result):
    t = result["twod_band"]
    # the second-order region resolves into the full multi-band set
    assert len(t["peaks"]) >= 5
    assert t["r_squared"] > 0.99
    # the dominant *2D* component is near 2655 and is the metric peak
    assert t["peak"]["label"] == "2D"
    assert t["peak"]["center"] == pytest.approx(2655, abs=15)
    # the 2D' band (overtone of D') near 3185 is resolved
    dprime = [p for p in t["peaks"] if p["label"] == "2D'"]
    assert len(dprime) == 1
    assert dprime[0]["center"] == pytest.approx(3185, abs=40)
    # the baseline is a flat constant at the true continuum floor; it tracks the
    # far plateau past 2D' (~3400-3500) within ~15 cps and is horizontal
    base = t["baseline"]
    xs, ys = t["x"], t["y"]
    assert np.ptp(base) < 1e-6                       # constant (flat)
    sel = (xs >= 3400) & (xs <= 3500)
    assert abs(base[0] - float(np.median(ys[sel]))) < 15
    # components are distinct in center
    centers = sorted(p["center"] for p in t["peaks"])
    assert all(b - a > 5 for a, b in zip(centers, centers[1:]))


def test_choose_spectrum_file_headless(monkeypatch):
    # with the dialog disabled, the picker returns the default (notebooks stay
    # runnable non-interactively)
    from ramanfit_swnt import choose_spectrum_file
    monkeypatch.setenv("RAMANFIT_NO_DIALOG", "1")
    assert choose_spectrum_file(default="data/x.tsv") == "data/x.tsv"
    assert choose_spectrum_file() is None


def test_config_sidecar_and_shared(tmp_path):
    import json
    from ramanfit_swnt import (load_config, save_rbm_centers,
                               rbm_sidecar_path, SwntConfig)
    spec = str(tmp_path / "spec.txt")

    # no files -> defaults
    c = load_config(spec, shared_path=str(tmp_path / "none.json"))
    assert c.rbm_extra_centers == ()
    assert c.laser_nm == SwntConfig().laser_nm

    # per-file RBM sidecar round-trips as a tuple
    p = save_rbm_centers(spec, (190.0, 263.0))
    assert p == rbm_sidecar_path(spec)
    c = load_config(spec, shared_path=str(tmp_path / "none.json"))
    assert c.rbm_extra_centers == (190.0, 263.0)

    # shared config overlays known keys (tuple fields survive), ignores unknown
    shared = tmp_path / "_config.json"
    shared.write_text(json.dumps({"laser_nm": 633, "rbm_region": [180, 360],
                                  "unknown_key": 1}))
    c = load_config(spec, shared_path=str(shared))
    assert c.laser_nm == 633
    assert c.rbm_region == (180, 360)          # list -> tuple
    assert c.rbm_extra_centers == (190.0, 263.0)  # sidecar still applies

    # explicit kwargs win over the shared file
    assert load_config(spec, shared_path=str(shared), laser_nm=532).laser_nm == 532

    # clearing removes the sidecar (back to auto-detect)
    save_rbm_centers(spec, ())
    assert not os.path.exists(rbm_sidecar_path(spec))


def test_rbm_diameter_relation_monotonic():
    from ramanfit_swnt.swnt import rbm_to_diameter
    # higher RBM frequency -> smaller diameter
    assert rbm_to_diameter(200) > rbm_to_diameter(300)
    assert rbm_to_diameter(248, "248/w") == pytest.approx(1.0)


def test_kataura_geometry_and_metallicity():
    # (10,10) armchair: metallic, d ~ 1.36 nm
    assert kataura.is_metallic(10, 10)
    assert kataura.diameter(10, 10) == pytest.approx(1.357, abs=0.01)
    # (10,0) zigzag: mod(n-m,3)=1 -> semiconducting
    assert not kataura.is_metallic(10, 0)


def test_nm_assignment_returns_candidates(result):
    # at least one RBM peak should get a candidate (n,m) within resonance window
    n_with = sum(1 for pk in result["rbm"]["peaks"] if pk.get("nm_candidates"))
    assert n_with >= 1


def test_no_si_peak_skips_calibration():
    # a flat spectrum with no Si line -> calibration not applied, x unchanged
    from ramanfit_swnt import calibrate
    x = np.linspace(400, 600, 200)
    y = np.full_like(x, 100.0)
    x_cal, info = calibrate(x, y)
    assert info["applied"] is False
    assert np.allclose(x_cal, x)
