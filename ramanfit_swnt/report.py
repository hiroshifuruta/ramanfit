"""Plotting and tabular reporting for SWNT analysis results."""
import numpy as np


def plot_analysis(results, ax=None):
    """Draw a 4-panel figure (RBM, D, G, 2D) from an ``analyze_swnt`` result.

    Returns the matplotlib Figure.  Imports matplotlib lazily so importing the
    package does not require a display backend.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    axes = axes.ravel()

    # --- RBM ---
    rbm = results["rbm"]
    ax0 = axes[0]
    if rbm.get("result") is not None:
        out = rbm["result"]
        xr, yr = rbm["x"], rbm["y"]
        comps = out.eval_components(x=xr)
        ax0.plot(xr, yr, "C1.", ms=2, alpha=0.4)
        ax0.plot(xr, out.best_fit, "k-", lw=1.5)
        for key in comps:
            if key == "lin_":
                continue
            ax0.fill_between(xr, comps[key] + comps["lin_"], comps["lin_"], alpha=0.3)
        for pk in rbm["peaks"]:
            ax0.annotate(f"{pk['center']:.0f}\n{pk['diameter_nm']:.2f}nm",
                         (pk["center"], 0), fontsize=7, ha="center", va="bottom")
    ax0.set_title("RBM -> diameter"); ax0.set_xlabel("Raman shift [cm$^{-1}$]")

    # --- D band ---
    ax1 = axes[1]
    d = results["d_band"]
    ax1.plot(d["x"], d["y"], "C1.", ms=2, alpha=0.4)
    if d["peak"] is not None:
        ax1.axvline(d["peak"]["center"], color="C2", ls="--", lw=1)
    ax1.set_title("D band"); ax1.set_xlabel("Raman shift [cm$^{-1}$]")

    # --- G band ---
    ax2 = axes[2]
    g = results["gband"]
    out = g["result"]
    xg, yg = g["x"], g["y"]
    comps = out.eval_components(x=xg)
    ax2.plot(xg, yg, "C1.", ms=2, alpha=0.4)
    ax2.plot(xg, out.best_fit, "k-", lw=1.5)
    lin = comps.get("lin_", np.zeros_like(xg))
    for key, col, lab in [("gp_", "C3", "G+"), ("gm_", "C0", "G-")]:
        if key in comps:
            ax2.fill_between(xg, comps[key] + lin, lin, alpha=0.3, color=col, label=lab)
    verdict = "metallic" if g.get("metallic") else "semiconducting"
    ax2.legend(); ax2.set_title(f"G band ({verdict})"); ax2.set_xlabel("Raman shift [cm$^{-1}$]")

    # --- 2D band ---
    ax3 = axes[3]
    t = results["twod_band"]
    ax3.plot(t["x"], t["y"], "C1.", ms=2, alpha=0.4)
    if t["peak"] is not None:
        ax3.axvline(t["peak"]["center"], color="C4", ls="--", lw=1)
    ax3.set_title("2D band"); ax3.set_xlabel("Raman shift [cm$^{-1}$]")

    fig.tight_layout()
    return fig


def summary_text(results):
    """Human-readable multi-line summary of an ``analyze_swnt`` result."""
    L = []
    cal = results["calibration"]
    if cal["applied"]:
        L.append(f"Si calibration: observed {cal['si_center']:.2f} cm-1, "
                 f"offset {cal['offset']:+.2f} cm-1 (nominal {cal['nominal']})")
    else:
        L.append(f"Si calibration: NOT applied ({cal.get('warning', '')})")

    L.append("\nRBM peaks (diameter, candidate (n,m)):")
    for pk in results["rbm"]["peaks"]:
        nm = pk.get("nm_candidates", [])
        nm_s = ", ".join(f"({c['n']},{c['m']}){c['type']}" for c in nm[:3]) or "-"
        L.append(f"  {pk['center']:7.1f} cm-1  d={pk['diameter_nm']:.2f} nm   {nm_s}")

    g = results["gband"]
    L.append(f"\nG band: G+ {g['G_plus']['center']:.1f}, G- {g['G_minus']['center']:.1f} "
             f"(split {g['splitting']:.1f} cm-1) -> "
             f"{'metallic' if g['metallic'] else 'semiconducting'}")

    q = results["quality"]
    L.append(f"I(D)/I(G+) = {q['ID_IG']:.3f}" if q['ID_IG'] is not None else "I(D)/I(G+) = n/a")
    if q["twod_center"] is not None:
        L.append(f"2D band: {q['twod_center']:.1f} cm-1, I(2D)/I(G+) = {q['I2D_IG']:.3f}")
    return "\n".join(L)
