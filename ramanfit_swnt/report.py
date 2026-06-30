"""Plotting and tabular reporting for SWNT analysis results."""
import numpy as np


def plot_analysis(results, ax=None):
    """Draw the analysis as three band columns (RBM, D-G, 2D), each with the
    data + joint fit on top and the (fit − data) residual directly below it,
    sharing the x-axis so data and residual line up vertically.

    Returns the matplotlib Figure.  Imports matplotlib lazily so importing the
    package does not require a display backend.
    """
    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(2, 3, figsize=(16, 8), sharex="col",
                             gridspec_kw={"height_ratios": [3, 1]})

    def _band(col, x, y, out, title, comp_styles=None, annot=None):
        """Plot data+fit on the top row and residual on the bottom row."""
        top, bot = axes[0][col], axes[1][col]
        top.plot(x, y, "C1.", ms=2, alpha=0.4, label="data")
        if out is not None:
            comps = out.eval_components(x=x)
            lin = comps.get("lin_", np.zeros_like(x))
            top.plot(x, out.best_fit, "k-", lw=1.5, label="fit")
            if comp_styles is None:                 # auto colours, no legend
                for key in comps:
                    if key != "lin_":
                        top.fill_between(x, comps[key] + lin, lin, alpha=0.3)
            else:
                for key, col_, lab in comp_styles:
                    if key in comps:
                        top.fill_between(x, comps[key] + lin, lin,
                                         alpha=0.3, color=col_, label=lab)
                top.legend(fontsize=8)
            bot.plot(x, out.best_fit - y, "C3-", lw=0.8, alpha=0.7)
        if annot:
            annot(top)
        bot.axhline(0, color="k", lw=0.5)
        top.set_title(title)
        bot.set_xlabel("Raman shift [cm$^{-1}$]")
        bot.set_ylabel("fit − data")

    # --- RBM ---
    rbm = results["rbm"]
    def _rbm_annot(top):
        for pk in rbm["peaks"]:
            top.annotate(f"{pk['center']:.0f}\n{pk['diameter_nm']:.2f}nm",
                         (pk["center"], 0), fontsize=7, ha="center", va="bottom")
    _band(0, rbm["x"], rbm["y"], rbm.get("result"), "RBM -> diameter",
          annot=_rbm_annot)

    # --- D-G region (joint fit) ---
    g = results["dg"]
    verdict = "metallic" if g.get("metallic") else "semiconducting"
    _band(1, g["x"], g["y"], g["result"],
          f"D-G region ({verdict}, R²={g['r_squared']:.4f})",
          comp_styles=[("d_", "C2", "D"), ("dpp_", "C5", "D''"),
                       ("gm_", "C0", "G-"), ("gp_", "C3", "G+"),
                       ("dp_", "C6", "D'")])

    # --- 2D region (multi-component) ---
    t = results["twod_band"]
    n2d = sum(1 for p in t.get("peaks", []) if p.get("label") == "2D")
    has_gstar = any(p.get("label") == "G*" for p in t.get("peaks", []))
    title = f"2D band ({n2d} comp{'s' if n2d != 1 else ''}"
    title += " + G*)" if has_gstar else ")"
    _band(2, t["x"], t["y"], t.get("result"), title)

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

    g = results["dg"]
    L.append(f"\nD-G joint fit (R²={g['r_squared']:.4f}):")
    L.append(f"  D   {g['D']['center']:.1f} cm-1 (FWHM {g['D']['fwhm']:.0f})")
    L.append(f"  D'' {g['Dpp']['center']:.1f} cm-1 (FWHM {g['Dpp']['fwhm']:.0f}, broad disorder band)")
    L.append(f"  G+  {g['G_plus']['center']:.1f}, G- {g['G_minus']['center']:.1f} "
             f"(split {g['splitting']:.1f} cm-1) -> "
             f"{'metallic' if g['metallic'] else 'semiconducting'}")
    if "Dprime" in g:
        L.append(f"  D'  {g['Dprime']['center']:.1f} cm-1 (FWHM {g['Dprime']['fwhm']:.0f})")

    q = results["quality"]
    L.append(f"I(D)/I(G+) = {q['ID_IG']:.3f}" if q['ID_IG'] is not None else "I(D)/I(G+) = n/a")
    twod = results["twod_band"]
    peaks = twod.get("peaks", [])
    if peaks:
        L.append(f"\n2D region ({len(peaks)} components, R²={twod['r_squared']:.4f}):")
        for pk in sorted(peaks, key=lambda r: r["center"]):
            tag = pk.get("label", "2D")
            L.append(f"  {tag:3s} {pk['center']:7.1f} cm-1 "
                     f"(FWHM {pk['fwhm']:.0f}, height {pk['height']:.0f})")
        L.append(f"  dominant 2D {q['twod_center']:.1f} cm-1, "
                 f"I(2D)/I(G+) = {q['I2D_IG']:.3f}")
    elif q["twod_center"] is not None:
        L.append(f"2D band: {q['twod_center']:.1f} cm-1, I(2D)/I(G+) = {q['I2D_IG']:.3f}")
    return "\n".join(L)
