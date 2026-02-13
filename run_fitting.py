#!/usr/bin/env python
"""
Raman curve fitting with tuned parameters
"""
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, LorentzianModel
import sys

# Get file from command line or use last known file
if len(sys.argv) > 1:
    INFILE = sys.argv[1]
else:
    INFILE = r"E:\Users\fur\data\20240329\CFC20240329_position2_ObjX50D1.txt"

print(f"Loading: {INFILE}")

try:
    data = np.loadtxt(INFILE, delimiter='\t')
except FileNotFoundError:
    print(f"File not found: {INFILE}")
    sys.exit(1)

# Extract data
x = data[:, 0]
y = data[:, 1]

# Extract D-G region (1000-1800 cm-1)
xDGindex1000 = np.searchsorted(x, 1000)
xDGindex1800 = np.searchsorted(x, 1800)

xDG = data[xDGindex1000:xDGindex1800, 0]
yDG = data[xDGindex1000:xDGindex1800, 1]

print(f"Data range: {xDG[0]:.1f} - {xDG[-1]:.1f} cm-1")
print(f"Data points: {len(xDG)}")

# Setup models with TUNED parameters
bg = LinearModel(prefix='lin_')
pars = bg.guess(yDG, x=xDG)

# D peak (l1) - TUNED
lorentz1 = LorentzianModel(prefix='l1_')
pars.update(lorentz1.make_params())
pars['l1_center'].set(value=1346, min=1300, max=1380)
pars['l1_sigma'].set(value=28, min=10, max=50)
pars['l1_amplitude'].set(value=7500, min=100)

# G peak (l2) - TUNED
lorentz2 = LorentzianModel(prefix='l2_')
pars.update(lorentz2.make_params())
pars['l2_center'].set(value=1580, min=1520, max=1590)
pars['l2_sigma'].set(value=24, min=10, max=40)
pars['l2_amplitude'].set(value=10600, min=100)

# G' peak (l3) - TUNED MORE AGGRESSIVE
lorentz3 = LorentzianModel(prefix='l3_')
pars.update(lorentz3.make_params())
pars['l3_center'].set(value=1615, min=1600, max=1670)
pars['l3_sigma'].set(value=15, min=8, max=80)
pars['l3_amplitude'].set(value=3000, min=100, max=5000)

# L4 peak (l4) - TUNED MORE AGGRESSIVE
lorentz4 = LorentzianModel(prefix='l4_')
pars.update(lorentz4.make_params())
pars['l4_center'].set(value=1500, min=1450, max=1550)
pars['l4_sigma'].set(value=40, min=10, max=100)
pars['l4_amplitude'].set(value=1500, min=50, max=3000)

# Combine models and fit
print("\nFitting...")
mod = lorentz1 + lorentz2 + lorentz3 + lorentz4 + bg
out = mod.fit(yDG, pars, x=xDG)

print("\n" + "="*60)
print("FITTING RESULTS")
print("="*60)
print(out.fit_report())

# Extract results
od = out.params
l1_height_stderr = od['l1_height'].stderr
l2_height_stderr = od['l2_height'].stderr
vd = out.params.valuesdict()

l2_area = np.pi * vd['l2_amplitude'] * vd['l2_fwhm']
l1_area = np.pi * vd['l1_amplitude'] * vd['l1_fwhm']

GDAreaRatio = l2_area / l1_area
l1_height = vd['l1_height']
l2_height = vd['l2_height']
GDHeightRatio = l2_height / l1_height

GDHeightRatioMax = (l2_height + l2_height_stderr) / (l1_height - l1_height_stderr)
GDHeightRatioMin = (l2_height - l2_height_stderr) / (l1_height + l1_height_stderr)
GDHeightRatioPlus = GDHeightRatioMax - GDHeightRatio
GDHeightRatioMinus = GDHeightRatio - GDHeightRatioMin

print("\n" + "="*60)
print("SAMPLE ANALYSIS RESULTS")
print("="*60)
print(f"File: {INFILE}")
print(f"G/D Height Ratio = {GDHeightRatio:.6f} +/- {GDHeightRatioPlus:.6f}")
print(f"G/D Area Ratio   = {GDAreaRatio:.6f}")
print("="*60)

# Plot results
fig, ax = plt.subplots(3, 1, dpi=300, figsize=(10, 10))
ax = ax.ravel()

ax[0].plot(xDG, out.best_fit - yDG, 'C3-', alpha=0.5, linewidth=2)
ax[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax[0].set(ylabel="Residual [cps]")
ax[0].grid(True, alpha=0.3)

ax[1].plot(xDG, yDG, 'C1.', alpha=0.5, label='Data')
ax[1].plot(xDG, out.best_fit, '-', label='Best fit', zorder=10, lw=2, alpha=0.8)
ax[1].set(ylabel="Intensity [cps]")
ax[1].legend(loc='best', fontsize='small')
ax[1].grid(True, alpha=0.3)

ax[2].plot(xDG, yDG, 'C1.', alpha=0.5, label='Data')
comps = out.eval_components(x=xDG)
ax[2].plot(xDG, comps['l1_'] + comps['lin_'], 'C2--', label='D peak', linewidth=2)
ax[2].fill_between(xDG, comps['l1_'] + comps['lin_'], comps['lin_'], facecolor='C2', alpha=0.3)
ax[2].plot(xDG, comps['l2_'] + comps['lin_'], 'C3--', label='G peak', linewidth=2)
ax[2].fill_between(xDG, comps['l2_'] + comps['lin_'], comps['lin_'], facecolor='C3', alpha=0.3)
ax[2].plot(xDG, comps['l3_'] + comps['lin_'], 'C4--', label="G' peak", linewidth=2)
ax[2].fill_between(xDG, comps['l3_'] + comps['lin_'], comps['lin_'], facecolor='C4', alpha=0.3)
ax[2].plot(xDG, comps['l4_'] + comps['lin_'], 'C5--', label='L4 peak', linewidth=2)
ax[2].fill_between(xDG, comps['l4_'] + comps['lin_'], comps['lin_'], facecolor='C5', alpha=0.3)

for i in range(3):
    ax[i].set(xlabel="Raman shift [cm-1]")
ax[2].legend(loc='best', fontsize='small')
ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("ramfit_tuned.png", dpi=130, bbox_inches='tight')
print("\nPlot saved to: ramfit_tuned.png")
plt.show()
