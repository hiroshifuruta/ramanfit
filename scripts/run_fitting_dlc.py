#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Raman curve fitting optimized for DLC / Amorphous Carbon
- Uses Voigt profiles (Gaussian + Lorentzian) instead of pure Lorentzian
- 5-component model with strong L4 peak
- Better for broad, overlapping peaks in amorphous/DLC samples
"""
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, VoigtModel
import sys
import io
import os
os.environ['PYTHONIOENCODING'] = 'utf-8'

# Get file from command line or use last known file
if len(sys.argv) > 1:
    INFILE = sys.argv[1]
else:
    INFILE = r"E:\Users\fur\data\20240329\CFC20240329_position2_ObjX50D1.txt"

print("Loading data file...")

try:
    # Read with proper BOM handling
    with open(INFILE, 'r', encoding='utf-8-sig') as f:
        data = np.genfromtxt(f, delimiter=',', dtype=float)
except FileNotFoundError:
    print("File not found!")
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

# Setup models with DLC-optimized parameters (Voigt instead of Lorentzian)
bg = LinearModel(prefix='lin_')
pars = bg.guess(yDG, x=xDG)

# D peak (v1) - BROAD for amorphous carbon - VOIGT
voigt1 = VoigtModel(prefix='v1_')
pars.update(voigt1.make_params())
pars['v1_center'].set(value=1340, min=1300, max=1380)
pars['v1_sigma'].set(value=50, min=20, max=100)  # Much broader for DLC
pars['v1_amplitude'].set(value=8000, min=100)
pars['v1_gamma'].set(value=50, min=10, max=100)

# D' peak (v2) - shoulder on D peak - VOIGT
voigt2 = VoigtModel(prefix='v2_')
pars.update(voigt2.make_params())
pars['v2_center'].set(value=1500, min=1450, max=1550)  # L4 region
pars['v2_sigma'].set(value=60, min=20, max=100)  # Large in DLC
pars['v2_amplitude'].set(value=5000, min=100, max=10000)  # STRONG
pars['v2_gamma'].set(value=50, min=10, max=100)

# G peak (v3) - main peak - VOIGT
voigt3 = VoigtModel(prefix='v3_')
pars.update(voigt3.make_params())
pars['v3_center'].set(value=1580, min=1520, max=1610)
pars['v3_sigma'].set(value=40, min=15, max=80)  # Broader for amorphous
pars['v3_amplitude'].set(value=10000, min=100)
pars['v3_gamma'].set(value=40, min=10, max=100)

# G' peak (v4) - secondary peak - VOIGT
voigt4 = VoigtModel(prefix='v4_')
pars.update(voigt4.make_params())
pars['v4_center'].set(value=1620, min=1600, max=1680)
pars['v4_sigma'].set(value=30, min=10, max=100)
pars['v4_amplitude'].set(value=1000, min=10, max=3000)
pars['v4_gamma'].set(value=30, min=5, max=80)

# Extra component for broad background feature - VOIGT
voigt5 = VoigtModel(prefix='v5_')
pars.update(voigt5.make_params())
pars['v5_center'].set(value=1400, min=1350, max=1450)
pars['v5_sigma'].set(value=80, min=30, max=150)  # Very broad
pars['v5_amplitude'].set(value=2000, min=10, max=5000)
pars['v5_gamma'].set(value=60, min=20, max=120)

# Combine models and fit
print("\nFitting DLC/Amorphous carbon spectrum...")
mod = voigt1 + voigt2 + voigt3 + voigt4 + voigt5 + bg
out = mod.fit(yDG, pars, x=xDG)

print("\n" + "="*60)
print("FITTING RESULTS (DLC/Amorphous)")
print("="*60)
print(out.fit_report())

# Extract results
od = out.params
vd = out.params.valuesdict()

# Calculate areas for G and D peaks
v3_area = np.pi * vd.get('v3_amplitude', 0) * vd.get('v3_sigma', 1) * 2  # Voigt integral
v1_area = np.pi * vd.get('v1_amplitude', 0) * vd.get('v1_sigma', 1) * 2

# Heights
v1_height = vd.get('v1_amplitude', 0) / max(vd.get('v1_sigma', 1), 1e-15)
v3_height = vd.get('v3_amplitude', 0) / max(vd.get('v3_sigma', 1), 1e-15)

if v1_height > 0:
    GDHeightRatio = v3_height / v1_height
    GDAreaRatio = v3_area / v1_area
else:
    GDHeightRatio = 0
    GDAreaRatio = 0

print("\n" + "="*60)
print("SAMPLE ANALYSIS RESULTS (DLC)")
print("="*60)
print("File: [DLC sample data]")
print(f"G/D Height Ratio = {GDHeightRatio:.6f}")
print(f"G/D Area Ratio   = {GDAreaRatio:.6f}")
print(f"\nComponent Heights:")
print(f"  D peak (v1)  = {v1_height:.2f}")
print(f"  D' peak (v2) = {vd.get('v2_amplitude', 0) / max(vd.get('v2_sigma', 1), 1e-15):.2f}")
print(f"  G peak (v3)  = {v3_height:.2f}")
print(f"  G' peak (v4) = {vd.get('v4_amplitude', 0) / max(vd.get('v4_sigma', 1), 1e-15):.2f}")
print(f"  BG peak (v5) = {vd.get('v5_amplitude', 0) / max(vd.get('v5_sigma', 1), 1e-15):.2f}")
print("="*60)

# Plot results
fig, ax = plt.subplots(3, 1, dpi=300, figsize=(12, 10))
ax = ax.ravel()

# Residuals
ax[0].plot(xDG, out.best_fit - yDG, 'C3-', alpha=0.7, linewidth=1.5)
ax[0].axhline(y=0, color='k', linestyle='--', alpha=0.3)
ax[0].set(ylabel="Residual [cps]")
ax[0].grid(True, alpha=0.3)

# Best fit
ax[1].plot(xDG, yDG, 'C1.', alpha=0.6, label='Data')
ax[1].plot(xDG, out.best_fit, 'b-', label='Best fit', zorder=10, lw=2.5, alpha=0.9)
ax[1].set(ylabel="Intensity [cps]")
ax[1].legend(loc='best', fontsize='small')
ax[1].grid(True, alpha=0.3)

# Components
ax[2].plot(xDG, yDG, 'C1.', alpha=0.5, label='Data')
comps = out.eval_components(x=xDG)

colors = ['C2', 'C3', 'C4', 'C5', 'C6']
labels = ['D peak', "D' peak (L4)", 'G peak', "G' peak", 'BG feature']
prefixes = ['v1_', 'v2_', 'v3_', 'v4_', 'v5_']

for color, label, prefix in zip(colors, labels, prefixes):
    if prefix in comps:
        component = comps[prefix] + comps['lin_']
        ax[2].plot(xDG, component, '--', color=color, label=label, linewidth=2, alpha=0.7)
        ax[2].fill_between(xDG, component, comps['lin_'], facecolor=color, alpha=0.2)

ax[2].set(xlabel="Raman shift [cm-1]", ylabel="Intensity[cps]")
ax[2].legend(loc='best', fontsize='small', ncol=2)
ax[2].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig("output/ramfit_dlc.png", dpi=130, bbox_inches='tight')
print("\nPlot saved to: output/ramfit_dlc.png")
plt.show()
