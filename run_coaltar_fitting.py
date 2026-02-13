#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Raman Coaltar Fitting - 6-Component Model
Fixed D peak center + 800 cm-1 feature
"""
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, LorentzianModel, GaussianModel
import os
import sys

os.environ['PYTHONIOENCODING'] = 'utf-8'

# Data file
INFILE = r"D:\Users\fur\OneDrive - 高知工科大学・高知県立大学\05共同研究\honda\Raman\l405.csv"

print("="*70)
print("RAMAN SPECTRUM FITTING - 6 COMPONENT MODEL")
print("="*70)
print(f"\nLoading: {os.path.basename(INFILE)}...")

try:
    with open(INFILE, 'r', encoding='utf-8-sig') as f:
        data = np.genfromtxt(f, delimiter=',', dtype=float)
except:
    data = np.loadtxt(INFILE, delimiter='\t')

x = data[:, 0]
y = data[:, 1]

print(f"Data loaded: {len(x)} points")
print(f"Range: {x[0]:.1f} - {x[-1]:.1f} cm-1")

# Setup models
print("\nSetting up 6-component model...")
bg = LinearModel(prefix='lin_')
pars = bg.guess(y, x=x)

# FIXED D PEAK CENTER
d_center_fixed = 1373.0

# 1. Feature around 800 cm-1
lorentz_800 = LorentzianModel(prefix='l800_')
pars.update(lorentz_800.make_params())
pars['l800_center'].set(value=800, min=780, max=820)
pars['l800_sigma'].set(value=30, min=10, max=80)
pars['l800_amplitude'].set(value=5000, min=10)

# 2. D (broad) - FIXED CENTER
lorentz_d_broad = LorentzianModel(prefix='d_broad_')
pars.update(lorentz_d_broad.make_params())
pars['d_broad_center'].set(value=d_center_fixed, vary=False)
pars['d_broad_sigma'].set(value=361/2, min=100, max=250)
pars['d_broad_amplitude'].set(value=50000, min=1000)

# 3. D1 - FIXED CENTER (same as D broad)
lorentz_d1 = LorentzianModel(prefix='d1_')
pars.update(lorentz_d1.make_params())
pars['d1_center'].set(value=d_center_fixed, vary=False)
pars['d1_sigma'].set(value=196/2, min=50, max=150)
pars['d1_amplitude'].set(value=30000, min=1000)

# 4. D3 - Gaussian
gauss_d3 = GaussianModel(prefix='d3_')
pars.update(gauss_d3.make_params())
pars['d3_center'].set(value=1555, min=1540, max=1580)
pars['d3_sigma'].set(value=117/(2*np.sqrt(2*np.log(2))), min=20, max=80)
pars['d3_amplitude'].set(value=10000, min=100)

# 5. G - Lorentzian
lorentz_g = LorentzianModel(prefix='g_')
pars.update(lorentz_g.make_params())
pars['g_center'].set(value=1590, min=1575, max=1610)
pars['g_sigma'].set(value=79/2, min=20, max=80)
pars['g_amplitude'].set(value=50000, min=1000)

# 6. D' - Lorentzian
lorentz_dp = LorentzianModel(prefix='dp_')
pars.update(lorentz_dp.make_params())
pars['dp_center'].set(value=1622, min=1610, max=1640)
pars['dp_sigma'].set(value=57/2, min=15, max=50)
pars['dp_amplitude'].set(value=30000, min=500)

# Fit
print("\nFitting (this may take a moment)...")
mod = lorentz_800 + lorentz_d_broad + lorentz_d1 + gauss_d3 + lorentz_g + lorentz_dp + bg
out = mod.fit(y, pars, x=x)

print("Fitting complete!")
print("\n" + out.fit_report())

# Extract results
vd = out.params.valuesdict()

print("\n" + "="*70)
print("FITTED PEAK PARAMETERS")
print("="*70)

peaks = [
    ('800 cm-1 feature', 'l800_'),
    ('D (broad)', 'd_broad_'),
    ('D1', 'd1_'),
    ('D3', 'd3_'),
    ('G', 'g_'),
    ("D'", 'dp_')
]

for name, prefix in peaks:
    center = vd[f'{prefix}center']
    amplitude = vd[f'{prefix}amplitude']
    sigma = vd[f'{prefix}sigma']

    if 'd3' in prefix:
        fwhm = sigma * 2 * np.sqrt(2 * np.log(2))
    else:
        fwhm = sigma * 2

    center_err = out.params[f'{prefix}center'].stderr
    if center_err is None:
        center_err = 0.0

    print(f"\n{name}:")
    print(f"  Center:    {center:.2f} +/- {center_err:.2f} cm-1")
    print(f"  FWHM:      {fwhm:.2f} cm-1")
    print(f"  Amplitude: {amplitude:.2f}")

print(f"\n{'='*70}")
print(f"D (broad) and D1 centers FIXED at 1373 cm-1 (no shift)")
print(f"\nFit Quality:")
print(f"  R-squared: {1 - out.residual.var() / np.var(y):.6f}")
print(f"  Chi-square: {out.chisqr:.2f}")
print("="*70)

# Calculate ratios
h_d_broad = vd['d_broad_amplitude'] / vd['d_broad_sigma']
h_d1 = vd['d1_amplitude'] / vd['d1_sigma']
h_d3 = vd['d3_amplitude'] / vd['d3_sigma']
h_g = vd['g_amplitude'] / vd['g_sigma']
h_dp = vd['dp_amplitude'] / vd['dp_sigma']

a_d_broad = np.pi * vd['d_broad_amplitude'] * vd['d_broad_sigma']
a_d1 = np.pi * vd['d1_amplitude'] * vd['d1_sigma']
a_g = np.pi * vd['g_amplitude'] * vd['g_sigma']
a_dp = np.pi * vd['dp_amplitude'] * vd['dp_sigma']

h_d_total = h_d_broad + h_d1
a_d_total = a_d_broad + a_d1

print("\n" + "="*70)
print("SPECTROSCOPIC RATIOS")
print("="*70)
print(f"\nIntensity Ratios (I = height):")
print(f"  I(D_broad)/I(G)  = {h_d_broad/h_g:.3f}")
print(f"  I(D1)/I(G)       = {h_d1/h_g:.3f}")
print(f"  I(D+D1)/I(G)     = {h_d_total/h_g:.3f}")
print(f"  I(D3)/I(G)       = {h_d3/h_g:.3f}")
print(f"  I(D')/I(G)       = {h_dp/h_g:.3f}")

print(f"\nArea Ratios (A = amplitude × FWHM):")
print(f"  A(D_broad)/A(G)  = {a_d_broad/a_g:.3f}")
print(f"  A(D1)/A(G)       = {a_d1/a_g:.3f}")
print(f"  A(D+D1)/A(G)     = {a_d_total/a_g:.3f}")
print(f"  A(D')/A(G)       = {a_dp/a_g:.3f}")
print("="*70)

# Plot
print("\nGenerating plot...")
fig, ax = plt.subplots(2, 1, figsize=(14, 10), dpi=300)

# Top: Spectrum with fit and components
ax[0].plot(x, y, 'k.', alpha=0.4, markersize=2, label='Data')
ax[0].plot(x, out.best_fit, 'r-', lw=2.5, alpha=0.8, label='Best fit')

# Plot individual components
comps = out.eval_components(x=x)
colors = ['C7', 'C2', 'C3', 'C4', 'C1', 'C5']
component_names = [
    ('l800_', '800 cm-1'),
    ('d_broad_', 'D (broad)'),
    ('d1_', 'D1'),
    ('d3_', 'D3'),
    ('g_', 'G'),
    ('dp_', "D'")
]

for color, (prefix, name) in zip(colors, component_names):
    if prefix in comps:
        component = comps[prefix] + comps['lin_']
        ax[0].plot(x, component, '--', color=color, label=name, lw=2, alpha=0.6)
        ax[0].fill_between(x, component, comps['lin_'], facecolor=color, alpha=0.15)

ax[0].set_ylabel('Intensity', fontsize=12)
ax[0].legend(loc='upper right', fontsize=10, ncol=3)
ax[0].grid(True, alpha=0.3)
ax[0].set_xlim(x[0], x[-1])
ax[0].set_title('Raman Spectrum - 6 Component Fit (D center fixed at 1373 cm-1)', fontsize=12)

# Bottom: Residuals
residual = out.best_fit - y
ax[1].plot(x, residual, 'C3-', alpha=0.6, lw=1)
ax[1].axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax[1].fill_between(x, residual, 0, alpha=0.2, color='C3')
ax[1].set_xlabel('Raman shift [cm-1]', fontsize=12)
ax[1].set_ylabel('Residual', fontsize=12)
ax[1].grid(True, alpha=0.3)
ax[1].set_xlim(x[0], x[-1])

plt.tight_layout()
plt.savefig('ramanfit_coaltar_result.png', dpi=150, bbox_inches='tight')
print("Plot saved: ramanfit_coaltar_result.png")
plt.show()

print("\n" + "="*70)
print("FITTING COMPLETE")
print("="*70)
