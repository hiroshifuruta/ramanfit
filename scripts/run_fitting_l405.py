#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Raman Spectrum Fitting - OPTIMIZED 5-PEAK MODEL
Fit region: 750-2000 cm-1 (exclude noisy low-freq)
Better baseline correction, D3 shifted higher & narrower
Reference: 405 nm annealed sample
"""
import matplotlib.pyplot as plt
import numpy as np
from lmfit.models import LinearModel, LorentzianModel, GaussianModel
import os

os.environ['PYTHONIOENCODING'] = 'utf-8'

INFILE = r"D:\Users\fur\OneDrive - 高知工科大学・高知県立大学\05共同研究\honda\Raman\l405.csv"

print("="*70)
print("RAMAN SPECTRUM FITTING - OPTIMIZED 5-PEAK MODEL")
print("Fit region: 750-2000 cm-1 (exclude noise)")
print("="*70)

try:
    with open(INFILE, 'r', encoding='utf-8-sig') as f:
        data = np.genfromtxt(f, delimiter=',', dtype=float)
except:
    data = np.loadtxt(INFILE, delimiter='\t')

x_full = data[:, 0]
y_full = data[:, 1]

# EXTRACT FIT REGION: 750-2000 cm-1
idx_start = np.searchsorted(x_full, 750)
idx_end = np.searchsorted(x_full, 2000)

x = x_full[idx_start:idx_end]
y = y_full[idx_start:idx_end]

print(f"\nFull data: {len(x_full)} points ({x_full[0]:.1f}-{x_full[-1]:.1f} cm-1)")
print(f"Fit region: {len(x)} points ({x[0]:.1f}-{x[-1]:.1f} cm-1)")

print("\nSetting up 5-component model...")
bg = LinearModel(prefix='lin_')
pars = bg.guess(y, x=x)

# D PEAK CENTER - SAME position for both D and D1
d_center_fixed = 1360  # Both D peaks at same center

# 1. D (broad) - Lorentzian (broader width)
lorentz_d_broad = LorentzianModel(prefix='d_broad_')
pars.update(lorentz_d_broad.make_params())
pars['d_broad_center'].set(value=d_center_fixed, min=1350, max=1370)
pars['d_broad_sigma'].set(value=361/2, min=150, max=250)
pars['d_broad_amplitude'].set(value=50000, min=1000)

# 2. D1 - Lorentzian - SAME CENTER AS D_BROAD (narrower width)
lorentz_d1 = LorentzianModel(prefix='d1_')
pars.update(lorentz_d1.make_params())
pars['d1_center'].set(value=d_center_fixed, min=1350, max=1370)  # Same center as D_broad
pars['d1_sigma'].set(value=196/2, min=50, max=120)
pars['d1_amplitude'].set(value=30000, min=1000)

# 3. D3 - Gaussian (narrower, shifted higher)
gauss_d3 = GaussianModel(prefix='d3_')
pars.update(gauss_d3.make_params())
pars['d3_center'].set(value=1570, min=1540, max=1610)
pars['d3_sigma'].set(value=50/(2*np.sqrt(2*np.log(2))), min=15, max=80)
pars['d3_amplitude'].set(value=10000, min=100)

# 4. G - Lorentzian
lorentz_g = LorentzianModel(prefix='g_')
pars.update(lorentz_g.make_params())
pars['g_center'].set(value=1590, min=1560, max=1630)
pars['g_sigma'].set(value=79/2, min=20, max=80)
pars['g_amplitude'].set(value=50000, min=1000)

# 5. D' - Lorentzian (repositioned to 1520 cm-1)
lorentz_dp = LorentzianModel(prefix='dp_')
pars.update(lorentz_dp.make_params())
pars['dp_center'].set(value=1520, min=1510, max=1530)
pars['dp_sigma'].set(value=57/2, min=15, max=50)
pars['dp_amplitude'].set(value=30000, min=500)

print("Fitting...")
mod = lorentz_d_broad + lorentz_d1 + gauss_d3 + lorentz_g + lorentz_dp + bg
out = mod.fit(y, pars, x=x)

print("Fitting complete!")
print("\n" + out.fit_report())

vd = out.params.valuesdict()

print("\n" + "="*70)
print("FITTED PEAK PARAMETERS (750-2000 cm-1 region)")
print("="*70)

peaks = [
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
print(f"Fit Quality: R² = {1 - out.residual.var() / np.var(y):.6f}")
print(f"Chi-square: {out.chisqr:.2f}")
print("="*70)

# Ratios
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
print(f"Intensity Ratios (I = height):")
print(f"  I(D_broad)/I(G)    = {h_d_broad/h_g:.3f}")
print(f"  I(D1)/I(G)         = {h_d1/h_g:.3f}")
print(f"  I(D+D1)/I(G)       = {h_d_total/h_g:.3f}")
print(f"  I(D3)/I(G)         = {h_d3/h_g:.3f}")
print(f"  I(D')/I(G)         = {h_dp/h_g:.3f}")

print(f"\nArea Ratios (A = amplitude × FWHM):")
print(f"  A(D1)/A(G)         = {a_d1/a_g:.3f}")
print(f"  A(D+D1)/A(G)       = {a_d_total/a_g:.3f}")
print(f"  A(D')/A(G)         = {a_dp/a_g:.3f}")
print("="*70)

# Plot FULL spectrum with fit region highlighted
print("\nGenerating plots...")
fig, ax = plt.subplots(3, 1, figsize=(14, 12), dpi=300)

# Top: Full spectrum showing fit region
ax[0].plot(x_full, y_full, 'gray', alpha=0.3, lw=1, label='Full data')
ax[0].axvspan(x[0], x[-1], alpha=0.1, color='green', label='Fit region')
ax[0].plot(x, y, 'k.', alpha=0.4, markersize=2, label='Fit data')
ax[0].set_ylabel('Intensity', fontsize=12)
ax[0].legend(loc='upper right', fontsize=10)
ax[0].grid(True, alpha=0.3)
ax[0].set_title('Full Spectrum - 750-2000 cm-1 Region Selected', fontsize=12)

# Middle: Fit region detail
ax[1].plot(x, y, 'k.', alpha=0.4, markersize=2, label='Data')
ax[1].plot(x, out.best_fit, 'r-', lw=2.5, alpha=0.8, label='Total fit')

# Components
comps = out.eval_components(x=x)
colors = ['C2', 'C3', 'C4', 'C1', 'C5']
component_names = [
    ('d_broad_', 'D (broad)'),
    ('d1_', 'D1'),
    ('d3_', 'D3 (narrower)'),
    ('g_', 'G'),
    ('dp_', "D'")
]

for color, (prefix, name) in zip(colors, component_names):
    if prefix in comps:
        component = comps[prefix] + comps['lin_']
        ax[1].plot(x, component, '--', color=color, label=name, lw=2, alpha=0.6)
        ax[1].fill_between(x, component, comps['lin_'], facecolor=color, alpha=0.15)

ax[1].set_ylabel('Intensity', fontsize=12)
ax[1].set_title('Optimized Fit (750-2000 cm-1) - D3 Shifted Higher & Narrower', fontsize=12)
ax[1].legend(loc='upper right', fontsize=9, ncol=3)
ax[1].grid(True, alpha=0.3)
ax[1].set_xlim(x[0], x[-1])

# Bottom: Residuals
residual = out.best_fit - y
ax[2].plot(x, residual, 'C3-', alpha=0.7, lw=1.5, label='Residual')
ax[2].axhline(y=0, color='k', linestyle='--', alpha=0.5)
ax[2].fill_between(x, residual, 0, alpha=0.2, color='C3')
ax[2].set_xlabel('Raman shift [cm-1]', fontsize=12)
ax[2].set_ylabel('Residual', fontsize=12)
ax[2].set_title('Residuals (Data - Best Fit)', fontsize=12)
ax[2].grid(True, alpha=0.3)
ax[2].set_xlim(x[0], x[-1])

plt.tight_layout()
plt.savefig('output/ramfit_l405_optimized.png', dpi=150, bbox_inches='tight')
print("Plot saved: output/ramfit_l405_optimized.png")
plt.show()

print("\n" + "="*70)
print("FITTING COMPLETE - OPTIMIZED")
print("="*70)
