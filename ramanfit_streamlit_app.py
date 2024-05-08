#!/usr/bin/env python
# coding: utf-8

# <H3>Raman curve fit:</H3>
# curve fit with 3 lorentz peaks of G, D, G' for CSV text outputs obtained by LabRam HR-800 <br>
# 2021.08.10 ver.0.1 by fur.    curve fit with LMFIT and shows fitting curves.<br>
# 2021.08.10 ver.0.11 by fur.    Get G/D area ratio, and G/D height ratio <br>
# 2021.08.10 ver.0.12 by fur.   File chooser only works for jupyter notebook using ipyfilechooser <br>

# <H4>References:<H4>
# <OL>
#     <LI>LMFIT, https://lmfit.github.io/lmfit-py/</LI>
#     <LI>https://sabopy.com/py/lmfit-5/</LI>
#     <LI>Multi peak fitting, emilyripka, https://github.com/emilyripka/BlogRepo/blob/master/181119_PeakFitting.ipynb</LI>
# </OL>
import sys
import os
import streamlit as st    
import matplotlib.pyplot as plt
import numpy as np


st.title('Raman fit')

uploaded_file = st.file_uploader("Choose a Raman CSV file which holds 1000 - 2000 cm-1 data.")


if uploaded_file:
    INFILE = uploaded_file.name
    BASENAME = os.path.basename(INFILE)
    OUTPNGFILE = os.path.splitext(BASENAME)[0] + ".png"
    OUTCSVFILE = os.path.splitext(BASENAME)[0] + ".csv"
    data = np.loadtxt(uploaded_file, delimiter='\t')

from lmfit import Model
from lmfit.lineshapes import lorentzian
from lmfit.models import LinearModel, LorentzianModel

x = data[:,0]
y = data[:,1]

#print(data)

#plt.plot(x,y);plt.show()

xDGindex1000=np.searchsorted(x,1000)
xDGindex2000=np.searchsorted(x,2000)

xDG = data[xDGindex1000:xDGindex2000,0]
yDG = data[xDGindex1000:xDGindex2000,1]

fig = plt.figure()
plt.plot(xDG,yDG)
st.pyplot(fig)

# LMFIT

bg = LinearModel(prefix='lin_')
pars = bg.guess(yDG, x=xDG)
#pars

lorentz1 = LorentzianModel(prefix='l1_')  # D peak
#pars = lorentz1.guess(yDG, x=xDG)
pars.update(lorentz1.make_params())
pars['l1_center'].set(value=1336, min=1300, max=1380)
pars['l1_sigma'].set(value=10, min=5)
pars['l1_amplitude'].set(value=10000, min=5)
#pars

lorentz2 = LorentzianModel(prefix='l2_')  # G peak
pars.update(lorentz2.make_params())

pars['l2_center'].set(value=1550, min=1500, max=1590)
pars['l2_sigma'].set(value=23, min=5)
pars['l2_amplitude'].set(value=15000, min=5)
#pars

lorentz3 = LorentzianModel(prefix='l3_') # G' peak
pars.update(lorentz3.make_params())
pars['l3_center'].set(value=1603, min=1595, max=1620)
pars['l3_sigma'].set(value=10, min=5, max=100)
pars['l3_amplitude'].set(value=1000, min=5)
#pars

lorentz4 = LorentzianModel(prefix='l4_') # amorphous peak
pars.update(lorentz4.make_params())
pars['l4_center'].set(value=1500, min=1450, max=1520)
pars['l4_sigma'].set(value=10, min=5, max=100)
pars['l4_amplitude'].set(value=100, min=5, max=500)

mod = lorentz1 + lorentz2 + lorentz3 + lorentz4 + bg
init = mod.eval(pars, x=xDG)
out = mod.fit(yDG, pars, x=xDG)

print(out.fit_report())

fig, ax = plt.subplots(3,1,dpi=130)
ax=ax.ravel()

ax[0].plot(xDG, out.best_fit - yDG, 'C3-', alpha=0.5)

ax[1].plot(xDG, yDG, 'C1.',alpha=0.5)
ax[1].plot(xDG, out.best_fit, '-', label='best fit',zorder=10,lw=2, alpha=0.6)

ax[2].plot(xDG, yDG, 'C1.')
comps = out.eval_components(x=xDG)
ax[2].plot(xDG, comps['l1_']+comps['lin_'], 'C2--', label='Lorentzian compo. 1')
ax[2].fill_between(xDG, comps['l1_']+comps['lin_'], comps['lin_'],facecolor='C2',alpha=0.3)
ax[2].plot(xDG, comps['l2_']+comps['lin_'], 'C3--', label='Lorentzian compo. 2')
ax[2].fill_between(xDG, comps['l2_']+comps['lin_'], comps['lin_'],facecolor='C3',alpha=0.3)
ax[2].plot(xDG, comps['l3_']+comps['lin_'], 'C4--', label='Lorentzian compo. 3')
ax[2].fill_between(xDG, comps['l3_']+comps['lin_'], comps['lin_'],facecolor='C4',alpha=0.3)
ax[2].plot(xDG, comps['l4_']+comps['lin_'], 'C5--', label='Lorentzian compo. 4')
ax[2].fill_between(xDG, comps['l4_']+comps['lin_'], comps['lin_'],facecolor='C5',alpha=0.3)

ax[0].set(xlabel="",ylabel="Residual [cps]")
ax[1].set(xlabel="",ylabel="Intensity [cps]")
ax[2].set(xlabel="Raman shift [cm-1]",ylabel="Intensity[cps]")
ax[1].legend(loc='best', fontsize='x-small')
ax[2].legend(loc='best', fontsize='x-small')

plt.savefig(OUTPNGFILE,dpi=130)
#st.show()
st.pyplot(fig)


for parname, param in out.params.items():
    print("%s = %f +/- %f " % (parname, param.value, param.stderr))

od = out.params
vd = out.params.valuesdict()
#vd

print("##############################################")
print("FILE:\t", INFILE)
st.write("FILE:\t", INFILE)

l2_area = np.pi * vd['l2_amplitude'] * vd['l2_fwhm']
l1_area = np.pi * vd['l1_amplitude'] * vd['l1_fwhm']
GDAreaRatio = l2_area / l1_area

#print("G/D Area ratio:\t", GDAreaRatio)
#st.write("G/D Area ratio:\t", GDAreaRatio)

l1_height = vd['l1_height']
l2_height = vd['l2_height']
GDHeightRatio = l2_height / l1_height

l1_height_stderr=od['l1_height'].stderr
l2_height_stderr=od['l2_height'].stderr
GDHeightRatioMax = (l2_height + l2_height_stderr) / (l1_height - l1_height_stderr)
GDHeightRatioMin = (l2_height - l2_height_stderr) / (l1_height + l1_height_stderr)
GDHeightRatioPlus = GDHeightRatioMax - GDHeightRatio
GDHeightRatioMinus = GDHeightRatio - GDHeightRatioMin

print("G/D Height Ratio = %f +/- %f" % (GDHeightRatio, GDHeightRatioPlus))
st.write("G/D Height Ratio = %f +/- %f" % (GDHeightRatio, GDHeightRatioPlus))
print("G/D Area ratio:\t", GDAreaRatio)
st.write("G/D Area ratio:\t", GDAreaRatio)

with open(OUTCSVFILE, "w") as f:
    print("FILE:\t", INFILE, file=f)
    print("G/D Height ratio:\t", GDHeightRatio, file=f)
    print("G/D Area ratio:\t", GDAreaRatio, file=f)
    print("\n", file=f)
    print(out.fit_report(), file=f)

    



