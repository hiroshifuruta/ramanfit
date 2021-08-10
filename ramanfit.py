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

import matplotlib.pyplot as plt
import numpy as np
import os

from lmfit import Model
from lmfit.lineshapes import lorentzian
from lmfit.models import LinearModel, LorentzianModel

from ipyfilechooser import FileChooser

# Create and displays a FileChooser widget
cwd = os.getcwd()
fc = FileChooser(cwd)
display(fc)

#INFILE = "20210726MJ_MWI_28ul_std-D1.txt"

INFILE = fc.selected


# with open(INFILE, "r") as f:
#     print(f.read())


data = np.loadtxt(INFILE, delimiter='\t')

print(data)


x = data[:,0]
y = data[:,1]


plt.plot(x,y);plt.show()


xDGindex1000=np.searchsorted(x,1000)
xDGindex2000=np.searchsorted(x,2000)

xDG = data[xDGindex1000:xDGindex2000,0]
yDG = data[xDGindex1000:xDGindex2000,1]

plt.plot(xDG,yDG);plt.show()


# LMFIT


bg = LinearModel(prefix='lin_')
pars = bg.guess(yDG, x=xDG)
#pars


lorentz1 = LorentzianModel(prefix='l1_')
#pars = lorentz1.guess(yDG, x=xDG)
pars.update(lorentz1.make_params())
pars['l1_center'].set(value=1336, min=1300, max=1380)
pars['l1_sigma'].set(value=10, min=5)
pars['l1_amplitude'].set(value=10000, min=5)
#pars



lorentz2 = LorentzianModel(prefix='l2_')
pars.update(lorentz2.make_params())

pars['l2_center'].set(value=1550, min=1500, max=1590)
pars['l2_sigma'].set(value=23, min=5)
pars['l2_amplitude'].set(value=15000, min=5)
#pars



lorentz3 = LorentzianModel(prefix='l3_')
pars.update(lorentz3.make_params())

pars['l3_center'].set(value=1603, min=1600, max=1620)
pars['l3_sigma'].set(value=10, min=5)
pars['l3_amplitude'].set(value=1000, min=5)
pars



mod = lorentz1 + lorentz2 + lorentz3 + bg
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

ax[0].set(xlabel="",ylabel="Residual [cps]")
ax[1].set(xlabel="",ylabel="Intensity [cps]")
ax[2].set(xlabel="Raman shift [cm-1]",ylabel="Intensity[cps]")
ax[1].legend(loc='best')
ax[2].legend(loc='best')

plt.savefig("ramfit.png",dpi=130)
plt.show()


for parname, param in out.params.items():
    print("%s = %f +/- %f " % (parname, param.value, param.stderr))


vd = out.params.valuesdict()
#vd


l2_area = np.pi * vd['l2_amplitude'] * vd['l2_fwhm']
l1_area = np.pi * vd['l1_amplitude'] * vd['l1_fwhm']
GDAreaRatio = l2_area / l1_area

GDAreaRatio


l1_height = vd['l1_height']
l2_height = vd['l2_height']
GDHeightRatio = l2_height / l1_height


GDHeightRatio





