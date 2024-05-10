#!/usr/bin/env python
# coding: utf-8

# <H3>Raman curve fit:</H3>
# curve fit with 3 lorentz peaks of G, D, G' for CSV text outputs obtained by LabRam HR-800 <br>
# 2021.08.10 ver.0.1 by fur.    curve fit with LMFIT and shows fitting curves.<br>
# 2021.08.10 ver.0.11 by fur.    Get G/D area ratio, and G/D height ratio <br>
# 2021.08.10 ver.0.12 by fur.   File chooser only works for jupyter notebook using ipyfilechooser <br>
#
# 2024.05.08 streamlit_app ver.0.1 by fur.   ramanfit_streamlit_app.py

# References:
#     LMFIT, https://lmfit.github.io/lmfit-py/
#     https://sabopy.com/py/lmfit-5/
#     Multi peak fitting, emilyripka, https://github.com/emilyripka/BlogRepo/blob/master/181119_PeakFitting.ipynb 

import sys
import os
from datetime import datetime

import streamlit as st    
import matplotlib.pyplot as plt
#plt.rcParams['xtick.direction'] = 'in'
#plt.rcParams['ytick.direction'] = 'in'
import numpy as np

from lmfit import Model
from lmfit.lineshapes import lorentzian
from lmfit.models import LinearModel, LorentzianModel

st.title('Raman fit with lmfit')

uploaded_file = st.file_uploader("Choose a Raman CSV file which holds 1000 - 2000 cm-1 data.")

if uploaded_file is None:
    st.write("Click Browse files button to upload file")

if not os.path.exists('data'):
    os.mkdir('data')



if uploaded_file is not None:
    INFILE = uploaded_file.name
    BASENAME = os.path.basename(INFILE)
    DATAFOLDER = "data"
    OUTPNGFILE = os.path.splitext(BASENAME)[0] + ".png"
    OUTCSVFILE = os.path.splitext(BASENAME)[0] + ".csv"
    
    print(INFILE)
    #print(BASENAME)
    #print(DATAFOLDER)
    #print(OUTPNGFILE)
    #print(OUTCSVFILE) 
 
    try:
        data = np.loadtxt(uploaded_file, delimiter='\t')
        st.write("Data loaded.")

        if data is not None:
            try:
                x = data[:,0]
                y = data[:,1]
            except IndexError as e:
                st.error(f"Data format error: {e}")      
    except Exception as e:
        st.error(f"Error loading data: {e}")
        data = None

if uploaded_file is not None:
    fig, ax = plt.subplots()
    plt.plot(x,y)
    ax.set(xlabel="Raman shift [cm-1]",ylabel="Intensity[cps]")
    ax.minorticks_on()
    ax.xaxis.set_tick_params(which='minor', bottom=True)
    st.pyplot(fig)

    xDGindex1000=np.searchsorted(x,1000)
    xDGindex2000=np.searchsorted(x,2000)

    xDG = data[xDGindex1000:xDGindex2000,0]
    yDG = data[xDGindex1000:xDGindex2000,1]
#if data is not None and analyze_button is None:
#    st.write("Click Analyze button to start analysis.")

analyze_button = None

if st.button("Analyze"):
    analyze_button=1
# LMFIT
    LMFIT_TIME = datetime.now() # Make a time stamp of processing

    bg = LinearModel(prefix='lin_')
    pars = bg.guess(yDG, x=xDG)

    lorentz1 = LorentzianModel(prefix='l1_')  # D peak
    pars.update(lorentz1.make_params())
    pars['l1_center'].set(value=1336, min=1300, max=1380)
    pars['l1_sigma'].set(value=10, min=5)
    pars['l1_amplitude'].set(value=10000, min=5)

    lorentz2 = LorentzianModel(prefix='l2_')  # G peak
    pars.update(lorentz2.make_params())
    pars['l2_center'].set(value=1550, min=1500, max=1590)
    pars['l2_sigma'].set(value=15, min=3)
    pars['l2_amplitude'].set(value=20000, min=5)

    lorentz3 = LorentzianModel(prefix='l3_') # G' peak
    pars.update(lorentz3.make_params())
    pars['l3_center'].set(value=1615, min=1600, max=1630)
    pars['l3_sigma'].set(value=3, min=2, max=20)
    pars['l3_amplitude'].set(value=20, min=2, max=1000)

    lorentz4 = LorentzianModel(prefix='l4_') # amorphous peak
    pars.update(lorentz4.make_params())
    pars['l4_center'].set(value=1480, min=1440, max=1520)
    pars['l4_sigma'].set(value=6, min=1, max=100)
    pars['l4_amplitude'].set(value=200, min=2, max=2000)

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

    ax[0].minorticks_on()
    ax[1].minorticks_on()
    ax[2].minorticks_on()
    #ax[2].xaxis.set_tick_params(which='minor', bottom=True)

    ax[0].set(xlabel="",ylabel="Residual [cps]")
    ax[1].set(xlabel="",ylabel="Intensity [cps]")
    ax[2].set(xlabel="Raman shift [cm-1]",ylabel="Intensity[cps]")
    ax[1].legend(loc='best', fontsize='x-small')
    ax[2].legend(loc='best', fontsize='x-small')

    plt.savefig(DATAFOLDER+"/"+OUTPNGFILE,dpi=130)
    #st.show()
    st.pyplot(fig)

if analyze_button is not None:
    for parname, param in out.params.items():
        print("%s = %f +/- %f " % (parname, param.value, param.stderr))

    od = out.params
    vd = out.params.valuesdict()

    print("##############################################")
    print("FILE:\t", INFILE)
    st.write("FILE:\t", INFILE)

    print("Processed (UTC):\t", LMFIT_TIME)
    st.write("Processed (UTC):\t", LMFIT_TIME)

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

    print("G/D Height Ratio:\t",  GDHeightRatio, "+/-", GDHeightRatioPlus)
    st.write("G/D Height Ratio:\t",  GDHeightRatio, "+/-", GDHeightRatioPlus)
    print("G/D Area ratio:\t", GDAreaRatio)
    st.write("G/D Area ratio:\t", GDAreaRatio)

    with open(DATAFOLDER+"/"+OUTCSVFILE, "w") as f:
        print("FILE:\t", INFILE, file=f)
        print("Processed (UTC):\t", LMFIT_TIME, file=f)   
        print("G/D Height ratio:\t", GDHeightRatio, file=f)
        print("G/D Area ratio:\t", GDAreaRatio, file=f)
        print("\n", file=f)
        print(out.fit_report(), file=f)
    
    
#@st.cache_data
#def convert_df(df):
    # IMPORTANT: Cache the conversion to prevent computation on every rerun
#    return df.to_csv().encode("utf-8")
if analyze_button is not None:
    timestr = LMFIT_TIME.strftime("%Y%m%d_%H%M")
    DLCSVFILE = os.path.splitext(BASENAME)[0] + "_" + timestr + ".csv"
    #csv = convert_df(f)
    with open(DATAFOLDER+"/"+OUTCSVFILE, "rb") as f:
        btn = st.download_button(
                label="Download results as CSV",
                data=f,
                file_name=DATAFOLDER+"/"+DLCSVFILE,
                mime="text/csv",
            )

    



