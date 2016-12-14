__author__ = 'G. Pernelle'
# import pathos.multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
# from pathos.multiprocessing import ThreadingPool as tPool
import matplotlib as mpl
import numpy as np
mpl.use('Agg')
from scipy.optimize import curve_fit

from numba import autojit

import importlib as imp
import pandas as pd
import matplotlib.pyplot as plt
# import ipyparallel as ipp
import re, csv, os, datetime, os, sys, sh, math, socket, time
from scipy.fftpack import fft
from matplotlib import gridspec
from matplotlib.pyplot import cm
import matplotlib.colors as colors
from bokeh.plotting import figure, show, output_file, save
from matplotlib.mlab import psd
from scipy.misc import comb
from scipy import sparse, signal
import svgutils.transform as sg
import seaborn as sns
import json
from cycler import cycler
import matplotlib as mpl
from numba import autojit
import pyprind
import gc

# cx4 = cubehelix.cmap(reverse=False, start=0., rot=0.5)

plt.style.use('ggplot')

fontsize =14

mpl.rc('xtick', labelsize=fontsize)
mpl.rc('ytick', labelsize=fontsize)
mpl.rc('axes', labelsize = fontsize)
mpl.rc('axes', titlesize = fontsize)
mpl.rc('lines', linewidth=2)
mpl.rc('axes', facecolor="white")
mpl.rcParams['pdf.fonttype'] = 42

today = datetime.date.today()
todayStr = '%04d%02d%02d' % (today.year, today.month, today.day)
DIRECTORY = os.path.expanduser("~/Dropbox/0000_PhD/figures/"+todayStr+"/")
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

import warnings

def fxn():
    warnings.warn("deprecated", DeprecationWarning)

with warnings.catch_warnings():
    warnings.simplefilter("ignore")
    fxn()
