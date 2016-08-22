__author__ = 'G. Pernelle'
import pathos.multiprocessing as mp
from pathos.multiprocessing import ProcessingPool as Pool
from pathos.multiprocessing import ThreadingPool as tPool
import matplotlib
import numpy as np
matplotlib.use('nbagg')
# matplotlib.use('Agg')
from scipy.optimize import curve_fit
from numba import autojit

import time as t

import pandas as pd
import pylab as plt
import ipyparallel as ipp
import re, csv, os, datetime, os, sys, cubehelix, sh, subprocess, math, socket
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

cx4 = cubehelix.cmap(reverse=False, start=0., rot=0.5)

plt.style.use('ggplot')

fontsize =14

matplotlib.rc('xtick', labelsize=fontsize)
matplotlib.rc('ytick', labelsize=fontsize)
matplotlib.rc('axes', labelsize = fontsize)
matplotlib.rc('axes', titlesize = fontsize)
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('axes', facecolor = "white")

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
