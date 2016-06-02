__author__ = 'G. Pernelle'
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
import re, csv, os, datetime, os, sys, cubehelix, multiprocessing, sh, subprocess, math, socket
from matplotlib import gridspec
from matplotlib.pyplot import cm
import matplotlib.colors as colors
from bokeh.plotting import figure, show, output_file, save
from matplotlib.mlab import psd
from scipy.misc import comb
from scipy import sparse, signal


cx4 = cubehelix.cmap(reverse=False, start=0., rot=0.5)
from joblib import Parallel, delayed  
num_cores = multiprocessing.cpu_count()

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
DIRECTORY = os.path.expanduser("~/Dropbox/0000 - PhD/figures/"+todayStr+"/")
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
