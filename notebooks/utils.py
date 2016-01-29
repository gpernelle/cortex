__author__ = 'G. Pernelle'
import matplotlib
import numpy as np
matplotlib.use('nbagg')
from scipy.optimize import curve_fit

import time as t

import pandas as pd
import pylab as plt
import re, csv, os, datetime, os, sys, cubehelix, multiprocessing, sh, subprocess, math
import blaze
from matplotlib import gridspec
from matplotlib.pyplot import cm 
from bokeh.plotting import figure, show, output_file
from matplotlib.mlab import psd
from scipy.misc import comb

cx4 = cubehelix.cmap(reverse=False, start=0., rot=0.5)
from joblib import Parallel, delayed  
num_cores = multiprocessing.cpu_count()

plt.style.use('ggplot')

matplotlib.rc('xtick', labelsize=12)
matplotlib.rc('ytick', labelsize=12)
matplotlib.rc('axes', labelsize = 12)
matplotlib.rc('axes', titlesize = 12)
matplotlib.rc('lines', linewidth=2)
matplotlib.rc('axes', facecolor = "white")

today = datetime.date.today()
todayStr = '%04d%02d%02d' % (today.year, today.month, today.day)
DIRECTORY = os.path.expanduser("~/Dropbox/0000 - PhD/figures/"+todayStr+"/")
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
