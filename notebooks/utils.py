__author__ = 'G. Pernelle'
import matplotlib
import numpy as np
matplotlib.use('Agg')
from scipy.optimize import curve_fit

import time as t

import pandas as pd
import pylab as plt
import re, csv, os, datetime, os, sys, cubehelix, multiprocessing, sh, subprocess
import blaze
from matplotlib import gridspec
from matplotlib.pyplot import cm 
from bokeh.plotting import figure, show, output_file
from matplotlib.mlab import psd

cx4 = cubehelix.cmap(reverse=False, start=0., rot=0.5)
from joblib import Parallel, delayed  
num_cores = multiprocessing.cpu_count()

matplotlib.rc('xtick', labelsize=20) 
matplotlib.rc('ytick', labelsize=20) 
matplotlib.rc('axes', labelsize = 26)
matplotlib.rc('axes', titlesize = 26)

plt.style.use('ggplot')
