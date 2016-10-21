from IO import *
from cycler import cycler
import matplotlib as mpl
%matplotlib inline
from numba import autojit
from functionsTFhardbound import *

%load_ext autoreload
%autoreload 2

today = datetime.date.today()
todayStr = '%04d%02d%02d' % (today.year, today.month, today.day)
# todayStr = '20151005'
DIRECTORY = os.path.expanduser("~/Dropbox/0000_PhD/figures/"+todayStr+"/")
CSV_DIR_TODAY = os.path.expanduser("~/Dropbox/0000_PhD/csv/"+todayStr+"/")
CSV_DIR = os.path.expanduser("~/Dropbox/0000_PhD/csv/")
FIG_DIR = os.path.expanduser("~/Dropbox/0000_PhD/figures/")
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
    
from bokeh.io import output_notebook
output_notebook()
from IPython.display import clear_output, Image, display

PAPER = os.path.expanduser('~/Dropbox/ICL-2014/Presentations/2016-10-11-GJ-sync-paper/figures/')

sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

N, g, tauv, i, nu = 1000, 14,15,0,100
T = 10000

gpu = TfSingleNet(N=N,
                  T=T,
                  disp=False,
                  tauv=tauv,
                  device='/gpu:0',
                  spikeMonitor=False,
                  g0=g,
                  startPlast = 0,
                  nu = nu,
                  NUM_CORES = 1)
# gpu.input = apple
gpu.runTFSimul()

plt.plot(gpu.vvm)
plt.figure()
plt.plot(gpu.gamma)
