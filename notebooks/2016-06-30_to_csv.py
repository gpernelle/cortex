
# coding: utf-8

# In[1]:

from __future__ import division
from IO import *
from cycler import cycler
import matplotlib as mpl
get_ipython().magic('matplotlib inline')
from numba import autojit

get_ipython().magic('load_ext autoreload')
get_ipython().magic('autoreload 2')

today = datetime.date.today()
todayStr = '%04d%02d%02d' % (today.year, today.month, today.day)
# todayStr = '20151005'
DIRECTORY = os.path.expanduser("~/Dropbox/0000 - PhD/figures/"+todayStr+"/")
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)
    
from bokeh.io import output_notebook
output_notebook()


# In[2]:

get_ipython().run_cell_magic('javascript', '', 'IPython.OutputArea.auto_scroll_threshold = 9999;')


# In[3]:

# Instantiate classes
cortex = Cortex()
trn = TRN()
gr = GRAPH(cortex)

# styling
gammaColor = '#00cc99'


# In[7]:

# #-------------------------------
# # PARAMS SEARCH
# #-------------------------------
c = cortex
c.N = 400
c.g = 10
c.d1 = 10
c.d2 = 60000
c.after = 5000
c.d3 = 10
c.initTime()
c.sigma = 60
c.WII = 1400 #2800
c.S = 100
c.model = "gp-izh-subnetworks"
c.glob = 0
# shared weights
c.sG = 10
c.sWII = 10
c.FACT = 1
c.r=0
c.ratio = 15
c.LTD  = 1e-0*4.7e-6 * c.FACT * c.N
c.LTP = c.ratio * c.LTD
c.tauv=15
c.with_currents = True
c.dt =0.25


sWIIList = [0,10]
sGList = [0,1,5,10,15,20,25]
sGList = np.arange(0,30,2)
LTDList = [1e-9*4.7e-6 * c.FACT * c.N, 1e-0*4.7e-6 * c.FACT * c.N]
taulist = np.arange(15,95,2)

# # For testing
# c.N=20
# for c.tauv in taulist[:4]:
#     for c.d2 in [1000]:
#         for c.sWII in sWIIList:
#             for c.sG in sGList[:3]:
#                 for c.LTD in LTDList:b
#                     c.runSimulation()

FILENAME = DIRECTORY + "with_plast-ok.csv"

for c.tauv in taulist[:4]:
    for c.d2 in [60000]:
        for c.sWII in sWIIList[1:]:
            for c.sG in sGList[:3]:
                for c.LTD in LTDList:
                    c.LTP = c.ratio * c.LTD
                    print(c.tauv, c.sWII, c.sG)
                    Parallel(n_jobs=50)(delayed(c.readToCSV(filename=FILENAME))(i=0, tauv=c.tauv) for c.tauv in taulist)
                    c.readToCSV(filename="test.csv")


