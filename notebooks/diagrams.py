
# coding: utf-8

# <h1 id="tocheading">Table of Contents</h1>
# <div id="toc"></div>

# In[1]:



# In[2]:


from __future__ import division
from IO import *
from cycler import cycler
import matplotlib as mpl
from numba import autojit

import argparse
parser = argparse.ArgumentParser()

#-db DATABSE -u USERNAME -p PASSWORD -size 20
parser.add_argument("-sWII", "--sWII", help="shared Chemical Synapses", type=int)
parser.add_argument("-N", "--N", help="number of neurons", type=int)

args = parser.parse_args()


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


# In[4]:

# Instantiate classes
cortex = Cortex()
trn = TRN()
gr = GRAPH(cortex)


# In[5]:

# #-------------------------------
# # PARAMS SEARCH
# #-------------------------------
c = cortex
c.N = args.N
c.g = 10
c.d1 = 10
c.d2 = 30000
c.d3 = 10
c.initTime()
c.sigma = 60
c.WII = 1400 #2800
c.S = 100
c.model = "gp-izh-subnetworks"
c.glob = 0
c.plast = 1

# shared weights
c.sG = 10
c.sWII = args.sWII

c.FACT = 1
c.r=0

c.ratio = 15
c.LTD  = 1e-0*4.7e-6 * c.FACT * c.N
c.LTP = c.ratio * c.LTD
c.dt = 0.25
c.tauv=15

c.with_currents = True

sWIIList = [0,10]
sGList = [0,1,5,10,15,20,25]
sGList = np.arange(0,25,4)
LTDList = [1e-9*4.7e-6 * c.FACT * c.N, 1e-0*4.7e-6 * c.FACT * c.N]
taulist = np.arange(11,95,4)


# -1   1
# 
# -2   2

# ## READ CSV DATA

# In[6]:

# df = pd.read_csv('/Users/GP1514/Dropbox/0000 - PhD/figures/20160628/' + 'df-plast-ok2.csv')
columns = ['tauv', 'd2','sWII', 'sG', 'LTD',
                             'maxBegin', 'argmaxBegin', 'maxEnd', 'argmaxEnd',
                            'f1Begin', 'p1Begin', 'f2Begin', 'p2Begin', 
                             'f1End', 'p1End', 'f2End', 'p2End', 
                             'fBothBegin', 'pBothBegin', 'fBothEnd', 'pBothEnd',
                               '_gn1', '_gn2', 'gnshared_', '_initmean',
                               '_gnmean1', '_gnmean2', 'gnmeanshared_',
                                'gn1', 'gn2', 'gnshared',
                               'gnmean1', 'gnmean2', 'gnmeanshared',
                           'key']
# df2 = pd.read_csv(CSV_DIR + 'with_plast-ok-t30-N100.csv', names=columns)
day = 20160701
sWII = args.sWII
c.sWII = sWII
df2 = pd.read_csv('%s%s/with_plast-ok-t30-N100.csv' % (CSV_DIR, day), names=columns)
day = 20160706
if c.N == 400:
    df = pd.read_csv('%s%s/with_plast-T30-N400-ok_1467798040.csv' % (CSV_DIR, day), names=columns)
else:
    df = pd.read_csv('%s%s/with_plast-T30-N100-ok_1467798950.csv' % (CSV_DIR, day), names=columns)
    

df.columns


# In[7]:

# df2.sort(columns=['tauv','sWII','sG','LTD']).head()


# In[8]:

# df.sort(columns=['tauv','sWII','sG','LTD']).head()


# In[9]:

N = c.N
df = df[(df['tauv']>13) & (df['sWII']==sWII) ]
dfplast = df[(df['LTD']==True)]
dfnoplast = df[(df['LTD']==False)]
dfplast.head()


# ## GAMMA

# In[33]:

gr = GRAPH(cortex)
# gr.plotGamma(dfnoplast, 'gn')
gr.plotGamma(dfnoplast, '_gnmean')
gr.plotGamma(dfplast, 'gnmean')
svg2pdf('gnmean_gamma_cluster_%d_%d'%(sWII,N),DIRECTORY)


# # Subnetwork Frequencies

# In[34]:

gr= GRAPH(cortex)
gr.plotEvolution(dfplast, kind = 'frequency', LTD=True, vmin = 25, vmax=60, sWII = sWII)
gr.plotEvolution(dfnoplast, kind = 'frequency', LTD=False, vmin = 25 ,vmax=60, sWII=sWII)


# In[35]:

gr=GRAPH(cortex)
gr.plotEvolution(dfplast, kind='power', LTD=True, sWII=sWII)
gr.plotEvolution(dfnoplast, kind='power', LTD=False, sWII=sWII)


# In[13]:

gr = GRAPH(cortex)
gr.plotCoherenceEvolution(df, kind='max', LTD=True, sWII=sWII)
gr.plotCoherenceEvolution(df, kind='max', LTD=False, sWII=sWII)


# In[14]:

gr.plotCoherenceEvolution(df, kind='argmax', LTD=True,sWII=sWII)
gr.plotCoherenceEvolution(df, kind='argmax', LTD=False,sWII=sWII)


# In[15]:

gr = GRAPH(cortex)
gr.plotCoherenceEvolution(df, kind='fBoth', LTD=True, sWII = sWII, 
                         vmax=dfplast['fBothEnd'].max(), vmin=dfplast['fBothEnd'].min())
gr.plotCoherenceEvolution(df, kind='fBoth', LTD=False, sWII = sWII,
                          vmax=dfplast['fBothEnd'].max(), vmin=dfplast['fBothEnd'].min())

gr.plotCoherenceEvolution(df, kind='pBoth', LTD=True, sWII = sWII,
                         vmax=dfplast['pBothEnd'].max(), vmin=dfplast['pBothEnd'].min())
gr.plotCoherenceEvolution(df, kind='pBoth', LTD=False, sWII = sWII, 
                          vmax=dfplast['pBothEnd'].max(), vmin=dfplast['pBothEnd'].min())


# In[ ]:




# In[16]:

gr = GRAPH(cortex)
gr.plotChange2(df, 'f', 'N1', both=True, title='Change of oscillation frequency', sWII=sWII, LTD=True)
gr.plotChange2(df, 'p', 'N1', both=True, title='Change of subnetwork coherence', sWII=sWII)
# plotChange('f1', 'N1')
# plotChange('f1', 'N1', both=True)
# plotChange('f2', 'N2', both=True)
# plotChange('p1', 'N1', both=True)
# plotChange('p2', 'N2', both=True)


# ## CSD BEFORE AFTER

# In[17]:

gr = GRAPH(cortex)
# power
gr.plotChange(df, 'max', title="Change of coherence between both networks", sWII=sWII, LTD=True)
gr.plotChange(df, 'max', title="Change of coherence between both networks", sWII=sWII, LTD=False)
# freq
# gr.plotChange(df, 'argmax', title="Change of frequency between both networks", sWII=sWII, LTD=True)
# gr.plotChange(df, 'argmax', title="Change of frequency between both networks", sWII=sWII, LTD=False)


# ## FOURIER (i1+i2) BEFORE AFTER

# In[18]:

# power
gr.plotChange(df, 'pBoth', title="Change of full internetwork coherence_plast", LTD=True, sWII=sWII)
gr.plotChange(df, 'pBoth', title="Change of full internetwork coherence_noplast", LTD=False, sWII =sWII)
# freq
gr.plotChange(df, 'fBoth', title="Change of full internetwork frequency_plast", LTD=True, sWII = sWII)
gr.plotChange(df, 'fBoth', title="Change of full internetwork frequency_noplast", LTD=False, sWII = sWII)


# ## Plot difference between with and without plasticity

# In[36]:

gr.plotChangePlast2(df, 'f', 'N1', both=True, title='f_changeplast', sWII=sWII)
plt.savefig(DIRECTORY+'f_changeplast_%d_%d.svg'%(sWII,N))
plt.figure()
gr.plotChangePlast2(df, 'p', 'N1', both=True, title='p_changeplast', sWII=sWII)
plt.savefig(DIRECTORY+'p_changeplast_%d_%d.svg'%(sWII,N))


# In[20]:

gr = GRAPH(cortex)
# gr.plotChangePlast(df, 'max', title='Change of oscillation frequency', sWII=10, vmax=1.3)
# gr.plotChangePlast(df, 'argmax', title='Change of subnetwork coherence', sWII=10)
gr.plotChangePlast(df, 'fBoth', title='Change of oscillation frequency fBoth', sWII=sWII)
# plt.savefig(DIRECTORY+'change_of_fBoth_plast_vs_noplast.pdf')
plt.figure()
gr.plotChangePlast(df, 'pBoth', title='Change of subnetwork coherence pBoth', sWII=sWII)
# plt.savefig(DIRECTORY+'change_of_pBoth_plast_vs_noplast.pdf')


# In[21]:

# # c = cortex
# c.with_currents = True
# c.N = 300
# c.g = 10
# c.d1 = 10
# c.d2 = 1012
# c.d3 = 10
# c.T = 1020
# c.tauv = 15
# c.sG=20
# c.sWII=0
# c.dt=0.25
# # c.runSimulation(tauv=c.tauv)
# # c.readSimulation()
# # gr.plotWeights(tauv=c.tauv)


# In[22]:

# t0 = t.time()

# c.runSimulation(tauv=c.tauv)
# print(t.time()-t0)


# In[23]:

# c.readSimulation()
# plt.plot(c.voltage)


# In[24]:

# c.dt = 0.25
# c.N = 400
# c.g = 10
# c.d1 = 10
# c.d2 = 30000
# c.d3 = 10
# c.initTime()
# c.sigma = 60
# c.WII = 1400 #2800
# c.S = 100
# c.model = "gp-izh-subnetworks"
# c.glob = 0

# # shared weights
# c.sG = 10
# c.sWII = 10

# c.FACT = 1
# c.r=0

# c.ratio = 15
# c.LTD  = 1e-0*4.7e-6 * c.FACT * c.N
# c.LTP = c.ratio * c.LTD

# c.tauv=15

# c.with_currents = True

# # for S in np.arange(50,400,50):
# #     Parallel(n_jobs=8)(delayed(cortex.runSimulation)(N,0, G, S, d1, d2, d3, before, after, sigma, \
# # WII, LTP, LTD) for G in np.linspace(1,9,"8))

# sWIIList = [0,10]
# sGList = [0,1,5,10,15,20,25]
# sGList = np.arange(0,25,4)
# LTDList = [1e-9*4.7e-6 * c.FACT * c.N, 1e-0*4.7e-6 * c.FACT * c.N]
# taulist = np.arange(11,95,4)
# c.LTD = LTDList[0]
# c.sG = sGList[0]
# c.LTP = c.ratio * c.LTD
# c.readSimulation()


# In[25]:

# dfnoplast['fBothEnd'].var()


# In[26]:

# c.d2 = 10000
# c.N = 50
# c.after = 1000
# c.dt=0.25
# c.LTD = LTDList[1]*10
# c.LTP = c.LTD * c.ratio
# c.T = c.d1+c.d2+c.d3
# c.runSimulation()


# In[27]:

# c.readSimulation()
# c.gammaN1[-10:]


# ## PLOT DIAGRAM + PLAST COMPARISON FOR COHERENCE

# In[28]:



#create new SVG figure
fig = sg.SVGFigure("20cm", "7cm")

# load matpotlib-generated figures
fig1 = sg.fromfile(FIG_DIR + str(todayStr) + '/power_clusterTrue_%d_%d.svg'%(sWII,N))
fig2 = sg.fromfile(FIG_DIR + str(todayStr) + '/power_clusterFalse_%d_%d.svg'%(sWII,N))
fig3 = sg.fromfile(FIG_DIR + str(todayStr) + '/p_changeplast_%d_%d.svg'%(sWII,N))

# get the plot objects
scale = 0.45
plot1 = fig1.getroot()
plot1.moveto(0,10,scale= scale)
plot2 = fig2.getroot()
plot2.moveto(280, 10, scale = scale)
plot3 = fig3.getroot()
plot3.moveto(560, 20, scale = scale*0.95)

# add text labels
txt1 = sg.TextElement(5,30, "A", size=12, weight="bold")
plast = sg.TextElement(75,10, "Plasticity", size=12, weight="normal")
noplast = sg.TextElement(350,10, "No plasticity", size=12, weight="normal")
txt2 = sg.TextElement(285,30, "B", size=12, weight="bold")
txt3 = sg.TextElement(560,30, "C", size=12, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2, plot3])
fig.append([txt1, txt2, txt3, plast, noplast])

# save generated SVG files
fig.save(DIRECTORY + "powercluster_%d_%d.svg"%(sWII,N))
svg2pdf('powercluster_%d_%d'%(sWII,N),DIRECTORY)



# ## PLOT DIAGRAM + PLAST COMPARISON FOR FREQUENCY

# In[29]:

#create new SVG figure
fig = sg.SVGFigure("20cm", "10cm")

# load matpotlib-generated figures
fig1 = sg.fromfile(FIG_DIR + str(todayStr) + '/frequency_clusterTrue_%d_%d.svg'%(sWII,N))
fig2 = sg.fromfile(FIG_DIR + str(todayStr) + '/frequency_clusterFalse_%d_%d.svg'%(sWII,N))
fig3 = sg.fromfile(FIG_DIR + str(todayStr) + '/f_changeplast_%d_%d.svg'%(sWII,N))

# get the plot objects
scale = 0.45
plot1 = fig1.getroot()
plot1.moveto(0,10,scale= scale)
plot2 = fig2.getroot()
plot2.moveto(280, 10, scale = scale)
plot3 = fig3.getroot()
plot3.moveto(560, 20, scale = scale*0.95)

# add text labels
txt1 = sg.TextElement(5,30, "A", size=12, weight="bold")
plast = sg.TextElement(75,10, "Plasticity", size=12, weight="normal")
noplast = sg.TextElement(350,10, "No plasticity", size=12, weight="normal")
txt2 = sg.TextElement(285,30, "B", size=12, weight="bold")
txt3 = sg.TextElement(560,30, "C", size=12, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2, plot3])
fig.append([txt1, txt2, txt3, plast, noplast])

# save generated SVG files
fig.save(DIRECTORY + 'freqcluster_%d_%d.svg'%(sWII,N))
svg2pdf('freqcluster_%d_%d'%(sWII,N),DIRECTORY)


# ## PLOT DIAGRAM + PLAST COMPARISON FOR FREQUENCY : INTERNETWORK

# In[30]:

#create new SVG figure
fig = sg.SVGFigure("20cm", "15cm")

# load matpotlib-generated figures
fig1 = sg.fromfile(FIG_DIR + str(todayStr) + '/fBoth_cluster_plastFalse_%d_%d.svg'%(sWII,N))
fig2 = sg.fromfile(FIG_DIR + str(todayStr) + '/fBoth_cluster_plastTrue_%d_%d.svg'%(sWII,N))
fig3 = sg.fromfile(FIG_DIR + str(todayStr) + '/fBothChange_of_full_internetwork_frequency_noplast_%d_%d.svg'%(sWII,N))
fig4 = sg.fromfile(FIG_DIR + str(todayStr) + '/fBothChange_of_full_internetwork_frequency_plast_%d_%d.svg'%(sWII,N))
fig5 = sg.fromfile(FIG_DIR + str(todayStr) + '/fboth_changePLAST_cluster_%d_%d.svg'%(sWII,N))


# get the plot objects
scale = 0.45
plot1 = fig1.getroot()
plot1.moveto(0,0,scale= scale)
plot2 = fig2.getroot()
plot2.moveto(0, 180, scale = scale)
plot3 = fig3.getroot()
plot3.moveto(350, 0, scale = scale*0.85)
plot4 = fig4.getroot()
plot4.moveto(350, 180, scale = scale*0.85)
plot5 = fig5.getroot()
plot5.moveto(170, 340, scale = scale*0.85)

# add text labels
# txt1 = sg.TextElement(5,20, "A", size=12, weight="bold")
# txt2 = sg.TextElement(285,20, "B", size=12, weight="bold")
# txt3 = sg.TextElement(560,20, "C", size=12, weight="bold")
txt1 = sg.TextElement(10,130, "No plasticity", size=12, weight="bold")
txt1.rotate(-90, 10,130)
txt2 = sg.TextElement(10,310, "Plasticity", size=12, weight="bold")
txt2.rotate(-90, 10,310)

# append plots and labels to figure
fig.append([plot1, plot2, plot3, plot4, plot5])
fig.append([txt1, txt2])

# save generated SVG files
fig.save(DIRECTORY + 'freq_change_%d_%d.svg'%(sWII,N))
svg2pdf('freq_change_%d_%d'%(sWII, N),DIRECTORY)


# ## PLOT DIAGRAM + PLAST COMPARISON FOR POWER : INTERNETWORK

# In[31]:

#create new SVG figure
fig = sg.SVGFigure("14cm", "14cm")

# load matpotlib-generated figures
fig1 = sg.fromfile(FIG_DIR + str(todayStr) + '/pBoth_cluster_plastFalse_%d_%d.svg'%(sWII,N))
fig2 = sg.fromfile(FIG_DIR + str(todayStr) + '/pBoth_cluster_plastTrue_%d_%d.svg'%(sWII,N))
fig3 = sg.fromfile(FIG_DIR + str(todayStr) + '/pBothChange_of_full_internetwork_coherence_noplast_%d_%d.svg'%(sWII,N))
fig4 = sg.fromfile(FIG_DIR + str(todayStr) + '/pBothChange_of_full_internetwork_coherence_plast_%d_%d.svg'%(sWII,N))
fig5 = sg.fromfile(FIG_DIR + str(todayStr) + '/pboth_changePLAST_cluster_%d_%d.svg'%(sWII,N))


# get the plot objects
scale = 0.45
plot1 = fig1.getroot()
plot1.moveto(0,0,scale= scale)
plot2 = fig2.getroot()
plot2.moveto(0, 180, scale = scale)
plot3 = fig3.getroot()
plot3.moveto(350, 0, scale = scale*0.85)
plot4 = fig4.getroot()
plot4.moveto(350, 180, scale = scale*0.85)
plot5 = fig5.getroot()
plot5.moveto(170, 340, scale = scale*0.85)

# add text labels
# txt1 = sg.TextElement(5,20, "A", size=12, weight="bold")
# txt2 = sg.TextElement(285,20, "B", size=12, weight="bold")
# txt3 = sg.TextElement(560,20, "C", size=12, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2, plot3, plot4, plot5])
# fig.append([txt1, txt2, txt3])

# save generated SVG files
fig.save(DIRECTORY + 'power_change_%d_%d.svg'%(sWII,N))
svg2pdf('power_change_%d_%d'%(sWII,N),DIRECTORY)


# ## PLOT CSD PLAST EFFECT

# In[32]:


#create new SVG figure
fig = sg.SVGFigure("14cm", "14cm")

# load matpotlib-generated figures
fig1 = sg.fromfile(FIG_DIR + str(todayStr) + '/pBoth_cluster_plastFalse_%d_%d.svg'%(sWII,N))
fig2 = sg.fromfile(FIG_DIR + str(todayStr) + '/pBoth_cluster_plastTrue_%d_%d.svg'%(sWII,N))
fig3 = sg.fromfile(FIG_DIR + str(todayStr) + '/pBothChange_of_full_internetwork_coherence_noplast_%d_%d.svg'%(sWII,N))
fig4 = sg.fromfile(FIG_DIR + str(todayStr) + '/pBothChange_of_full_internetwork_coherence_plast_%d_%d.svg'%(sWII,N))
fig5 = sg.fromfile(FIG_DIR + str(todayStr) + '/pboth_changePLAST_cluster_%d_%d.svg'%(sWII,N))


# get the plot objects
scale = 0.45
plot1 = fig1.getroot()
plot1.moveto(0,0,scale= scale)
plot2 = fig2.getroot()
plot2.moveto(0, 180, scale = scale)
plot3 = fig3.getroot()
plot3.moveto(350, 0, scale = scale*0.85)
plot4 = fig4.getroot()
plot4.moveto(350, 180, scale = scale*0.85)
plot5 = fig5.getroot()
plot5.moveto(170, 340, scale = scale*0.85)

# add text labels
txt1 = sg.TextElement(10,130, "No plasticity", size=12, weight="bold")
txt1.rotate(-90, 10,130)
txt2 = sg.TextElement(10,310, "Plasticity", size=12, weight="bold")
txt2.rotate(-90, 10,310)
# txt2 = sg.TextElement(285,20, "B", size=12, weight="bold")
# txt3 = sg.TextElement(560,20, "C", size=12, weight="bold")

# append plots and labels to figure
fig.append([plot1, plot2, plot3, plot4, plot5])
fig.append([txt1, txt2])

# save generated SVG files
fig.save(DIRECTORY + 'power_change_%d_%d.svg'%(sWII,N))
svg2pdf('power_change_%d_%d'%(sWII,N),DIRECTORY)


# In[ ]:



