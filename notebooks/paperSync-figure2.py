
# coding: utf-8

# In[2]:

import fns
from fns import *
from fns.functionsTF import *
get_ipython().magic('matplotlib inline')

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


# In[3]:

PAPER = os.path.expanduser('~/Dropbox/ICL-2014/Presentations/2016-10-11-GJ-sync-paper/figures/')


# In[4]:

# gr = Graph()


# In[5]:

plt.style.use(['seaborn-paper'])
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[6]:

def f(w=20,h=3):
    plt.figure(figsize=(w,h), linewidth=0.1)


# In[7]:

# look at lfp, oscillation, voltage at the end of the simulation? does the behavior change


# In[8]:

# ratio low, 2nd dyn
# global : 0.4-0.5
# local : 0.9


# In[9]:

gR=[]
ratio = 2
rNI=0.2
gGap=0
fast = 0
if fast:
    N=1000
    T=10000
else:
    N=1000
    T=800000
    FACT = 100

path = '../data/GetGammaSteadyState/GetSteadyState220'
c=['red', 'blue', 'black']
# f(8,3)
# ax = plt.subplot(111)
# for nu in np.arange(0,200,30):
#     for k,g in enumerate([10]):
#         WEE = 1000
#         WEI = 1000
#         WIE = -3000
#         WII = -1000
#         filepath = path + '-tauv-15_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d'%(g,N,T, nu,ratio, WEE, WEI, WIE, WII)
#         try:
#             a = np.load(filepath)
#             ax.plot(a['gamma'][:-2]*N*0.2, c=c[k],alpha=(nu+20)/200)
#             gR.append(a['gamma'])
#         except:
#             print('can\'t open %s'%filepath)
#     ax.set_title('g mean')
            


# In[12]:

def genPath(path, g,N,T, nu,ratio, WEE, WEI, WIE, WII, FACT, rNI, k, IAF, inE):
    fpath = path + '-tauv-15_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_k-%d_IAF-%d_inE-%d'%(g,N,T, 
                                                                                                                            nu,ratio, 
                                                                                                                            WEE, WEI, WIE, WII, 
                                                                                                                            FACT, rNI, k, IAF, inE)
#     if os.path.isfile(fpath):
#         print("exist")
    return fpath


g = 3
ratio = 10
rNI=0.2
gGap=0
fast = 0
IAF = 1
if fast:
    N=1000
    T=10000
else:
    N=1000
    T=50000
FACT = 50
WEE = 1000
WEI = 1000
WII = -1000
WIE = -2000
kmult = 4
inE = 100

path = '../data/GetGammaSteadyState/GetSteadyState300mean'
c=['red', 'blue', 'black']
f(8,6)
ax = plt.subplot(311)
for nu in np.arange(0,101,20):
    for k,g in enumerate([g]):
        filepath = genPath(path, g,N,T, nu,ratio, WEE, WEI, WIE, WII, FACT, rNI, kmult, IAF, inE)
        try:
            a = np.load(filepath)
            ax.plot(a['gamma'][:-2]*200, c=c[k],alpha=(nu+20)/200, label=nu)
            ax.legend()
        except:
            print('can\'t open %s'%filepath)
    ax.plot(a['gamma'][:-2]*0, c='g')
    ax.set_ylim([-0.001,None])
    ax.set_title('GJ mean')
            
            
# f(8,3)
ax = plt.subplot(312)
ax2 = plt.subplot(313)
nb = 30
l = 5000
for nu in np.arange(0,101,10):
    for k,g in enumerate([3]):
        filepath = genPath(path, g,N,T, nu,ratio, WEE, WEI, WIE, WII, FACT, rNI, kmult, IAF, inE)
        try:
            a = np.load(filepath)['vvmE']
            
            L = len(a)
            step = L/nb
            p=[]
            freq=[]
            for i in range(nb-1):
                s = i*step
                e = i*step + l
                if e<L:
                    p.append(fourier(a[s:e])[1])
                    freq.append(fourier(a[s:e])[0])
            ax.plot(p, c=c[k],alpha=(nu+20)/200)
            ax2.plot(freq, c=c[k],alpha=(nu+20)/200)
            ax2.set_ylim([0,200])
        except:
            print('can\'t open %s'%filepath)
    ax.set_title('power vmE')

plt.suptitle('global %d | ratioNI %.2f | FACT %d | IAF %d'%(gGap,rNI,FACT, IAF ),y=1.05)
plt.tight_layout()
plt.savefig(DIRECTORY + 'weight_evol-tauv-15_g-%d_N-%d_T-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_k-%d_IAF-%d_inE-%d.pdf'%(g,N,T,ratio, WEE, WEI, WIE, WII, FACT, rNI, kmult, IAF, inE), bbox_inches='tight')


# In[13]:

df = pd.DataFrame(columns=('nu', 'g', 'T', 'N', 'f', 'p', 'burst', 'spike', 'ratio'
                          ) )
i=-1
d = 2000
d2 = 7000
start = 5900
s0 = 100
sigma = 8
params=[]
inE = 100

for T in [8000]:
        for N in [1000]:
            for g in np.arange(0, 8,0.5):
                for nu in np.arange(0, 101, 5):
                        i+=1
                        params.append([T, N, g, nu, i, inE])

def getDF(params):
    wII = -1000
    wIE = -2000
    wEE = 1000
    wEI = 1000
    k=4
    T, N, g, nu, i, inE = params
    filename = "../data/PhasePlan7/PhasePlan290_nu-%d_g-%.2f_N-%d_k-%d_T-%d_WEE-%d_WEI-%d_WIE-%d_WII-%d_inE-%d" % (nu, g, N, k, T,
                                                                                                                       wEE, wEI, wIE, wII, inE)
    try:
        a = np.load(filename)
        I = a['vmE']

        # compute frequency and power with fourier transform of the lfp (mean current)
        f = fourier(I[10:])[0]
        p = fourier(I[10:])[1]

        return [i, int(nu), g, int(T), int(N), f, p, 
                     float(a['burst']), float(a['spike']), (a['burst']/a['spike'])]
    except:
        print(filename)
        return [i, int(nu), g, int(T), int(N), 0,0, 
                    0,0,0]

q = Pool(nodes=56)
re = q.amap(getDF, params)
res = re.get()
for r in res:
    df.loc[r[0]]=r[1:]


# In[14]:

df.head()


# In[15]:

cols = ['f', 'p']
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})
# df = df_[(df_['g']<3)&(df_['nu']<20)]
def setLabels(fig):
    color = '#FF6898'
    fig.text(2*2, 20, '#', fontweight='bold', fontsize=30, color=color)
    fig.text(7*2, 20, '*', fontweight='bold', fontsize=30, color=color)
    fig.set_ylabel(r'$\nu$')
    fig.set_xlabel(r'Gap-junction coupling')
    
    

fig = plotHeatmap(df, col='ratio', y='nu', x='g', title='Mean Activity Ratio (bursts/spikes)', cmap='viridis')
setLabels(fig)

g = 3
ratio = 10
rNI=0.2
gGap=0
fast = 0
IAF = 1
N=1000
T=50000
FACT = 50
WEE = 1000
WEI = 1000
WII = -1000
WIE = -2000
kmult = 4
inE = 100

path = '../data/GetGammaSteadyState/GetSteadyState300mean'

for nu in np.arange(0,100,10):
    for k,g in enumerate([g]):
        filepath = genPath(path, g,N,T, nu,ratio, WEE, WEI, WIE, WII, FACT, rNI, kmult, IAF, inE)
        try:
            a = np.load(filepath)
            plt.plot(np.mean(a['gamma'][-20:-2])*200*2,(100-nu)/5, 'o', c='w')
        except:
            print('can\'t open %s'%filepath)
    

fig = plotHeatmap(df, col='p', y='nu', x='g', rasterized=True,
                  title='Power of strongest Fourier component', cmap='RdBu_r')
setLabels(fig)
# plt.xlim([2*2,7*2])
plt.tight_layout()

for nu in np.arange(0,100,10):
    for k,g in enumerate([g]):
        filepath = genPath(path, g,N,T, nu,ratio, WEE, WEI, WIE, WII, FACT, rNI, kmult, IAF, inE)
        try:
            a = np.load(filepath)
            plt.plot(np.mean(a['gamma'][-20:-2])*200*2,(100-nu)/5, 'o', c='b')
        except:
            print('can\'t open %s'%filepath)


# gR=[]
# N=300
# ratio=30
# path = '../data/GetGammaSteadyState/GetSteadyState200'
# for k in np.arange(0,200,20):
#     a = np.load(path + '-tauv-15_g-5_N-%d_T-10000_nu-%d_ratio-%.2f'%(N,k,ratio))
#     gR.append(a['gamma'])
#     plt.plot(np.mean(gR[-100:-1])*N*0.2,(200-k)/5, 'o', color='w')
# plt.savefig(PAPER + 'ratio-steadystate.svg')



# ## Bursting protocol

# In[15]:

fig = plt.figure(figsize=(6,4.5))

class Izh:
    def __init__(self, T=3000, tauv=15):
        self.bTh = 1.5
        self.u_, self.v_, self.p_, self.lowsp_, self.vv_, self.gamma_ = [], [], [], [], [], []
        self.alphaLTD = 4.7e-6
        self.T = T
        self.tauv = tauv
        # stimulation
        self.I = [300 if i%500<50 else -82 for i in range(T)]
        
    def sim(self):
        dt = 0.25
        v = -70
        u, lowsp, p = 0, 0, 0
        gamma = 10
        for i in range(self.T):
            # voltage
            v = v + dt / self.tauv * ((v+60) * (v+50) - 20 * u + 8 * self.I[i])
            # adaptation
            u = u + dt * 0.044 * (v+55 - u)
            # spike
            vv = v> 25
            # reset
            v = vv * -40 + (1-vv)*v
            u = u + vv * 50
            # burst
            lowsp = lowsp + dt/8.0 * (vv * 8.0/dt - lowsp)
            p = lowsp>self.bTh
            gamma += p*(-self.alphaLTD)
            # save stuffs
            self.lowsp_.append(lowsp)
            self.p_.append(p)
            self.v_.append(v)
            self.u_.append(u)
            self.vv_.append(vv)
            self.gamma_.append(gamma)

izh = Izh(T=3000)
izh.sim()
T = izh.T
t = np.arange(T)
dt = 0.25

## PLOT FIGURE 2
ax1 = plt.subplot2grid((3, 5), (0, 0), colspan=3)
ax = plt.subplot2grid((3, 5), (1, 0), colspan=3)
ax2 = plt.subplot2grid((3, 5), (2, 0), colspan=3)
ax3 = plt.subplot2grid((3, 5), (1, 3), rowspan=2, colspan=2)
ax4 = plt.subplot2grid((3, 5), (0, 3), colspan=2)

# bursting neurons
bursting  = izh.p_

xlim=[0,3100]
##########################################
# V
##########################################
ax.plot(t, izh.v_, label='membrane voltage [mV]', color="#66ccff")

ax.set_xlim(xlim)
ax.set_ylim([-200, 80])
ax.set_yticks([-70,0])
ax.set_title("")
ax.set_xticklabels([])
ax.spines['right'].set_color('none')
ax.spines['top'].set_color('none')
ax.set_ylabel(r'$v_i$')
ax.set_axis_bgcolor((1, 1,1))
ax.xaxis.set_ticks_position('bottom')
ax.yaxis.set_ticks_position('left')

##########################################
# stim
##########################################
#     ax1.plot(s.t, ((s.stim - min(s.stim))/(max(s.stim) - min(s.stim)))*10, label='stimulation [pA]')
ax1.plot(t, izh.I, label='stimulation [pA]', color="#cc3333")

ax1.set_xlim(xlim)
ax1.set_ylim([-210,510])
ax1.set_yticks([-100,300])
ax1.set_axis_bgcolor((1, 1,1))
ax1.set_xticklabels([])
ax1.set_ylabel('Stim.')

# gamma
ax2.plot(t, np.array(izh.gamma_),"#66ccff")

ax2.set_xlim(xlim)

ax2.spines['right'].set_color('none')
ax2.spines['top'].set_color('none')
ax2.xaxis.set_ticks_position('bottom')
ax2.yaxis.set_ticks_position('left')
ax2.set_xlabel('Time [s]')
ax2.set_ylabel(r'$\gamma$')
ax2.set_axis_bgcolor((1, 1,1))
ax2.set_xticks([0,500,1000,2000,3000])
ax2.set_xticklabels([0,0.5,1,299,300])
ax2.set_yticks([10])
ax2.set_yticklabels('')


ax4.plot(t, izh.I, label='stimulation [pA]', color="#cc3333")
ax4.set_xlim([480,600])
ax4.set_ylim([-210,510])
ax4.set_yticks([])
ax4.set_xticks([500,550,600])
ax4.set_xticklabels([])
ax4.set_axis_bgcolor((1, 1,1))
ax4.set_xticklabels([])    

ax3.plot(t*dt, np.array(izh.lowsp_),"#66ccff", t*dt, np.ones(T)*izh.bTh, 'k--')
ax3.set_yticks([izh.bTh])
ax3.fill(t*dt, bursting*np.array(izh.lowsp_), facecolor='c', alpha=0.3,)
ax3.set_xlim([480,600])

ax3.spines['right'].set_color('none')
ax3.spines['top'].set_color('none')
ax3.xaxis.set_ticks_position('bottom')
ax3.yaxis.set_ticks_position('left')
ax3.set_xlabel('Time [ms]')
ax3.set_ylabel(r'$b_i$')
ax3.set_axis_bgcolor((1, 1,1))
ax3.set_xticks([500,550,600])
ax3.set_xticklabels([0,'',100])

ax1.set_title('Tonic bursting protocol', y=1.08)
plt.tight_layout()

# show()
plt.savefig(PAPER + 'fig2-bursting_protocol.svg', bbox_inches='tight', pad_inches=0)


# In[25]:

gpu = TfSingleNet(N=2,
                  T=1000,
                  disp=False,
                  tauv=15,
                  device='/cpu:0',
                  spikeMonitor=True,
                  g0=8,
                  startPlast = 0,
                  nu = 0,
                  NUM_CORES = 1)
gpu.ratio = 3
gpu.input = np.concatenate([np.ones(500)*0,np.ones(500)*200])
gpu.runTFSimul()


# In[26]:

dep = np.array(gpu.lowsp)[:,0]
plt.plot(dep)


# In[27]:

# # plt.imshow(gpu.raster.transpose())
# plt.figure()
spikes = gpu.raster.transpose()
# # plt.plot(spikes)

y = np.concatenate([spikes[i,:]*(i+1) for i in range(spikes.shape[0])])
x = list(np.arange(spikes.shape[1]))*spikes.shape[0]
plt.plot(x,y,'.')


# In[28]:

xlim = [1950,2050]
gammaColor = '#00cc99'
# P = gr.fourier(ssp1)
titlestr= 'Gap junction plasticity'

fig = plt.figure(figsize=(6,4.5))

N1 =[]
N2 = []
for i in range(len(x)):
    if y[i] == 2:
        N2.append([x[i],y[i]])
    elif y[i]==1:
        N1.append([x[i],y[i]])
N1 = np.array(N1)
N2 = np.array(N2)




### SPIKES
ax=fig.add_subplot(311)
ax.set_title(titlestr, y=1.08)
ax.plot(N1[:,0], N1[:,1], '|', color='#33ccff', markersize=14, markeredgewidth=2)
ax.plot(N2[:,0], N2[:,1]+0.5, '|', color='#6699cc', markersize=14,markeredgewidth=2)
# ax.set_xlim(xlim)
ax.set_ylim([-10,12])
ax.set_yticklabels([])

# stimulation
st=[]
for i in range((1000)):
    st.append(5 if i>500 else 4)
ax.plot(st)
# ax.set_xlim(xlim)
ax.set_xticks([])
ax.set_yticks([1,2.5,4.5])
ax.set_yticklabels(['N1', 'N2','I'])
ax.set_ylim([0.5,5.2])
ax.set_axis_bgcolor((1,1,1))

# # ax.set_ylim([-1,55])


### low sp threshold
ax = fig.add_subplot(312)
dep = np.array(gpu.lowsp)[:,0]
ax.plot(np.ones(len(dep))*1.5, 'k--')
ax.fill(np.arange(len(dep)), dep*(dep>1.3), facecolor='c', alpha=0.3,)
ax.plot(dep,"#66ccff")
ax.set_ylabel(r'$b_1$')
ax.set_xticks([])
ax.set_yticks([1.5])
ax.set_ylim([0,2.2])


### GAMMA
ax= fig.add_subplot(313)
# ax.set_xticklabels([])
# ax.set_xlim([1.95,2.05])
# ax.set_ylim([2.487,2.4876])
# ax.set_yticks([2.487,2.4875])
# ax.set_yticklabels(['2.487','2.4875'], fontsize=12)
ax.plot(gpu.gamma, color=gammaColor)
# ax.set_ylim([2.504,2.5052])
ax.set_ylabel(r'$\gamma_{12}$')
ax.set_xlabel('Time [ms]')
ax.set_yticks([])
# ax.set_xticks([1.950,2.050])
# ax.set_xticklabels([0,100])

plt.tight_layout()
plt.savefig(PAPER+'fig2-plasticity.svg')


# In[29]:

plt.plot(gpu.gamma)


# ## HAAS protocol

# In[16]:

izh = Izh(T=3000)
izh.sim()
T = izh.T
t = np.arange(T)
dt = 0.25


# In[27]:

def computeLTD(LTD, V0, burstime, dt):
    V = V0
    res_v = []
    for iteration in range(100):
        for i in range(len(izh.p_)):
            if izh.p_[i]:
                V -= LTD*V0*dt
            res_v.append(V)
    return V, res_v    


# In[28]:

def dichotomy(eps, goal, nbiter=20, dt=0.25):
    nbiter = 0
    err = 1
    res = 0.0010
    mmin = 0
    mmax = 0.0010
    LTD = 0.005
    g0 = 10
    i=0
    c = 0.005
    while(err > eps):
        i += 1
        c = (mmax-mmin)/2.000000000 + mmin
        res, _ = computeLTD(c, g0, izh.p_, dt)
        if res < goal:
            mmax = c
            sign = 1
        else:
            mmin = c
            sign = -1
        err = abs(goal- res)
        
    print(c) 
    return c


# In[29]:

# dt = 0.25
# bTh = 1.3 

# dt = 0.25
# bTh = 1.3 


# n = IzhNeuron("", a=0.02, b=0.25, c=-50, d=2, v0=-70)
# s = IzhSim(n, T=3000, dt=dt)
# for i, t in enumerate(s.t):
#     s.stim[i] = 300 if (t)%500<50 else -82
# sims.append(s)


# for i,s in enumerate(sims):
#     res = s.integrate()
    
# burstime = np.array(res[2]>bTh)


# In[30]:

LTD = dichotomy(1e-5, 10*0.87, dt)
_, res_v = computeLTD( LTD, 5.0, izh.p_, dt)


# In[32]:

_, res_v = computeLTD( LTD, 10.0, izh.p_*100, dt)
plt.plot(res_v[:4*5*60*1000])


# In[33]:

plt.plot(izh.v_)

plt.figure() 
plt.plot(izh.I)

plt.figure()
plt.plot(izh.vv_)

plt.figure()
plt.plot(izh.lowsp_)

plt.figure()
plt.plot(izh.p_)


# In[ ]:



