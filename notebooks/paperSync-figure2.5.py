
# coding: utf-8

# In[34]:

import fns
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
    
from IPython.display import clear_output, Image, display


# In[35]:

PAPER = os.path.expanduser('~/Dropbox/ICL-2014/Presentations/2016-10-11-GJ-sync-paper/figures/')


# In[36]:

plt.style.use(['seaborn-paper'])
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


# In[52]:

i=0
params = []
N = 1000
FACT = 3
T = 500000
spm = False
g = 5
ratio = 3
IAF = True
WII = -1000
WIE = -3000
WEE = 1000
WEI = 1000
inE = 100
k = 4

for nu in range(0, 100, 20):
    for step in [10,20,50,70]:
        i+=1
        params.append([T, FACT, N, g, i, nu, ratio, inE, WII, k, step])

nu=80
step=70
tauv = 15
IAF = True
spm = True
ratioNI = 0.2
NI = int(N*ratioNI)

filename = "../data/stimulations/stimulation0-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_k-%d_IAF-%d_inE-%d_step-%d"               % (tauv, g, N, T, nu, ratio,  WEE, WEI, WIE, WII, FACT, ratioNI, k, IAF*1, inE, step)
a = np.load(filename)

filename = "../data/stimulations/raster_stimulation0-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_k-%d_IAF-%d_inE-%d_step-%d"               % (tauv, g, N, T, nu, ratio,  WEE, WEI, WIE, WII, FACT, ratioNI, k, IAF*1, inE, step)
r = np.load(filename)
    


# In[46]:

def f(w=20,h=3):
    plt.figure(figsize=(w,h), linewidth=0.1)


# In[47]:

f()
plt.plot(a['vvmE'][10:])
plt.title('PSTH E')

f()
plt.plot(a['gamma']*NI)
plt.title('mean gamma')
# plt.figure()
# plt.plot(gpu.input)
# plt.title('Input current')

t0 = T//2 - 1000
t1 = T//2 + 1000

f()
plt.plot(a['vvmE'][t0:t1])
plt.title('PSTH E')

f()
plt.plot(a['gamma'][t0//100:t1//100])
plt.title('mean gamma')
# plt.figure()
# plt.plot(np.mean(np.array(gpu.lowsp).reshape(T,N).transpose(), axis=0))


# In[49]:

# print(r.shape)
# plt.imshow(r)


# In[ ]:



