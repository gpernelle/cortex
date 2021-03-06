import fns
from fns import *
from fns.functionsTFhardbound import *


today = datetime.date.today()
todayStr = '%04d%02d%02d' % (today.year, today.month, today.day)
# todayStr = '20151005'
DIRECTORY = os.path.expanduser("~/Dropbox/0000_PhD/figures/" + todayStr + "/")
CSV_DIR_TODAY = os.path.expanduser("~/Dropbox/0000_PhD/csv/" + todayStr + "/")
CSV_DIR = os.path.expanduser("~/Dropbox/0000_PhD/csv/")
FIG_DIR = os.path.expanduser("~/Dropbox/0000_PhD/figures/")
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

PAPER = os.path.expanduser('~/Dropbox/ICL-2014/Presentations/2016-10-11-GJ-sync-paper/figures/')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})


# g = 20
# N = 100
# T = 1000
# nu = 100
# sG = 5
# tauv = 25
# gpu1 = TfConnEvolveNet(N=N,
#                   T=T,
#                   disp=False,
#                   tauv=tauv,
#                   device='/gpu:0',
#                   spikeMonitor=True,
#                   g0=g,
#                   startPlast = 100,
#                   nu = nu,
#                   NUM_CORES = 1,
#                   both=False,
#                  sG = sG,
#                       memfraction = 0.5
#                       )
# # gpu.input = apple
# gpu1.dt = 0.1
# gpu1.initWGap = False
# gpu1.profiling = False
# gpu1.g1 = 10
# gpu1.g2 = 10
# gpu1.stabTime = 10
# gpu1.debug = False
# gpu1.connectTime=int(T/2)
# gpu1.FACT = 200
# gpu1.ratio = 0.5
# gpu1.runTFSimul()

DEVICE = '/gpu:0'

T=1100
g=20
### input 1: apple
nu = 100
gpu1 = TfSingleNet(N=1000, T=T, disp=False, tauv=15, nu=nu, g0=g,
                   device=DEVICE, spikeMonitor=True, startPlast=999999)
gpu1.input = np.ones(T)*0
gpu1.initWGap = False
gpu1.dt=0.1
gpu1.startPlast=0
gpu1.runTFSimul()

plt.imshow(gpu1.raster.T)
plt.savefig('raster.pdf')