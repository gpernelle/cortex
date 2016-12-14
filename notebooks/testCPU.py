import fns
from fns import *
from fns.functionsTF import *

N, g, tauv, i, nu = 100, 7,15,0,100
T = 100

gpu = TfSingleNet(N=N,
                  T=T,
                  disp=False,
                  tauv=45,
                  device='/cpu:0',
                  spikeMonitor=False,
                  g0=g,
                  startPlast = 100,
                  nu = nu,
                  NUM_CORES = 1,
                  ratioNI = 0.5)
# gpu.input = apple
print(gpu.lowspthresh)
gpu.lowspthresh = 1.5
gpu.weight_step = 10
gpu.input = np.concatenate([np.zeros(T//2),np.ones(T//2)*50])
gpu.dt = 0.1
gpu.ratio = 0.05
gpu.FACT = 100
gpu.runTFSimul()
