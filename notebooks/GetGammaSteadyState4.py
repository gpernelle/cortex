import fns
from fns import *
from fns.functionsTFhardbound import *


i=0
p = Pool(nodes=56)
params = []
N = 1000
for nu in range(0,200,10):
    for tauv in [15, 30, 45, 60, 90]:
            for ratio in [0.2,0.5,1,2,3]:
                for g in [10]:
                    i+=1
                    params.append([N, g, tauv, i, nu, ratio])

## colored noise
def runFn(things):
    N, g, tauv, i, nu, ratio = things
    T = 5000
    ### input: colored noise
    gpu = TfSingleNet(N=N,
                      T=T,
                      disp=False,
                      tauv=tauv,
                      device='/cpu:0',
                      spikeMonitor=False,
                      g0=g,
                      startPlast = 50,
                      nu = nu,
                      NUM_CORES = 1)
    # gpu.input = apple
    gpu.ratio = ratio
    gpu.FACT = 50
    gpu.runTFSimul()


    filename = "../data/GetGammaSteadyState/GetSteadyState9-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f" % (tauv, g, N, T, nu, ratio)
    with open(filename, 'wb') as f:
        four = fourier(gpu.vvm[100:])
        np.savez(f,
                 vvm = gpu.vvm,
                 im = gpu.im,
                 freq = four[0],
                 power = four[1],
                 gamma = gpu.gamma,
                )
    del gpu
    gc.collect()


re = p.amap(runFn, params)
re.get()
