import fns
from fns import *
from fns.functionsTF import *

nodes = 4
i=0
params = []
N = 1000
for nu in range(0,200,20):
    for tauv in [15]:
            for ratio in [1]:
                for g in [3]:
                    i+=1
                    params.append([N, g, tauv, i, nu, ratio])

## colored noise
def runFn(things):
    N, g, tauv, i, nu, ratio = things
    T = 10000
    print(things)
    ### input: colored noise
    gpu = TfSingleNet(N=N,
                      T=T,
                      disp=False,
                      tauv=tauv,
                      device='/gpu:0',
                      spikeMonitor=False,
                      g0=g,
                      startPlast = 50,
                      nu = nu,
                      ratioNI=0.2,
                      memfraction=0.12,
                      NUM_CORES = 1)
    # gpu.input = apple
    gpu.ratio = ratio
    gpu.barname = 'bar %d' % i#(i, len(params))
    gpu.barposition = 0
    gpu.dt = 0.1
    gpu.nuI = nu
    gpu.nuE = nu
    gpu.ratio = 1
    gpu.FACT = 1000
    gpu.runTFSimul()


    filename = "../data/GetGammaSteadyState/GetSteadyState600-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f" % (tauv, g, N, T, nu, ratio)
    with open(filename, 'wb') as f:
        np.savez(f,
                 vvmE = gpu.vvmE,
                 vvmI = gpu.vvmI,
                 vmE = gpu.vmE,
                 vmI = gpu.vmI,
                 imE = gpu.imE,
                 imI = gpu.imI,
                 gamma = gpu.gamma,
                )
    del gpu
    gc.collect()

print(len(params))
p = Pool(nodes=nodes)
re = p.amap(runFn, params)
re.get()
