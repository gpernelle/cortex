import fns
from fns import *
from fns.functionsTF import *

nodes = 4
i=0
params = []
N = 1000
for nu in range(0,200,50):
    for tauv in [15]:
            for ratio in [1]:
                for g in [20]:
                    i+=1
                    params.append([N, g, tauv, i, nu, ratio])

## colored noise
def runFn(things):
    N, g, tauv, i, nu, ratio = things
    T = 20000
    # print(things)
    ### input: colored noise
    gpu = TfSingleNet(N=N,
                      T=T,
                      tauv=tauv,
                      device='/gpu:0',
                      spikeMonitor=False,
                      g0=g,
                      startPlast = 1000,
                      ratioNI=0.2,
                      memfraction=0.15)
    # gpu.input = apple
    gpu.ratio = ratio
    gpu.barname = 'bar %d' % i#(i, len(params))
    gpu.barposition = 0
    gpu.dt = 0.1
    gpu.nuI = nu
    gpu.nuE = nu
    gpu.FACT = 50
    gpu.runTFSimul()


    filename = "../data/GetGammaSteadyState/GetSteadyState700-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f" % (tauv, g, N, T, nu, ratio)
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
