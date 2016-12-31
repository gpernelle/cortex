import fns
from fns import *
from fns.functionsTF import *

nodes = 4
i=0
params = []
N = 300
FACT = 100
T = 10000
for g in [5,0,10]:
    for nu in range(0,201,10):
        for tauv in [15]:
                for ratio in [0.01,0.1,1,1.5,2,10,100]:
                    i+=1
                    params.append([T, FACT, N, g, tauv, i, nu, ratio])

i = 0
params2 = []
N = 1000
FACT = 10
T = 100000
for g in [5, 0, 10]:
    for nu in range(0, 201, 10):
        for tauv in [15]:
            for ratio in [0.01, 0.1, 1, 1.5, 2, 10, 100]:
                i += 1
                params2.append([T, FACT, N, g, tauv, i, nu, ratio])

## colored noise
def runFn(things):
    T, FACT, N, g, tauv, i, nu, ratio = things
    # print(things)
    ### input: colored noise
    gpu = TfSingleNet(N=N,
                      T=T,
                      tauv=tauv,
                      device='/gpu:0',
                      spikeMonitor=False,
                      g0=g,
                      startPlast = 2000,
                      ratioNI=0.2,
                      memfraction=0.2,

                      )
    # gpu.input = apple
    gpu.ratio = ratio
    gpu.barname = 'bar %d' % i#(i, len(params))
    gpu.barposition = 0
    gpu.dt = 0.1
    gpu.nuI = nu
    gpu.nuE = nu
    gpu.FACT = FACT
    gpu.wII = 1000
    gpu.wIE = -2000
    gpu.wEE = 1000
    gpu.wEI = 1400
    gpu.runTFSimul()


    filename = "../data/GetGammaSteadyState/GetSteadyState130-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f" % (tauv, g, N, T, nu, ratio)
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


re = p.amap(runFn, params2)
re.get()
