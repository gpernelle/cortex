import fns
from fns import *
from fns.functionsTF import *


i=0
params = []
N = 1000
for nu in range(0,200,10):
    for tauv in [15, 30, 45, 60, 90]:
            for ratio in [1]:
                for g in [10]:
                    i+=1
                    params.append([N, g, tauv, i, nu, ratio])

## colored noise
def runFn(things):
    N, g, tauv, i, nu, ratio = things
    T = 40000
    print('*' * 80)
    print('%d / %d' % (i, len(params)))
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
                      memfraction=0.1,
                      NUM_CORES = 1)
    # gpu.input = apple
    gpu.ratio = ratio
    gpu.FACT = 50
    gpu.dt = 0.1
    gpu.runTFSimul()


    filename = "../data/GetGammaSteadyState/GetSteadyState100-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f" % (tauv, g, N, T, nu, ratio)
    with open(filename, 'wb') as f:
        four = fourier(gpu.vvmI[100:])
        np.savez(f,
                 vvmE = gpu.vvmE,
                 vvmI = gpu.vvmI,
                 imE = gpu.imE,
                 imI = gpu.imI,
                 freq = four[0],
                 power = four[1],
                 gamma = gpu.gamma,
                )
    del gpu
    gc.collect()

print(len(params))
p = Pool(nodes=3)
re = p.amap(runFn, params)
re.get()
