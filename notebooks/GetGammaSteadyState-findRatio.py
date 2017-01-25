import fns
from fns import *
from fns.functionsTF import *

nodes = 3
i=0
params = []
N = 1000
FACT = 50
T = 50000
spm = False
for g in [3]:
    for k in [4]:
        for ratio in [1,3,5,7,10]:
            for nu in range(0, 101, 10):
                for IAF in [True]:
                    for WII in [-2000]:
                        for inE in [100]:
                            i+=1
                            params.append([T, FACT, N, g, i, nu, ratio, inE, WII,k])


## colored noise
def runFn(things):
    T, FACT, N, g, i, nu, ratio, inE, WII, k = things
    tauv = 15
    IAF = True
    spm = False
    gGap = 0
    ratioNI = 0.2
    print("%d / %d \n"%(i, len(params)))
    ### input: colored noise
    gpu = TfSingleNet(N=N,
                      T=T,
                      tauv=tauv,
                      device='/gpu:0',
                      spikeMonitor=spm,
                      g0=g,
                      startPlast = 500,
                      ratioNI=0.2,
                      nu=nu,
                      memfraction=0.3,)
    # gpu.input = apple
    gpu.ratio = ratio
    gpu.barname = 'bar %d' % i#(i, len(params))
    gpu.barposition = 0
    gpu.dt = 0.1
    gpu.nuI = nu
    gpu.nuE = nu
    gpu.FACT = FACT
    gpu.k = k
    gpu.wII = -1000
    gpu.wIE = -2000
    gpu.wEE = 1000
    gpu.wEI = 1000
    gpu.inE = 100
    gpu.weightEvolution = True
    gpu.globalGap = 0
    gpu.IAF = True

    gpu.runTFSimul()


    filename = "../data/GetGammaSteadyState/GetSteadyState300mean-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_k-%d_IAF-%d_inE-%d"\
               % (tauv, g, N, T, nu, ratio,  gpu.wEE, gpu.wEI, gpu.wIE, gpu.wII, FACT, ratioNI, k, IAF*1, inE)
    with open(filename, 'wb') as f:
        np.savez(f,
                 vvmE = gpu.vvmE,
                 vvmI = gpu.vvmI,
                 vmE = gpu.vmE,
                 vmI = gpu.vmI,
                 imE = gpu.imE,
                 imI = gpu.imI,
                 gamma = gpu.gamma,
                 # gmin = gpu.gammaMin,
                 # gmax = gpu.gammaMax,
                 # gvar = gpu.gammaVar
                )
    if spm:
        filename = "../data/GetGammaSteadyState/raster_GetSteadyState300mean-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_k-%d_IAF-%d_inE-%d" \
                   % (tauv, g, N, T, nu, ratio, gpu.wEE, gpu.wEI, gpu.wIE, gpu.wII, FACT, ratioNI, k, IAF*1, inE)
        with open(filename, 'wb') as f:
            np.save(f, gpu.raster.transpose())
    del gpu
    gc.collect()

print(len(params))
p = Pool(nodes=nodes)
re = p.amap(runFn, params)
re.get()
