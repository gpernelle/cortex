import fns
from fns import *
from fns.functionsTF import *

nodes = 4
i=0
params = []
N = 1000
FACT = 100
T = 60000
spm = False
for g in [10]:
    for ratio in [2]:
        for tauv in [15]:
            for nu in range(0, 201, 50):
                for ratioNI in np.linspace(0.1,1,10):
                    for IAF in [True, False]:
                        for gGap in [0,1]:
                            for spm in [0,1]:
                                i+=1
                                params.append([T, FACT, N, g, tauv, i, nu, ratio, ratioNI, gGap, spm, IAF])

# params.append([100000, FACT, N, 10, 15, 0, 150, 2, 1.0, 0, spm])



## colored noise
def runFn(things):
    T, FACT, N, g, tauv, i, nu, ratio, ratioNI, gGap, spm, IAF = things
    print("%d / %d \n"%(i, len(params)))
    ### input: colored noise
    gpu = TfSingleNet(N=N,
                      T=T,
                      tauv=tauv,
                      device='/gpu:0',
                      spikeMonitor=spm,
                      g0=g,
                      startPlast = 2000,
                      ratioNI=ratioNI,
                      nu=nu,
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
    gpu.wII = -1000
    gpu.wIE = -3000
    gpu.wEE = 1000
    gpu.wEI = 1000
    gpu.weightEvolution = True
    gpu.globalGap = gGap
    gpu.IAF = IAF

    gpu.runTFSimul()


    filename = "../data/GetGammaSteadyState/GetSteadyState220mean-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_global-%d_IAF-%d"\
               % (tauv, g, N, T, nu, ratio,  gpu.wEE, gpu.wEI, gpu.wIE, gpu.wII, FACT, ratioNI, gGap, IAF*1)
    with open(filename, 'wb') as f:
        np.savez(f,
                 vvmE = gpu.vvmE,
                 vvmI = gpu.vvmI,
                 vmE = gpu.vmE,
                 vmI = gpu.vmI,
                 imE = gpu.imE,
                 imI = gpu.imI,
                 gamma = gpu.gamma,
                 gmin = gpu.gammaMin,
                 gmax = gpu.gammaMax,
                 gvar = gpu.gammaVar
                )
    if spm:
        filename = "../data/GetGammaSteadyState/raster_GetSteadyState211mean-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_global-%d_IAF-%d" \
                   % (tauv, g, N, T, nu, ratio, gpu.wEE, gpu.wEI, gpu.wIE, gpu.wII, FACT, ratioNI, gGap, IAF*1)
        with open(filename, 'wb') as f:
            np.save(f, gpu.raster.transpose())
    del gpu
    gc.collect()

print(len(params))
p = Pool(nodes=nodes)
re = p.amap(runFn, params)
re.get()


# re = p.amap(runFn, params2)
# re.get()

