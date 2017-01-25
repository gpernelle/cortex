import fns
from fns import *
from fns.functionsTFRon import *

nodes = 2
i=0
params = []
N = 1000
FACT = 50
T = 50000
spm = False
g = 3
IAF = True
WII = -1000
inE = 100
k = 4

# for nu in range(20, 100, 20):
for nu in [20,40,60]:
    for step in np.arange(0,101,10):
        for ratio in [4,5,7,10,12]:
            i+=1
            params.append([T, FACT, N, g, i, nu, ratio, inE, WII, k, step])


## colored noise
def runFn(things):
    T, FACT, N, g, i, nu, ratio, inE, WII, k, step = things
    tauv = 15
    IAF = True
    spm = True
    ratioNI = 0.2
    NI = int(N*ratioNI)
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
                      memfraction=0.46,)
    gpu.input = np.concatenate([np.zeros(T//3), np.ones(T - int(T//3)) * step])
    gpu.ratio = ratio
    gpu.barname = 'bar %d' % i
    gpu.barposition = 0
    gpu.dt = 0.1
    gpu.nuI = nu
    gpu.nuE = nu
    gpu.FACT = FACT
    gpu.k = k
    gpu.wII = WII
    gpu.wIE = -2000
    gpu.wEE = 1000
    gpu.wEI = 1000
    gpu.wEEron = 3000
    gpu.inE = inE
    gpu.weightEvolution = True
    gpu.globalGap = 0
    gpu.IAF = IAF


    filename = "../data/stimulations/stimulation5-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_k-%d_IAF-%d_inE-%d_step-%d"\
               % (tauv, g, N, T, nu, ratio,  gpu.wEE, gpu.wEI, gpu.wIE, gpu.wII, FACT, ratioNI, k, IAF*1, inE, step)

    if not os.path.isfile(filename) or 1:
        gpu.runTFSimul()
        with open(filename, 'wb') as f:
            np.savez(f,
                     vvmE = gpu.vvmE,
                     vvmERon = gpu.vvmERon,
                     vvmI = gpu.vvmI,
                     vmE = gpu.vmE,
                     vmERon = gpu.vmERon,
                     vmI = gpu.vmI,
                     imE = gpu.imE,
                     imI = gpu.imI,
                     gamma = gpu.gamma
                    )
        if spm:
            filename_t = "../data/stimulations/raster_stimulation5-tauv-%d_g-%d_N-%d_T-%d_nu-%d_ratio-%.2f_WEE-%d_WEI-%d_WIE-%d_WII-%d_FACT-%d_rNI-%.2f_k-%d_IAF-%d_inE-%d_step-%d" \
                       % (tauv, g, N, T, nu, ratio, gpu.wEE, gpu.wEI, gpu.wIE, gpu.wII, FACT, ratioNI, k, IAF*1, inE, step)
            with open(filename_t, 'wb') as f:
                print(gpu.raster.transpose().shape)
                np.save(f, gpu.raster.transpose()[700:900,max(0,T//2-10000):])
    del gpu
    gc.collect()

print(len(params))
p = Pool(nodes=nodes)
re = p.amap(runFn, params[:])
re.get()
