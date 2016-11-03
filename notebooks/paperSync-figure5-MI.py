import fns
from fns import *
from fns.functionsTFhardbound import *
import os


i=0
p = Pool(nodes=56)
params = []
for nu in [50,100]:
    for tauv in [15, 30, 45, 60, 90]:
        for sG in [0,10,50,100,200]:
            i+=1
            params.append([sG, tauv, i, nu])

## colored noise
def runFn(things):
    N = 2000
    g = 10
    ratio = 0.2
    T = 20000
    sG, tauv, i, nu = things
    ### input: colored noise

    filename = "../data/rasters/rastervarPlastMI-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f" % (tauv, sG, N, T, nu,  ratio)
    if not os.path.isfile(filename):
        gpu = TfConnEvolveNet(N=N,
                              T=T,
                              disp=False,
                              tauv=tauv,
                              device='/cpu:0',
                              spikeMonitor=True,
                              g0=g,
                              startPlast=0,
                              nu=nu,
                              NUM_CORES=1,
                              both=True,
                              sG=sG,
                              )

        apple = generateInput2(3, T) * 20
        gpu.input = apple
        gpu.debug = False
        gpu.initWGap = False
        gpu.connectTime = 10000
        gpu.FACT = 50
        gpu.ratio = ratio
        gpu.runTFSimul()

        filename = "../data/rasters/rastervarPlastMI-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f" % (tauv, sG, N, T, nu, gpu.ratio)
        with open(filename, 'wb') as f:
            np.savez(f, vvmN1=gpu.vvmN1, vvmN2=gpu.vvmN2, vvm=gpu.vvm,
                     g1N1=gpu.gammaN1, g1N2=gpu.gammaN2, g1=gpu.gamma, g1s=gpu.gammaNS,
                     i1N1=gpu.i1, i1N2=gpu.i2, )

        filename = "../data/rasters/rasterPlastMI-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f" % (tauv, sG, N, T, nu, gpu.ratio)
        with open(filename, 'wb') as f:
            r = np.array(gpu.raster)
            r = r.reshape(r.shape[0], r.shape[1]).transpose()
            np.save(f, r)
        del gpu
        gc.collect()
    else:
        print('already done!')


re = p.amap(runFn, params)
re.get()