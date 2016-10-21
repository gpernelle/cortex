import fns
from fns import *
from fns.functionsTFhardbound import *

import gc

i=0
p = Pool(nodes=56)
params = []
for nu in range(0,200,20):
    for tauv in [15, 30, 45, 60, 90]:
            for N in [100]:
                for sG in np.arange(0,100,10):
                    i+=1
                    params.append([N, sG, tauv, i, nu])

## colored noise
def runFn(things):
    N, sG, tauv, i, nu = things
    T = 100
    ### input: colored noise


    g=7

    gpu = TfConnEvolveNet(N=N,
                      T=T,
                      disp=False,
                      tauv=tauv,
                      device='/cpu:0',
                      spikeMonitor=True,
                      g0=g,
                      startPlast = 50,
                      nu = nu,
                      NUM_CORES = 1,
                      both=True,
                     sG = sG,
                          )
    # gpu.input = apple
    gpu.FACT = 10
    gpu.ratio = 3
    gpu.connectTime = 2500
    gpu.initWGap = False
    gpu.runTFSimul()

    filename = "../data/rasters/rastervarPlast-tauv-%d_g-%d_N-%d_T-%d_k-%d" % (tauv, sG, N, T, nu)
    with open(filename, 'wb') as f:
        np.savez(f,
                 vvmN1=gpu.vvmN1,
                 vvmN2=gpu.vvmN2,
                 i1=gpu.i1,
                 i2=gpu.i2,
                 g1=gpu.gammaN1,
                 g2=gpu.gammaN2
                 )

    filename = "../data/rasters/rasterPlast-tauv-%d_g-%d_N-%d_T-%d_k-%d" % (tauv, sG, N, T, nu)
    with open(filename, 'wb') as f:
        r = np.array(gpu.raster)
        r = r.reshape(r.shape[0], r.shape[1]).transpose()
        np.save(f, r)

    del gpu
    gc.collect()

    # gpu = TfConnEvolveNet(N=N,
    #                        T=T,
    #                        disp=False,
    #                        tauv=tauv,
    #                        device='/cpu:0',
    #                        spikeMonitor=True,
    #                        g0=g,
    #                        startPlast=9999999,
    #                        nu=nu,
    #                        NUM_CORES=1,
    #                        both=True,
    #                        sG=sG,
    #                        )
    # # gpu.input = apple
    # gpu.connectTime = 2500
    # gpu.runTFSimul()
    #
    #
    # filename = "../data/rasters/rastervarNoplast-tauv-%d_g-%d_N-%d_T-%d_k-%d" % (tauv, sG, N, T, nu)
    # with open(filename, 'wb') as f:
    #     four = fourier(gpu.vvm[100:])
    #     np.savez(f,
    #              vvmN1=gpu.vvmN1,
    #              vvmN2=gpu.vvmN2,
    #              i1=gpu.i1,
    #              i2=gpu.i2,
    #              g1=gpu.gammaN1,
    #              g2=gpu.gammaN2
    #             )
    #
    # filename = "../data/rasters/rasterNoplast-tauv-%d_g-%d_N-%d_T-%d_k-%d" % (tauv, sG, N, T, nu)
    # with open(filename, 'wb') as f:
    #     r = np.array(gpu.raster)
    #     r = r.reshape(r.shape[0], r.shape[1]).transpose()
    #     np.save(f, r)
    #
    # del gpu
    # gc.collect()

    print('\n'*3)
    print('*'*80)
    print('%d / %d'%(i,25*40*10))
    print('\n' * 3)


re = p.amap(runFn, params)
re.get()

