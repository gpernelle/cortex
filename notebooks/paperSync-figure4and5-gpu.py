import fns
from fns import *
from fns.functionsTFhardbound import *
import os

#### GPU ######
i=0

for nu in [100]:
    for tauv in np.arange(15,60,5): #9
        for sG in np.arange(0,100,10): #5
            i+=1
            N = 1000
            g = 20
            ratio = 0.5
            T = 200000
            dt = 0.1
            k=0
            ### input: colored noise


            ext = "7-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f_dt-%.2f_nu-%d" % (tauv, sG, N, T, k, ratio, dt, nu)
            filename = "../data/rasters/rastervarPlast" + ext
            if not os.path.isfile(filename):
                gpu = TfConnEvolveNet(N=N,
                                      T=T,
                                      disp=False,
                                      tauv=tauv,
                                      device='/gpu:0',
                                      spikeMonitor=False,
                                      g0=g,
                                      startPlast=10000,
                                      nu=nu,
                                      NUM_CORES=1,
                                      both=False,
                                      sG=sG,
                                      )
                gpu.debug = False
                gpu.initWGap = -1
                gpu.g1 = 10
                gpu.g2 = 10
                gpu.connectTime = int(T/2)
                gpu.FACT = 100
                gpu.dt = dt
                gpu.ratio = ratio
                gpu.runTFSimul()



                filename = "../data/rasters/rastervarPlast" + ext
                with open(filename, 'wb') as f:
                    np.savez(f, vvmN1=gpu.vvmN1, vvmN2=gpu.vvmN2,
                             g1N1=gpu.gammaN1, g1N2=gpu.gammaN2, g1s=gpu.gammaNS,
                             i1N1=gpu.i1, i1N2=gpu.i2, )

                # filename = "../data/rasters/rasterPlast3" + ext
                # with open(filename, 'wb') as f:
                #     r = np.array(gpu.raster)
                #     r = r.reshape(r.shape[0], r.shape[1]).transpose()
                #     np.save(f, r)
                del gpu
                gc.collect()
            else:
                print('already done!')




for k in [50, 100, 30, 10, 1]:
    for nu in [100]:
        for tauv in np.arange(15,60,5): #9
            for sG in np.arange(0,100,10): #10
                i+=1
                N = 1000
                g = 10
                ratio = 0.5
                T = 200000
                dt = 0.1
                ### input: colored noise


                ext = "MI7-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f_dt-%.2f_nu-%d" % (tauv, sG, N, T, k, ratio, dt, nu)
                filename = "../data/rasters/rastervarPlast" + ext
                if not os.path.isfile(filename):
                    gpu = TfConnEvolveNet(N=N,
                                          T=T,
                                          disp=False,
                                          tauv=tauv,
                                          device='/gpu:0',
                                          spikeMonitor=False,
                                          g0=g,
                                          startPlast=500,
                                          nu=nu,
                                          NUM_CORES=1,
                                          both=False,
                                          sG=sG,
                                          )
                    apple = generateInput2(3, T) * k
                    gpu.input = apple
                    gpu.debug = False
                    gpu.initWGap = -1
                    gpu.g1 = 10
                    gpu.g2 = 10
                    gpu.connectTime = int(T/2)
                    gpu.FACT = 200
                    gpu.dt = dt
                    gpu.ratio = ratio
                    gpu.runTFSimul()



                    filename = "../data/rasters/rastervarPlast" + ext
                    with open(filename, 'wb') as f:
                        np.savez(f, vvmN1=gpu.vvmN1, vvmN2=gpu.vvmN2,
                                 g1N1=gpu.gammaN1, g1N2=gpu.gammaN2, g1s=gpu.gammaNS,
                                 i1N1=gpu.i1, i1N2=gpu.i2, )

                    # filename = "../data/rasters/rasterPlast3" + ext
                    # with open(filename, 'wb') as f:
                    #     r = np.array(gpu.raster)
                    #     r = r.reshape(r.shape[0], r.shape[1]).transpose()
                    #     np.save(f, r)
                    del gpu
                    gc.collect()
                else:
                    print('already done!')



