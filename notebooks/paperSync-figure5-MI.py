import fns
from fns import *
from fns.functionsTFhardbound import *
import os

### CPU ###
# i=0
# p = Pool(nodes=56)
# params = []
# for nu in [50,100]:
#     for tauv in [15, 30, 45, 60, 90]:
#         for sG in [0,10,50,100,200]:
#             i+=1
#             params.append([sG, tauv, i, nu])
#
# ## colored noise
# def runFn(things):
#     N = 2000
#     g = 10
#     ratio = 0.2
#     T = 20000
#     sG, tauv, i, nu = things
#     ### input: colored noise
#
#     filename = "../data/rasters/rastervarPlastMI-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f" % (tauv, sG, N, T, nu,  ratio)
#     if not os.path.isfile(filename):
#         gpu = TfConnEvolveNet(N=N,
#                               T=T,
#                               disp=False,
#                               tauv=tauv,
#                               device='/cpu:0',
#                               spikeMonitor=True,
#                               g0=g,
#                               startPlast=0,
#                               nu=nu,
#                               NUM_CORES=1,
#                               both=True,
#                               sG=sG,
#                               )
#
#         apple = generateInput2(3, T) * 20
#         gpu.input = apple
#         gpu.debug = False
#         gpu.initWGap = False
#         gpu.connectTime = 10000
#         gpu.FACT = 50
#         gpu.ratio = ratio
#         gpu.runTFSimul()
#
#         filename = "../data/rasters/rastervarPlastMI-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f" % (tauv, sG, N, T, nu, gpu.ratio)
#         with open(filename, 'wb') as f:
#             np.savez(f, vvmN1=gpu.vvmN1, vvmN2=gpu.vvmN2, vvm=gpu.vvm,
#                      g1N1=gpu.gammaN1, g1N2=gpu.gammaN2, g1=gpu.gamma, g1s=gpu.gammaNS,
#                      i1N1=gpu.i1, i1N2=gpu.i2, )
#
#         filename = "../data/rasters/rasterPlastMI-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f" % (tauv, sG, N, T, nu, gpu.ratio)
#         with open(filename, 'wb') as f:
#             r = np.array(gpu.raster)
#             r = r.reshape(r.shape[0], r.shape[1]).transpose()
#             np.save(f, r)
#         del gpu
#         gc.collect()
#     else:
#         print('already done!')
#
#
# re = p.amap(runFn, params)
# re.get()

##### GPU ONLY ######

'''
MI2 :  ratio=0.5, dt=0.1, FACT=400, k input=20
Mi3 : idem but k input = 50
'''

i=0
for nu in [50,100]:
    for tauv in [15, 30, 45, 60, 90]:
        for sG in [0,10,50,100,200]:
            i+=1
            N = 2000
            g = 10
            ratio = 0.5
            T = 20000

            extension = "MI3-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f_dt-0.1" % (tauv, sG, N, T, nu,  ratio)
            filename = "../data/rasters/rastervarPlast" +  extension
            if not os.path.isfile(filename):
                gpu = TfConnEvolveNet(N=N,
                                      T=T,
                                      disp=False,
                                      tauv=tauv,
                                      device='/gpu:0',
                                      spikeMonitor=True,
                                      g0=g,
                                      startPlast=500,
                                      nu=nu,
                                      NUM_CORES=1,
                                      both=True,
                                      sG=sG,
                                      )
                ### input: colored noise
                apple = generateInput2(3, T) * 50
                gpu.input = apple
                gpu.debug = False
                gpu.connectTime = 10000
                gpu.FACT = 400
                gpu.dt = 0.1
                gpu.ratio = ratio
                gpu.runTFSimul()

                filename = "../data/rasters/rastervarPlast" +  extension
                with open(filename, 'wb') as f:
                    np.savez(f, vvmN1=gpu.vvmN1, vvmN2=gpu.vvmN2, vvm=gpu.vvm,
                             g1N1=gpu.gammaN1, g1N2=gpu.gammaN2, g1=gpu.gamma, g1s=gpu.gammaNS,
                             i1N1=gpu.i1, i1N2=gpu.i2, )

                filename = "../data/rasters/rasterPlast" +  extension
                with open(filename, 'wb') as f:
                    r = np.array(gpu.raster)
                    r = r.reshape(r.shape[0], r.shape[1]).transpose()
                    np.save(f, r)
                del gpu
                gc.collect()
            else:
                print('already done!')
