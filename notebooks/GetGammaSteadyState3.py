import tensorflow as tf
from functionsTF import *
from functions import *
from IO import *
import time
import numpy as np
from io import BytesIO
import mutual_info
import gc

i=0
p = Pool(nodes=3)
params = []
for nu in range(0,200,10):
    for tauv in [15]:
            for N in [1000]:
                for g in [7]:
                    i+=1
                    params.append([N, g, tauv, i, nu])

## colored noise
def runFn(things):
    N, g, tauv, i, nu = things
    T = 40000
    ### input: colored noise
    print('*'*80)
    print('%d / %d'%(i,25*40*10))

    gpu = TfSingleNet(N=N,
                      T=T,
                      disp=False,
                      tauv=tauv,
                      device='/gpu:0',
                      spikeMonitor=False,
                      g0=g,
                      startPlast = 0,
                      nu = nu,
                      NUM_CORES = 1,
                      memfraction=0.3)
    # gpu.input = apple
    gpu.runTFSimul()


    filename = "GetSteadyState4-tauv-%d_g-%d_N-%d_T-%d_k-%d" % (tauv, g, N, T, nu)
    with open(filename, 'wb') as f:
        four = fourier(gpu.vvm[100:])
        np.savez(f,
                 vvm = gpu.vvm,
                 im = gpu.im,
                 freq = four[0],
                 power = four[1],
                 gamma = gpu.gamma,
                )
    del gpu
    gc.collect()


re = p.amap(runFn, params)
re.get()
