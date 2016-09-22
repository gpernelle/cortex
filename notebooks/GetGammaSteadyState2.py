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
p = Pool(nodes=56)
params = []
for k in range(0,50,5):
    for tauv in np.arange(15,90,5):
            for N in [1000]:
                for g in [7]:
                    i+=1
                    params.append([N, g, tauv, i, k])

## colored noise
def runFn(things):
    N, g, tauv, i, k = things
    T = 4000
    ### input: colored noise
    apple = generateInput2(2, T) * k
    print('*'*80)
    print('%d / %d'%(i,25*40*10))

    gpu = TfSingleNet(N=N,
                      T=T,
                      disp=False,
                      tauv=tauv,
                      device='/cpu:0',
                      spikeMonitor=False,
                      g0=g,
                      startPlast = 0,
                      NUM_CORES = 1)
    gpu.input = apple
    gpu.runTFSimul()


    filename = "GetSteadyState3-tauv-%d_g-%d_N-%d_T-%d_k-%d" % (tauv, g, N, T, k)
    with open(filename, 'wb') as f:
        four = fourier(gpu.vvm[100:])
        np.savez(f,
                 vvm = gpu.vvm,
                 freq = four[0],
                 power = four[1],
                 gamma = gpu.gamma,
                )
    del gpu
    gc.collect()


re = p.amap(runFn, params)
re.get()
