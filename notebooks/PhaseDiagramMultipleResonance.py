import tensorflow as tf
from functionsTF import *
from functions import *
from IO import *
import time
import numpy as np
from io import BytesIO
import mutual_info
import gc

DEVICE = '/cpu:0'
i=0
p = Pool(nodes=56)
params = []
T = 4000
for tauv in np.arange(15,90,5):
        for N in [400]:
            for sG in [0,10,20,50]:
                for tauv in np.arange(15,95,5):
                    i+=1
                    params.append([T, both, N, sG, tauv, i])


                    
scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70
                    
                    
def runFn(things):
    F = 10
    T, both, N, sG, tauv, i = things
    apple = generateInput2(2, T, F)
    pear = generateInput2(3, T, F)
    print('*'*80)
    print('%d / %d'%(i,3*2*1*4*16))
    ### input 1: apple
    gpu1 = Tfnet(N=N,T=T, disp=False, tauv=tauv, sG=sG, device=DEVICE, both=both, spikeMonitor=False)
    gpu1.input = apple
    gpu1.runTFSimul()
    apple_out = gpu1.vvm[-1000:]
    
    ### input 2: pear
    disp=False
    gpu2 = Tfnet(N=N,T=T, disp=False, tauv=tauv, sG=sG, device=DEVICE, both=both, spikeMonitor=False)
    gpu2.input = pear
    gpu2.runTFSimul()
    pear_out = gpu2.vvm[-1000:]

    filename = "MI11-both-%s_tauv-%d_sg-%d_N-%d_input-%s_T-%d" % (str(both), tauv,sG, N, 'noise', T)
    with open(filename, 'wb') as f:
        np.savez(f,vvmN1 = gpu1.vvmN1, vvmN2 = gpu1.vvmN2, vvm = gpu1.vvm,
                vvmN1_2 = gpu2.vvmN1, vvmN2_2 = gpu2.vvmN2, vvm_2 = gpu2.vvm)
    del gpu1
    del gpu2
    gc.collect()

re = p.amap(runFn, params)
re.get()
