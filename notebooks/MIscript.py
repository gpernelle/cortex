import tensorflow as tf
from functionsTF import *
from functions import *
from IO import *
import time
import numpy as np
from io import BytesIO
import mutual_info

DEVICE = '/cpu:0'
i=0
for T in [60000, 10000, 6000]:
    apple = generateInput(2, T, 60)
    pear = generateInput(3, T, 60)
    for both in [True, False]:
        for N in [400]:
            for sG in [0,10,20,50]:
                for tauv in np.arange(15,95,5):
                    i+=1
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

                    filename = "MI7-both-%s_tauv-%d_sg-%d_N-%d_input-%s_T=%d" % (str(both), tauv,sG, N, 'test', T)
                    with open(filename, 'wb') as f:
                        np.savez(f,vvmN1 = gpu1.vvmN1, vvmN2 = gpu1.vvmN2, vvm = gpu1.vvm,
                                vvmN1_2 = gpu2.vvmN1, vvmN2_2 = gpu2.vvmN2, vvm_2 = gpu2.vvm)
                    del gpu1
                    del gpu2
