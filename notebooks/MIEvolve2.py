import tensorflow as tf
from functionsTF import *
from functions import *
from IO import *
import time
import numpy as np
from io import BytesIO
import mutual_info
import gc

i = 0
params = []
for T in [16000]:
    for both in [False, True]:
        for N in [2000]:
            for sG in [0, 20, 100, 200]:
                for tauv in np.arange(15, 90, 20):
                    for k in [100]:
                        i += 1
                        params.append([T, both, N, sG, tauv, i, k])

scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70


'''
Read the data file and compile them into a pandas dataframe
'''
# df = pd.DataFrame(columns=('tauv', 'g', 'k', 'T', 'N', 'freq', 'power', 'gSteady') )
# d = 500
# i=-1
# sigma = 1
# for T in [4000]:
#     for k in range(0,50,5):
#         for N in [1000]:
#             for g in [7]:
#                 for tauv in np.arange(15,90,5):
#                     i+=1
#                     filename = "../data/GetGammaSteadyState/GetSteadyState3-tauv-%d_g-%d_N-%d_T-%d_k-%d" % (tauv, g, N, T, k)
#                     a = np.load(filename)
#                     df.loc[i] = [int(tauv), int(g), int(k), int(T), int(N),
#                                 a['freq'], a['power'], np.mean(a['gamma'][-2000:])]

# df.to_csv('gSteady.csv')
df = pd.read_csv('gSteady.csv')

def runFnNoPlast(things):
    DEVICE = '/gpu:0'
    T, both, N, sG, tauv, i, k = things
    apple = generateInput2(2, T)
    pear = generateInput2(3, T)
    print('*' * 80)
    print('%d / %d' % (i, 160))
    ### input 1: apple
    gpu1 = TfConnEvolveNet(N=N, T=T, disp=False, tauv=tauv, sG=sG, device=DEVICE,
                 both=both, spikeMonitor=False, startPlast=999999)
    gpu1.input = apple*k
    gpu1.initWGap = True
    gpu1.disp = False
    gpu1.runTFSimul()

    ### input 2: pear
    gpu2 = TfConnEvolveNet(N=N, T=T, disp=False, tauv=tauv, sG=sG, device=DEVICE,
                 both=both, spikeMonitor=False, startPlast=99999)
    gpu2.input = pear*k
    gpu2.initWGap = True
    gpu2.disp = False
    gpu2.runTFSimul()

    filename = "MIEvolveNoPlast5-both-%s_tauv-%d_sg-%d_N-%d_input-%s_T-%d_k-%d" % (str(both), tauv, sG, N, 'noise', T, k)
    with open(filename, 'wb') as f:
        np.savez(f, vvmN1=gpu1.vvmN1, vvmN2=gpu1.vvmN2, vvm=gpu1.vvm,
                 vvmN1_2=gpu2.vvmN1, vvmN2_2=gpu2.vvmN2, vvm_2=gpu2.vvm,
                 g1N1=gpu1.gammaN1, g1N2 = gpu1.gammaN2, g1=gpu1.gamma, g1s=gpu1.gammaNS,
                 g2N1=gpu2.gammaN1, g2N2=gpu2.gammaN2, g2=gpu2.gamma,  g2s=gpu2.gammaNS, )
    del gpu1
    del gpu2
    gc.collect()

def runFnPlast(things):
    DEVICE = '/gpu:0'
    T, both, N, sG, tauv, i, k = things
    apple = generateInput2(2, T)
    pear = generateInput2(3, T)
    print('*' * 80)
    print('%d / %d' % (i,160))
    ### input 1: apple
    gpu1 = TfConnEvolveNet(N=N, T=T, disp=False, tauv=tauv, sG=sG, device=DEVICE,
                 both=both, spikeMonitor=False, startPlast=0)
    gpu1.input = apple*k
    gpu1.initWGap = True
    gpu1.disp = False
    gpu1.runTFSimul()

    ### input 2: pear
    gpu2 = TfConnEvolveNet(N=N, T=T, disp=False, tauv=tauv, sG=sG, device=DEVICE,
                 both=both, spikeMonitor=False, startPlast=0)
    gpu2.input = pear*k
    gpu2.initWGap = True
    gpu2.disp = False
    gpu2.runTFSimul()

    filename = "MIEvolvePlast5-both-%s_tauv-%d_sg-%d_N-%d_input-%s_T-%d_k-%d" % (str(both), tauv, sG, N, 'noise', T, k)
    with open(filename, 'wb') as f:
        np.savez(f, vvmN1=gpu1.vvmN1, vvmN2=gpu1.vvmN2, vvm=gpu1.vvm,
                 vvmN1_2=gpu2.vvmN1, vvmN2_2=gpu2.vvmN2, vvm_2=gpu2.vvm,
                 g1N1=gpu1.gammaN1, g1N2=gpu1.gammaN2, g1=gpu1.gamma,  g1s=gpu1.gammaNS,
                 g2N1=gpu2.gammaN1, g2N2=gpu2.gammaN2, g2=gpu2.gamma,  g2s=gpu2.gammaNS,)
    del gpu1
    del gpu2
    gc.collect()



q = Pool(nodes=1)
re = q.amap(runFnPlast, params)
re.get()

p = Pool(nodes=1)
re = p.amap(runFnNoPlast, params)
re.get()
