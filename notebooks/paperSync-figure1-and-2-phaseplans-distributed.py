import fns
from fns import *
from fns.functionsTFDistributed import *
import sys
sys.setrecursionlimit(10000)
i = 0
params = []
for T in [80]:
        for N in [1000]:
            for g in np.arange(0, 15, 0.5):
                for nu in np.arange(0, 200, 5):
                        i += 1
                        filename = "../data/PhasePlan7/PhasePlan7_nu-%d_g-%.2f_N-%d_input-%s_T-%d" % (
                        nu, g, N, 'noise', T)
                        device = '/job:local/task:%d'%(i%50)
                        params.append([T, N, g, nu, i, device, filename])

scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70

# 3->4: fix ratio g0 to 7

def runFnNoPlast(params):
    DEVICE = '/cpu:0'

    apple = generateInput2(2, T)

    gpu1 = TfSingleNet(N=N, T=T, disp=False, tauv=15, nu=nu, g0=g, NUM_CORES=1,
                       device=DEVICE, spikeMonitor=False, startPlast=999999, memfraction=0.1)
    gpu1.params = params
    gpu1.input = apple*0
    gpu1.initWGap = False
    gpu1.dt =0.1
    gpu1.runTFSimul()


    with open(filename, 'wb') as f:
        np.savez(f, vvm=gpu1.vvm, i=gpu1.im, burst=gpu1.burstingActivity, spike=gpu1.spikingActivity)
    del gpu1
    gc.collect()

print(len(params))
print(runFnNoPlast(params))


