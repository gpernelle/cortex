import fns
from fns import *
from fns.functionsTFhardbound import *
import sys
sys.setrecursionlimit(1000000)
i = 0
params = []
for T in [8000]:
        for N in [1000]:
            for g in np.arange(0, 10, 0.5):
                for nu in np.arange(0, 200, 5):
                        i += 1
                        params.append([T, N, g, nu, i])

scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70

# 3->4: fix ratio g0 to 7

def runFnNoPlast(things):
    dt = 0.1
    DEVICE = '/cpu:0'
    T, N, g, nu, i = things

    print('*' * 80)
    print('%d / %d' % (i, len(params)))
    print(things)
    ### input 1: apple
    gpu1 = TfSingleNet(N=1000, T=T, disp=False, tauv=15, nu=nu, g0=g, ratioNI=0.5,
                   device=DEVICE, spikeMonitor=False, startPlast=999999)
    apple = generateInput2(2, T // dt)
    gpu1.input = apple*0
    gpu1.initWGap = False
    gpu1.dt = 0.1
    gpu1.runTFSimul()

    filename = "../data/PhasePlan7/PhasePlan71_nu-%d_g-%.2f_N-%d_input-%s_T-%d" % (nu, g, N, 'noise', T)
    with open(filename, 'wb') as f:
        np.savez(f, vvm=gpu1.vvm, i=gpu1.im, burst=gpu1.burstingActivity, spike=gpu1.spikingActivity)
    del gpu1
    gc.collect()

print(len(params))
p = Pool(nodes=55)
re = p.amap(runFnNoPlast, params)
re.get()
# runFnNoPlast(params[0])