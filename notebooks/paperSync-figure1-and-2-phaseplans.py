import fns
from fns import *
from fns.functionsTF import *
import sys
sys.setrecursionlimit(1000000)
i = 0
params = []
for T in [4000]:
        for N in [300]:
            for g in np.arange(0, 10, 1):
                for nu in np.arange(0, 200, 10):
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
    gpu1 = TfSingleNet(N=1000, T=T, disp=False, tauv=15, nu=nu, g0=g, ratioNI=0.2,
                   device=DEVICE, spikeMonitor=False, startPlast=999999)
    gpu=gpu1
    apple = generateInput2(2, T // dt)
    gpu1.input = apple*0
    gpu1.initWGap = False
    gpu1.dt = 0.1
    gpu.nuI = nu
    gpu.nuE = nu
    gpu.ratio = 1
    gpu.FACT = 50
    gpu.wII = 1000
    gpu.wIE = -2000
    gpu.wEE = 1000
    gpu.wEI = 1400
    gpu1.runTFSimul()

    filename = "../data/PhasePlan7/PhasePlan151_nu-%d_g-%.2f_N-%d_input-%s_T-%d" % (nu, g, N, 'noise', T)
    with open(filename, 'wb') as f:
        np.savez(f, vvmE=gpu1.vvmE, vvmI=gpu1.vvmI, vmE=gpu1.vmE, vmI=gpu1.vmI,
                 iI=gpu1.imI,  iE=gpu1.imE,
                 burst=gpu1.burstingActivity, spike=gpu1.spikingActivity)
    del gpu1
    gc.collect()

print(len(params))
p = Pool(nodes=8)
re = p.amap(runFnNoPlast, params)
re.get()
# runFnNoPlast(params[0])