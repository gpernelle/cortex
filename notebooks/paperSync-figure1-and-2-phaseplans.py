import fns
from fns import *
from fns.functionsTF import *
import sys
sys.setrecursionlimit(1000000)
i = 0
params = []
N=1000
for T in [8000]:
    for w in [[-1000, -3000, 1000, 1000] ]:
        for k in [4]:
            for g in np.arange(0, 8, 0.5):
                for nu in np.arange(0, 101, 5):
                        i += 1
                        params.append([T, N, g, nu, i, w, k])


scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70

# 3->4: fix ratio g0 to 7

def runFnNoPlast(things):
    dt = 0.1
    DEVICE = '/cpu:0'
    T, N, g, nu, i, w, k = things
    WII, WIE, WEE, WEI = w

    print('*' * 80)
    print('%d / %d' % (i, len(params)))
    print(things)
    ### input 1: apple
    gpu1 = TfSingleNet(N=N, T=T, disp=False, tauv=15, nu=nu, g0=g, ratioNI=0.2,
                   device=DEVICE, spikeMonitor=False, startPlast=999999)
    gpu=gpu1
    apple = generateInput2(2, T // dt)
    gpu1.input = apple*0
    gpu1.initWGap = False
    gpu1.dt = 0.1
    gpu.nuI = nu
    gpu.nuE = nu
    gpu.ratio = 15
    gpu.FACT = 100
    gpu.wII = WII
    gpu.wIE = WIE
    gpu.wEE = WEE
    gpu.wEI = WEI
    gpu.inE = 100
    gpu.k = k
    gpu.IAF = True


    filename = "../data/PhasePlan7/PhasePlan290_nu-%d_g-%.2f_N-%d_k-%d_T-%d_WEE-%d_WEI-%d_WIE-%d_WII-%d_inE-%d" \
               % (nu, g, N, k, T, gpu.wEE, gpu.wEI, gpu.wIE, gpu.wII, gpu.inE)

    if not os.path.isfile(filename):
        gpu1.runTFSimul()
        with open(filename, 'wb') as f:
            np.savez(f, vvmE=gpu1.vvmE, vvmI=gpu1.vvmI, vmE=gpu1.vmE, vmI=gpu1.vmI,
                     iI=gpu1.imI,  iE=gpu1.imE,
                     burst=gpu1.burstingActivity, spike=gpu1.spikingActivity)
    del gpu1
    gc.collect()

print(len(params))
p = Pool(nodes=55)
re = p.amap(runFnNoPlast, params)
re.get()
# runFnNoPlast(params[0])