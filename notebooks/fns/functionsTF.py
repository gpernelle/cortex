from fns.utils import *
import PIL.Image
from io import BytesIO
from IPython.display import clear_output, Image, display

import tensorflow as tf
from tensorflow.python.client import timeline

# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('summaries_dir', '/tmp/tensorflow_logs', 'Summaries directory')
# flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')

print("*"*80)
print("functionsTFhardbound loaded!")
print("*"*80)
G = 12
DEBUG = False

def namestr(obj, namespace):
    return [name for name in namespace if namespace[name] is obj]

def ps(vars):
    for var in vars:
        if DEBUG:
            print(namestr(var, globals()['TfSingleNet']), var)


def getGSteady(tauv, k, N=100):
    '''
    Get steady state value of the gap junction strenght
    '''
    df = pd.read_csv('gSteady.csv')
    df2 = df[(df['tauv']==tauv) & (df['nu']==k) & (df['N']==N) & (df['ratio']==1) & (df['g']==10)]
    return df2['gSteady'].values[0]

'''
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////

'''

def makeConn(N, ratio=None, NE=0, NI=0, TF=True):
    if ratio is not None:
        NI = int(N * ratio)
        NE = int(N - NI)
        N = int(N)
    print(N, NE, NI)
    conn = np.ones((N,N)) - np.diag(np.ones(N))
    connII_ = np.concatenate([np.zeros(NE), np.ones(NI)])
    connII = connII_.reshape(N,1)
    connII = (connII @ connII.T) * conn

    connEE_ = 1 - connII_
    connEE = connEE_.reshape(N,1)
    connEE = (connEE @ connEE.T) * conn

    connIE = (conn - (connEE + connII))* np.tril(np.ones((N,N)))
    connEI = (conn - (connEE + connII))* np.tril(np.ones((N,N))).T

    if TF:
        conn = tf.Variable(conn, dtype='float32', name='all')
        connEE = tf.Variable(connEE, dtype='float32', name='EE')
        connII = tf.Variable(connII, dtype='float32', name='II')
        connEI = tf.Variable(connEI, dtype='float32', name='EI')
        connIE = tf.Variable(connIE, dtype='float32', name='IE')
        tf.global_variables_initializer().run()
    return conn, connEE, connII, connEI, connIE

makeConn(10,0.2,TF=0)

def makeVect(N, ratio=None, NE=0, NI=0):
    if ratio is not None:
        NI = int(N*ratio)
        NE = N - NI
    vConnI = np.concatenate([np.zeros((NE,1)), np.ones((NI,1))])
    vConnE = (1-vConnI)
    return tf.Variable(vConnE, dtype='float32'), tf.Variable(vConnI, dtype='float32')


class TfSingleNet:

    def __init__(self, N=400,
                 T=400,
                 disp=False,
                 spikeMonitor=False,
                 input=None,
                 tauv=15,
                 device='/gpu:0',
                 NUM_CORES=1,
                 g0=7,
                 nu=100,
                 ratioNI = 0.2,
                 startPlast=500,
                 memfraction=0.95):
        tf.reset_default_graph()
        self.N = N
        self.NI = int(N * ratioNI)
        self.NE = int(N - self.NI)
        self.T = T
        self.disp = disp
        self.spikeMonitor = spikeMonitor
        self.tauv = tauv
        self.g = g0
        self.g0fromFile = False
        self.device = device
        self.startPlast = startPlast
        self.raster = []
        self.nu = nu
        self.ratio = 1
        self.FACT = 10
        self.lowspthresh = 1.5
        self.weight_step = 100
        self.wII = 700
        self.wEE = 700
        self.wEI = 1000
        self.wIE = -1000
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memfraction)

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(
            inter_op_parallelism_threads=NUM_CORES,
            intra_op_parallelism_threads=NUM_CORES,
            gpu_options=gpu_options,
            device_count={'GPU': (device[:4] == '/gpu') * 1}
        )
        )
        if input is None:
            self.input = np.ones((T, 1))

    def varname(self):
        d = {v: k for k, v in globals().items()}
        return d[self]

    def DisplayArray(self, a, fmt='jpeg', rng=[0, 1], text=""):
        """Display an array as a picture."""
        a = (a - rng[0]) / float(rng[1] - rng[0]) * 255
        a = np.uint8(np.clip(a, 0, 255))
        f = BytesIO()
        PIL.Image.fromarray(np.uint8(plt.cm.YlGnBu_r(a) * 255)).save(f, fmt)
        display(Image(data=f.getvalue()))
        print(text)

    def init_float(self, shape, name):
        #     return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)
        return tf.Variable(tf.zeros(shape), name=name)

    def variable_summaries(self, var, name):
        """Attach a lot of summaries to a Tensor."""
        with tf.name_scope('summaries'):
            mean = tf.reduce_mean(var)
            tf.scalar_summary('mean/' + name, mean)
            with tf.name_scope('stddev'):
                stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
            tf.scalar_summary('sttdev/' + name, stddev)
            tf.scalar_summary('max/' + name, tf.reduce_max(var))
            tf.scalar_summary('min/' + name, tf.reduce_min(var))
            tf.histogram_summary(name, var)
        return 0

    def runTFSimul(self):
        #################################################################################
        ### INITIALISATION
        #################################################################################
        N = self.N
        NI = self.NI
        NE = self.NE
        T = self.T
        with tf.device(self.device):
            scaling = 1 / (1 / (2 * 2 / self.dt)) ** 0.5 * 70

            with tf.name_scope('membrane_var'):
                # Create variables for simulation state
                u = self.init_float([N, 1], 'u')
                v = self.init_float([N, 1], 'v')
                # currents
                iBack = self.init_float([N, 1], 'iBack')
                iChem = self.init_float([N, 1], 'iChem')
                input = tf.cast(tf.constant(self.input, name="input"), tf.float32)

            with tf.name_scope('spiking_bursting'):
                LowSp = self.init_float([N, 1], 'bursting')
                vv = self.init_float([N, 1], 'spiking')

            with tf.name_scope('monitoring'):
                vmE = self.init_float([T], "vm")
                vmI = self.init_float([T], "vm")
                umE = self.init_float([T], "um")
                umI = self.init_float([T], "um")
                vvmE = self.init_float([T], "vvm")
                vvmI = self.init_float([T], "vvm")
                pmE = self.init_float([T], "pm")
                pmI = self.init_float([T], "pm")
                lowspm = self.init_float([T], "lowspm")
                imE = self.init_float([T], "im")
                imI = self.init_float([T], "im")
                gm = self.init_float([T//self.weight_step + 1], "gm")
                iEffm = self.init_float([T], "iEffm")
                spikes = self.init_float([T, N], "spikes")

            with tf.name_scope('synaptic_connections'):
                # synaptics connection
                # conn = tf.constant(np.ones((N, N), dtype='float32') - np.diag(np.ones((N,), dtype='float32')))
                conn, connEE, connII, connEI, connIE = makeConn(N, NE=NE, NI=NI)
                vectE, vectI = makeVect(N, NE=NE, NI=NI)


                self.conn = conn.eval()
                nbOfGaps = NI*(NI-1)


                if self.g0fromFile:
                    self.g = getGSteady(self.tauv, 5, 1000)
                g0 = self.g / (nbOfGaps**0.5)
                wGap_init = (tf.random_normal((N, N), mean=g0, stddev=g0/2, dtype=tf.float32,
                                              seed=None, name=None))

                wII_init = self.wII / ((NI*(NI-1))**0.5) / self.dt
                if NE>0:
                    wEE_init = self.wEE / ((NE*(NE-1))**0.5) / self.dt
                else:
                    wEE_init = 0
                wIE_init = self.wIE / (NI*NE-1)**0.5 / self.dt
                wEI_init = self.wEI / (NI*NE-1)**0.5 / self.dt

                wGap = tf.Variable(tf.mul(wGap_init, connII))
                WII = tf.Variable(tf.mul(wII_init, connII))
                WEE = tf.Variable(tf.mul(wEE_init, connEE))
                WEI = tf.Variable(tf.mul(wEI_init, connEI))
                WIE = tf.Variable(tf.mul(wIE_init, connIE))

                # plasticity learning rates
                A_LTD_ = 2.45e-5 * self.FACT * 400 / N
                A_LTD = tf.constant(A_LTD_, name="A_LTP", dtype=tf.float32)
                A_LTP = tf.constant(self.ratio * A_LTD_, name="A_LTD", dtype=tf.float32)

            with tf.name_scope("simulation_params"):
                # stimulation

                TImean = tf.constant(self.nu * 1.0, name="mean_input_current", dtype=tf.float32)
                # timestep
                dt = tf.constant(self.dt * 1.0, name="timestep", dtype=tf.float32)
                tauv = tf.constant(self.tauv*1.0, dtype=tf.float32)

                startPlast = self.startPlast
                weight_step = self.weight_step

            sim_index = tf.Variable(0.0, name="sim_index")
            one = tf.Variable(1.0)
            ones = tf.ones((1, N))

        #################################################################################
        ## Computation
        #################################################################################
        with tf.device(self.device):
            with tf.name_scope('Currents'):
                # Discretized PDE update rules
                ps([WII, vv, vectI])

                iChem_ = iChem + \
                         tf.mul( dt / 5 * (-iChem + tf.matmul(WII + WEI, tf.to_float(vv))), vectI) + \
                         tf.mul( dt / 3 * (-iChem + tf.matmul(WEE + WIE, tf.to_float(vv))), vectE)
                # current
                iBack_ = iBack + dt / 5 * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
                                                                     seed=None, name=None))
                input_ = input[tf.to_int32(sim_index)]

                # input to network: colored noise + external input
                iEff_ = iBack_ * scaling + input_ + TImean

                iGap_ = tf.matmul(wGap, v) - tf.mul(tf.reshape(tf.reduce_sum(wGap, 0), (N, 1)), v)

                I_ = iGap_ + iChem_ + iEff_
                ps([I_, iGap_, iEff_, input_, iBack_, iChem_])

            # IZHIKEVICH
            with tf.name_scope('Izhikevich'):
                # voltage
                v_ = tf.mul(v + dt / tauv * (tf.mul((v + 60), (v + 50)) - 20 * u + 8 * I_), vectI) + \
                    tf.mul(v + dt / 100 * (0.7 *  (v + 60) * (v + 40) - u + I_ ), vectE)
                # adaptation
                u_ = u + tf.mul(dt * 0.044 * (v_ + 55 - u), vectI) + \
                    tf.mul(dt * 0.03 * (-2 * (v + 60) - u), vectE)
                # spikes
                vv_ = tf.mul(tf.to_float(tf.greater(v_, 25.0)), vectI) + \
                    tf.mul(tf.to_float(tf.greater(v_, 35.0)), vectE)
                # reset
                v_ = tf.mul(vv_, -40.0)*vectI + tf.mul(vv_, -50.0)*vectE + tf.mul((1 - vv_), v_)

                u_ = u_ + 50*vv_*vectI + 100*vectE*vv_

            # bursting
            with tf.name_scope('bursting'):
                LowSp_ = (LowSp + dt / 8.0 * (vv_ * 8.0 / dt - LowSp))
                p_ = tf.to_float(tf.greater(LowSp_, 1.5))

            # plasticity
            with tf.name_scope('plasticity'):
                A = tf.matmul(p_, ones, name="bursts")  # bursts
                B = tf.matmul(vv_ * vectI, ones, name="spikes")  # spikes

                dwLTD_ = A_LTD * tf.add(A, tf.transpose(A, name="tr_bursts"))
                dwLTP_ = A_LTP * tf.add(B, tf.transpose(B, name="tr_spikes"))

                dwGap_ = dt * tf.sub(dwLTP_, dwLTD_)
                wGap_ = tf.clip_by_value(wGap + dwGap_, clip_value_min=0, clip_value_max=10 ** 10)

            # monitoring
            with tf.name_scope('Monitoring'):
                vvmeanE_ = tf.reduce_sum(vv_*vectE)
                vvmeanI_ = tf.reduce_sum(vv_*vectI)
                vmeanE_ = tf.reduce_mean(v_*vectE)
                vmeanI_ = tf.reduce_mean(v_*vectI)
                umeanE_ = tf.reduce_mean(u_*vectE)
                umeanI_ = tf.reduce_mean(u_*vectI)
                pmeanE_ = tf.reduce_mean(p_*vectE)
                pmeanI_ = tf.reduce_mean(p_*vectI)
                lowspmean_ = tf.reduce_mean(LowSp_)
                imeanE_ = tf.reduce_mean(I_*vectE)
                imeanI_ = tf.reduce_mean(I_*vectI)
                iEffm_ = tf.reduce_mean(iEff_)
                update = tf.group(
                    tf.scatter_update(vvmE, tf.to_int32(sim_index), vvmeanE_),
                    tf.scatter_update(vvmI, tf.to_int32(sim_index), vvmeanI_),
                    tf.scatter_update(vmE, tf.to_int32(sim_index), vmeanE_),
                    tf.scatter_update(vmI, tf.to_int32(sim_index), vmeanI_),
                    tf.scatter_update(umE, tf.to_int32(sim_index), umeanE_),
                    tf.scatter_update(umI, tf.to_int32(sim_index), umeanI_),
                    tf.scatter_update(pmE, tf.to_int32(sim_index), pmeanE_),
                    tf.scatter_update(pmI, tf.to_int32(sim_index), pmeanI_),
                    tf.scatter_update(lowspm, tf.to_int32(sim_index), lowspmean_),
                    tf.scatter_update(imE, tf.to_int32(sim_index), imeanE_),
                    tf.scatter_update(imI, tf.to_int32(sim_index), imeanI_),
                    tf.scatter_update(iEffm, tf.to_int32(sim_index), iEffm_),
                    sim_index.assign_add(one),
                )

            with tf.name_scope('Weights_monitoring'):
                gm_ = tf.reduce_sum(wGap*connII)
                update_weights = tf.group(
                    tf.scatter_update(gm, tf.to_int32(sim_index / weight_step), gm_),
                )

            with tf.name_scope('Raster_Plot'):
                spike_update = tf.group(
                    tf.scatter_update(spikes, tf.to_int32(sim_index), tf.reshape((vv_), (N,))),
                )

            # Operation to update the state
            step = tf.group(
                v.assign(v_),
                vv.assign(vv_),
                u.assign(u_),
                iBack.assign(iBack_),
                LowSp.assign(LowSp_),
            )

            plast = tf.group(
                wGap.assign(wGap_),
            )

            # update_index = tf.group(
            #     sim_index.assign_add(one),
            # )


            # initialize the graph
            tf.global_variables_initializer().run()

            t0 = time.time()
            for i in range(T):

                # Step simulation
                ops = {'plast': [step, plast, update],
                       'static': [step, update]
                       }
                if self.spikeMonitor:
                    for k, v in ops.items():
                        ops[k] = v + [spike_update]

                if i>startPlast:
                    self.sess.run(ops['plast'])
                else:
                    self.sess.run(ops['static'])

                if i % weight_step == 0:
                    self.sess.run([update_weights])

                # self.sess.run([update_index])

                # Visualize every X steps
                if i % 1 == 0:
                    if self.disp:
                        clear_output(wait=True)
                        self.DisplayArray(wGap.eval(), rng=[0, 1.5 * g0], text="%.2f ms" % (i * self.dt))

                if i==0:
                    self.w0 = wGap.eval()
                elif i==T-1:
                    self.wE = wGap.eval()

                # monitoring variables
                self.vvmE = vvmE.eval()
                self.vvmI = vvmI.eval()
                self.pE = pmE.eval()
                self.pI = pmI.eval()
                self.lowsp = lowspm.eval()
                self.imE = imE.eval()
                self.imI = imI.eval()
                self.iEff = iEffm.eval()
                self.gamma = gm.eval() / np.sum(nbOfGaps)
                if self.spikeMonitor:
                    self.raster = spikes.eval()
                self.burstingActivity = np.mean(self.pI)
                self.spikingActivity = np.mean(self.vvmI)

        print('%.2f' % (time.time() - t0))
        self.sess.close()


'''
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
'''

class TfConnEvolveNet:
    # DEVICE = '/gpu:0'

    def __init__(self, N=400, T=400, disp=False, spikeMonitor=False, input=None, tauv=15,
                 sG=10, device='/gpu:0', both=False, NUM_CORES=1, g0=7, startPlast=500, nu=0, memfraction=1,
                 profiling=False):
        tf.reset_default_graph()
        self.N = N
        self.T = T
        self.debug = False
        self.disp = disp
        self.spikeMonitor = spikeMonitor
        self.tauv = tauv
        self.sG = sG
        self.g = g0
        self.device = device
        self.both = both
        self.initWGap = True
        self.startPlast = startPlast
        self.raster = []
        self.showProgress = False
        connectTime = 0
        self.FACT = 10
        self.ratio = 1
        self.weight_step = 100
        self.nu = nu
        self.profiling = profiling
        self.stabTime = 2000

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(
            log_device_placement = True,
            inter_op_parallelism_threads=NUM_CORES,
            intra_op_parallelism_threads=NUM_CORES,
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memfraction,
                                        allow_growth=True),
        )
        )
        if input is None:
            self.input = np.ones((T,1), dtype='int32')

    def DisplayArray(self, a, fmt='jpeg', rng=[0,1], text=""):
        """Display an array as a picture."""
        a = (a - rng[0])/float(rng[1] - rng[0])*255
        a = np.uint8(np.clip(a, 0, 255))
        f = BytesIO()
        PIL.Image.fromarray(np.uint8(plt.cm.YlGnBu_r(a)*255)).save(f, fmt)
        display(Image(data=f.getvalue()))
        print(text)

    def init_float(self, shape, name):
        return tf.Variable(tf.zeros(shape), name=name)

    def runTFSimul(self):
        #################################################################################
        ### INITIALISATION
        #################################################################################
        N = self.N
        T = self.T

        with tf.device('/cpu:0'):
            sG = int(self.sG)
            nbInCluster = int(N // 2)

            ones = tf.ones((nbInCluster + sG, nbInCluster + sG), dtype=tf.bool)
            conn_1 = tf.pad(ones, [[0, N - (nbInCluster + sG)], [0, N - (nbInCluster + sG)]])
            conn_2 = tf.pad(ones, [[N - (nbInCluster + sG), 0], [N - (nbInCluster + sG), 0]])

            tzr = tf.ones((N, N)) - tf.diag(tf.ones(N))

            ones = tf.ones((nbInCluster - sG, nbInCluster - sG), dtype=tf.bool)
            conn1 = tf.pad(ones, [[0, N - (nbInCluster - sG)], [0, N - (nbInCluster - sG)]])
            conn2 = tf.pad(ones, [[N - (nbInCluster - sG), 0], [N - (nbInCluster - sG), 0]])

            connS1 = tf.logical_and(tf.logical_not(conn1), conn_1)
            connS2 = tf.logical_and(tf.logical_not(conn2), conn_2)
            connS = tf.logical_or(connS1, connS2)

            # N1 and N2 without shared gap
            conn0_ = tf.mul(tf.to_float(tf.logical_or(conn1, conn2)), tzr)
            # N1 and N2 with shared gap
            conn_ = tf.mul(tf.to_float(tf.logical_or(tf.logical_or(conn1, conn2), connS)), tzr)
            # just shared gap
            connS_ = tf.mul(tf.to_float(connS), tzr)

            # finally also convert to float conn1 and conn2
            conn1_ = tf.to_float(conn1)
            conn2_ = tf.to_float(conn2)

            conn = tf.Variable(tf.ones((N,N)), name="connCPU")
            conn0 = tf.Variable(tf.ones((N,N)), name="conn0CPU")
            conn1 = tf.Variable(tf.ones((N,N)), name="conn1CPU")
            conn2 = tf.Variable(tf.ones((N,N)), name="conn2CPU")
            connS = tf.Variable(tf.ones((N,N)), name="connSCPU")

            synapses = tf.group(
                conn.assign(conn_),
                conn0.assign(conn0_),
                conn1.assign(conn1_),
                conn2.assign(conn2_),
                connS.assign(connS_)
            )
            tf.global_variables_initializer().run()
            self.sess.run([synapses],
                          )

            # to check connections
            self.conn = conn.eval()
            self.conn1 = conn1.eval()
            self.conn2 = conn2.eval()
            self.connS = connS.eval()
            self.conn0 = conn0.eval()

        with tf.device(self.device):
            scaling = tf.Variable(1 / (1 / (2 * 2 / self.dt)) ** 0.5 * 70, name="scaling")

            # Create variables for simulation state
            u = self.init_float([N, 1], 'u')
            v = tf.Variable(tf.ones([N, 1], tf.float32)*(-70))
            ind = tf.Variable(0, dtype='float32')
            LowSp =self.init_float([N, 1], 'bursting')
            vv =self.init_float([N, 1], 'spiking')

            # membrane potential time constant
            subnet = tf.concat(0, [tf.ones([N // 2, 1]), tf.zeros([N // 2, 1])])
            subnetout = tf.concat(0, [tf.zeros([N // 2, 1]), tf.ones([ N // 2, 1])])
            tauvSubnet = tf.Variable(subnet * 15 + subnetout * self.tauv, name="tauv")

            # debug
            wcontrol = tf.Variable(0, dtype='float32')
            LTPcontrol = tf.Variable(0, dtype='float32')
            LTDcontrol = tf.Variable(0, dtype='float32')
            dwcontrol = tf.Variable(0, dtype='float32')

            # monitoring variables
            spikes = self.init_float([self.startPlast*3, N], "spikes")
            vvmN1 = self.init_float([T], "vv1")
            vvmN2 = self.init_float([T], "vv2")
            pN1 = self.init_float([T], "b1")
            pN2 = self.init_float([T], "b2")
            i1 = self.init_float([T], "i1")
            i2 = self.init_float([T], "i2")
            iEffm = self.init_float([T], "noise")

            weight_step = self.weight_step
            g1m = self.init_float([T//weight_step], "gamma_N1")
            g2m = self.init_float([T//weight_step], "gamma_N2")
            gSm = self.init_float([T//weight_step], "gamma_NS")

            # currents
            iBack =self.init_float([N, 1], 'iBack')
            iEff =self.init_float([N, 1], 'iEff')
            iGap =self.init_float([N, 1], 'iGap')
            iChem =self.init_float([N, 1], 'iChem')


            conn = tf.constant(self.conn, name="N1_N2_NS")
            conn0 = tf.constant(self.conn0, name="N1_N2")
            conn1 = tf.constant(self.conn1, name="N1")
            conn2 = tf.constant(self.conn2, name="N2")
            connS = tf.constant(self.connS, name="NS")

            input = tf.cast(tf.constant(self.input), tf.float32)
            sumSubnetGap = ((nbInCluster-sG)*(nbInCluster-sG-1))**0.5 * 2

            # plasticity learning rates
            A_LTD_ = 2.45e-5 * self.FACT * 400 / N
            A_LTD = tf.constant(A_LTD_, name="A_LTP")
            A_LTP = tf.constant(self.ratio * A_LTD_, name="A_LTD")

            # stimulation
            TImean = tf.constant(self.nu*1.0, name="mean_input_current")
            # timestep
            dt = tf.constant(self.dt * 1.0, name="timestep")
            connectTime = self.connectTime
            startPlast = self.startPlast
            stabTime = self.stabTime
            # connection and plasticity times
            sim_index = tf.Variable(-1.0, name="sim_index")
            one = tf.Variable(1.0)


            if self.initWGap==True:
                '''
                We initialize the gap junctions with steady state values previously calculated
                '''
                try:
                    g_1 = getGSteady(15, self.nu, 1000)
                except:
                    g_1 = 10
                g0_1 = g_1 / sumSubnetGap
                wGap_init_1 = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                          seed=None, name=None) * (1 - 0.001) + 0.001) * g0_1
                wGap_init_1 = tf.mul(wGap_init_1, conn1)

                try:
                    g_2 = getGSteady(self.tauv, self.nu, 1000)
                except:
                    g_2 = 10
                g0_2 = g_2 / sumSubnetGap
                wGap_init_2 = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                          seed=None, name=None) * (1 - 0.001) + 0.001) * g0_2
                wGap_init_2 = tf.mul(wGap_init_2, conn2)

                wGap_init0 = wGap_init_1 + wGap_init_2


            elif self.initWGap == -1:
                '''
                We initialize the gap junctions to arbitrary values, g1, g2, and (g1+g2)/2
                '''
                g_1 = self.g1
                g0_1 = g_1 / sumSubnetGap
                wGap_init_1 = (tf.random_normal((N, N), mean=g0_1, stddev=g0_1/2, dtype=tf.float32,
                                          seed=None, name=None))
                wGap_init_1 = tf.mul(wGap_init_1, conn1)

                g_2 = self.g2
                g0_2 = g_2 / sumSubnetGap
                wGap_init_2 = (tf.random_normal((N, N), mean=g0_2, stddev=g0_2/2, dtype=tf.float32,
                                          seed=None, name=None))
                wGap_init_2 = tf.mul(wGap_init_2, conn2)

                wGap_init0 = wGap_init_1 + wGap_init_2

            else:
                '''
                We initialize the gap junctions to a unique arbitrary values g0
                '''
                g0 = self.g / (N/2)
                random_mat = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                          seed=None, name=None) * (1 - 0.001) + 0.001) * g0
                wGap_init0 = tf.mul(random_mat, (conn1+conn2))

            wII_init = tf.ones((N, N), dtype=tf.float32) * tf.to_float(700 / (tf.reduce_sum(conn1)**0.5) / self.dt)

            wGap = tf.Variable(wGap_init0)
            WII = tf.Variable(tf.mul(wII_init, conn))

        #################################################################################
        ## Computation
        #################################################################################
        with tf.device(self.device):
            with tf.name_scope('Connect'):
                g0_S = tf.reduce_mean(wGap*conn1)*2 + tf.reduce_mean(wGap*conn2)
                wGapS = tf.mul(tf.random_normal((N, N), mean=g0_S, stddev=g0_S / 2, dtype=tf.float32,
                                                seed=None, name=None), connS)
                connect = tf.group(
                    wGap.assign(tf.add(wGap, wGapS))
                )

            with tf.name_scope('Currents'):
                # divide the weights by 2 at connection time
                # wgap = wgap / (1 + tf.cast(tf.equal(sim_index, connectTime), tf.float32))
                iGap_ = tf.matmul(wGap, v, name="GJ1") - tf.mul(tf.reshape(tf.reduce_sum(wGap, 0), (N, 1)), v, name="GJ2")
                iChem_ = iChem + dt / 5 * (-iChem + tf.matmul(WII, tf.to_float(vv), name="IPSPs"))

                # current
                iBack_ = iBack + dt / 5 * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
                                                                     seed=None, name=None))
                input_ = tf.gather(input, sim_index)
                # ind_ = tf.transpose(tf.one_hot(sim_index, N, dtype=tf.float32, name='one_hot'))
                # input_ = tf.matmul(ind_, input)


                if self.both:
                    # input to both subnet
                    iEff_ = iBack_ * scaling + input_ + TImean
                else:
                    # input first subnet
                    iEff_ = iBack_ * scaling + input_ * subnet + TImean

                # sum all currents
                I_ = iGap_ + iChem_ + iEff_

            # IZHIKEVICH
            with tf.name_scope('Izhikevich'):
                ind_ = ind + 1
                # voltage
                v_ = v + dt / tauvSubnet * (tf.mul((v + 60), (v + 50)) - 20 * u + 8 * I_)
                # adaptation
                u_ = u + dt * 0.044 * (v_ + 55 - u)

                # spikes
                vv_ = tf.to_float(tf.greater(v_, 25.0))
                vvN1_ = vv_*subnet
                vvN2_ = vv_*subnetout
                # reset
                v_ = tf.mul(vv_, -40.0) + tf.mul((1 - vv_), v_)


                u_ = u_ + tf.mul(vv_, (50.0))

            # bursting
            with tf.name_scope('bursting'):
                LowSp_ = LowSp + dt / 8.0 * (vv_ * 8.0 / dt - LowSp)
                p_ = tf.to_float(tf.greater(LowSp_, 1.5))
                pN1_ = p_*subnet
                pN2_ = p_*subnetout

            # plasticity
            with tf.name_scope('plasticity'):
                A = tf.matmul(p_, tf.ones((1, N)), name="bursts")  # bursts
                B = tf.matmul(vv_, tf.ones((1, N)), name="spikes")  # spikes

                dwLTD_ = A_LTD * tf.add(A , tf.transpose(A, name="tr_bursts"))
                dwLTP_ = A_LTP * tf.add(B , tf.transpose(B, name="tr_spikes"))

                dwGap_ = dt * tf.sub(dwLTP_ , dwLTD_)

                wgap = tf.clip_by_value(wGap + dwGap_, clip_value_min=0, clip_value_max=10 ** 10)
                '''
                multiply by 0 where there is no gap junction (upper right and lower left of
                the connection matrix
                '''
                wGap_before_ = tf.mul(wgap, conn0)
                wGap_after_ = tf.mul(wgap, conn)

            # debug
            with tf.name_scope('debug'):
                LTDcontrol_ = tf.reduce_sum(dwLTD_)
                LTPcontrol_ =  tf.reduce_sum(dwLTP_)
                wcontrol_ = tf.reduce_sum(wGap)
                dwcontrol_ = tf.reduce_sum(dwGap_)

            # monitoring
            with tf.name_scope('Monitoring'):
                vvmeanN1_ = tf.reduce_sum(vvN1_)
                vvmeanN2_ = tf.reduce_sum(vvN2_)
                pmeanN1_ = tf.reduce_sum(pN1_)
                pmeanN2_ = tf.reduce_sum(pN2_)
                imean1_ = tf.reduce_mean(I_ * subnet)
                imean2_ = tf.reduce_mean(I_ * subnetout)
                iEffm_ = tf.reduce_mean(iEff_)
                update = tf.group(
                    tf.scatter_update(vvmN1, sim_index, vvmeanN1_),
                    tf.scatter_update(vvmN2, sim_index, vvmeanN2_),
                    tf.scatter_update(pN1, sim_index, pmeanN1_),
                    tf.scatter_update(pN2, sim_index, pmeanN2_),
                    tf.scatter_update(i1, sim_index, imean1_),
                    tf.scatter_update(i2, sim_index, imean2_),
                    tf.scatter_update(iEffm, sim_index, iEffm_),
                )

            with tf.name_scope('Weights_monitoring'):
                g1m_ = tf.reduce_sum(wGap*conn1)
                g2m_ = tf.reduce_sum(wGap*conn2)
                gSm_ = tf.reduce_sum(wGap*connS)
                update_weights = tf.group(
                    tf.scatter_update(g1m, tf.to_int32(sim_index/weight_step), g1m_),
                    tf.scatter_update(g2m, tf.to_int32(sim_index/weight_step), g2m_),
                    tf.scatter_update(gSm, tf.to_int32(sim_index/weight_step), gSm_),
                )

            with tf.name_scope('Raster_Plot'):
                spike_update = tf.group(
                    tf.scatter_update(spikes, sim_index, tf.reshape((vv_), (N,))),
                    spike_index.assign(tf.add(spike_index, one)),
                )

            # Operation to update the state
            step = tf.group(
                v.assign(v_),
                vv.assign(vv_),
                u.assign(u_),
                iBack.assign(iBack_),
                iEff.assign(iEff_),
                LowSp.assign(LowSp_),
            )

            # debug unstability
            debug = tf.group(
                wcontrol.assign(wcontrol_),
                LTPcontrol.assign(LTPcontrol_),
                LTDcontrol.assign(LTDcontrol_),
                dwcontrol.assign(dwcontrol_),
            )

            # plasticity
            plast_before = tf.group(
                wGap.assign(wGap_before_),
                        )
            plast_after = tf.group(
                wGap.assign(wGap_after_),
            )

            update_index = tf.group(
                sim_index.assign(tf.add(sim_index, one)),
            )

        # initialize the graph
        tf.global_variables_initializer().run()

        if self.profiling:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        else:
            self.run_metadata = None
            self.run_options = None

        t0 = time.time()
        for i in range(T):
            # Step simulation
            ops = {'before': [update_index, step, plast_before, update],
                   'after': [update_index, step, plast_after, update],
                   'static': [update_index, step, update]
                   }
            # if self.spikeMonitor:
            #     for k, v in ops.items():
            #         ops[k] = v + [spike_update]

            if i == connectTime:
                self.sess.run([connect],
                              options=self.run_options,
                              run_metadata=self.run_metadata
                              )

            # plasticity before connection
            if i>startPlast and i<connectTime:
                self.sess.run(ops['before'],
                              options=self.run_options,
                              run_metadata=self.run_metadata
                              )
            # just after connection, let the network stabilize before fixing plasticity
            elif i>connectTime and i<=(connectTime + stabTime):
                self.sess.run(ops['after'],
                              options=self.run_options,
                              run_metadata=self.run_metadata
                              )
            # after connection, the network has been static for startPlast time
            elif i>=(connectTime + startPlast + stabTime):
                self.sess.run(ops['after'],
                              options=self.run_options,
                              run_metadata=self.run_metadata
                              )
            # network is static
            else:
                self.sess.run(ops['static'],
                              options=self.run_options,
                              run_metadata=self.run_metadata
                              )

            if self.spikeMonitor:
                if i <= startPlast \
                        or (i >= (connectTime +stabTime) and i < (connectTime+stabTime+startPlast)) \
                        or i >= (T-startPlast):
                    self.sess.run([spike_update],
                                  options=self.run_options,
                                  run_metadata=self.run_metadata
                                  )

            if self.debug:
                self.sess.run([debug])

                if i==0:
                    self.wcontrol = []
                    self.dwcontrol = []
                    self.LTDcontrol = []
                    self.LTPcontrol = []

                self.wcontrol.append(wcontrol.eval())
                self.dwcontrol.append(dwcontrol.eval())
                self.LTDcontrol.append(LTDcontrol.eval())
                self.LTPcontrol.append(LTPcontrol.eval())

            if i % weight_step == 0:
                self.sess.run([update_weights])

            if i==0:
                self.w0 = wGap.eval()
            elif i==T-1:
                self.wE = wGap.eval()

            #

        # monitoring variables
        self.vvmN1 = vvmN1.eval()
        self.vvmN2 = vvmN2.eval()
        self.pN1 = pN1.eval()
        self.pN2 = pN2.eval()
        self.i1 = i1.eval()
        self.i2 = i2.eval()
        self.iEff = iEffm.eval()
        self.gammaN1 = g1m.eval() / np.sum(self.conn1)
        self.gammaN2 = g2m.eval() / np.sum(self.conn2)
        self.gammaNS = gSm.eval() / np.sum(self.connS)
        if self.spikeMonitor:
            self.raster = spikes.eval()

        # profiling information
        # Create the Timeline object, and write it to a json
        if self.profiling:
            tl = timeline.Timeline(self.run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)

        print('\n%.2f\n' % (time.time() - t0))
        self.sess.close()
