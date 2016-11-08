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

class TfSingleNet:
    # DEVICE = '/gpu:0'


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
                 startPlast=500,
                 memfraction=1):
        tf.reset_default_graph()
        self.N = N
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
        gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memfraction)

        self.sess = tf.InteractiveSession(config=tf.ConfigProto(
            inter_op_parallelism_threads=NUM_CORES,
            intra_op_parallelism_threads=NUM_CORES,
            gpu_options=gpu_options,
            device_count={'GPU': (device[:4] == '/gpu') * 1})
        )
        if input is None:
            self.input = np.ones((T, 1), dtype='int32')

    def to_JSON(self):
        return json.dumps(self, default=lambda o: o.__dict__,
                          sort_keys=True, indent=4)

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
        T = self.T
        with tf.device(self.device):
        #if 1:
            dt = tf.placeholder(tf.float32, shape=(), name='dt')
            tauv = tf.placeholder(tf.float32, shape=(), name='tauv')
            sim_index = tf.placeholder(tf.int32, shape=(), name='sim_index')

            scaling = 1 / (1 / (2 * 2 / self.dt)) ** 0.5 * 70

            # Create variables for simulation state
            u = self.init_float([N, 1], 'u')
            v = self.init_float([N, 1], 'v')
            p = self.init_float([N, 1], 'v')

            LowSp = self.init_float([N, 1], 'bursting')
            vv = self.init_float([N, 1], 'spiking')

            vmean = tf.Variable(0, dtype='float32')
            umean = tf.Variable(0, dtype='float32')
            vvmean = tf.Variable(0, dtype='float32')
            pmean = tf.Variable(0, dtype='float32')
            imean = tf.Variable(0, dtype='float32')
            gammamean = tf.Variable(0, dtype='float32')
            iEffm = tf.Variable(0, dtype='float32')
            iGapm = tf.Variable(0, dtype='float32')
            iChemm = tf.Variable(0, dtype='float32')

            # currents
            iBack = self.init_float([N, 1], 'iBack')
            iEff = self.init_float([N, 1], 'iEff')
            iChem = self.init_float([N, 1], 'iChem')
            iGap = self.init_float([N, 1], 'iChem')

            # synaptics connection
            conn = np.ones((N, N), dtype='float32') - np.diag(np.ones((N,), dtype='float32'))
            nbOfGaps = N*(N-1)

            input = tf.cast(tf.constant(self.input), tf.float32)

            if self.g0fromFile:
                self.g = getGSteady(self.tauv, 5, 1000)
            g0 = self.g / (nbOfGaps**0.5)
            wGap_init = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                          seed=None, name=None) * (1 - 0.001) + 0.001) * g0
            wII_init = tf.ones((N, N), dtype=tf.float32)* 700 / (nbOfGaps**0.5) / self.dt

            wGap = tf.Variable(tf.mul(wGap_init, conn))

            WII = tf.Variable(tf.mul(wII_init, conn))

            A_LTD =  2.45e-5 * self.FACT * 400 / N
            A_LTP = self.ratio * A_LTD
            lowspthresh = self.lowspthresh

            # stimulation
            TImean = self.nu

            self.spikes = self.init_float([T, N], "spikes")

            net = tf.ones((N,1))
            tauvSubnet = net * self.tauv

        #################################################################################
        ## Computation
        #################################################################################
        with tf.device(self.device):
        #if 1:
            with tf.name_scope('Currents'):
                # Discretized PDE update rules
                iChem_ = iChem + dt / 5 * (-iChem + tf.matmul(WII, tf.to_float(vv)))

                # current
                iBack_ = iBack + dt / 5 * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
                                                                     seed=None, name=None))
                input_ = tf.gather(input, sim_index)

                # input to network: colored noise + external input
                iEff_ = iBack_ * scaling + input_ + TImean

                iGap_ = tf.matmul(wGap, v) - tf.mul(tf.reshape(tf.reduce_sum(wGap, 0), (N, 1)), v)

                I_ = iGap_ + iChem_ + iEff_

            # IZHIKEVICH
            with tf.name_scope('Izhikevich'):
                # voltage
                v_ = v + dt / tauvSubnet * (tf.mul((v + 60), (v + 50)) - 20 * u + 8 * I_)
                # adaptation
                u_ = u + dt * 0.044 * (v_ + 55 - u)
                # spikes
                vv_ = tf.to_float(tf.greater(v_, 25.0))
                # reset
                v_ = tf.mul(vv_, -40.0) + tf.mul((1 - vv_), v_)

                u_ = u_ + tf.mul(vv_, (50.0))

            # bursting
            with tf.name_scope('bursting'):
                LowSp_ = LowSp + dt / 8.0 * (vv_ * 8.0 / dt - LowSp)
                p_ = tf.to_float(tf.greater(LowSp_, lowspthresh))

            # plasticity
            with tf.name_scope('plasticity'):
                A = tf.matmul(p_, tf.ones((1, N))) # bursts
                B = tf.matmul(vv_, tf.ones((1, N))) # spikes

                dwLTD_ = A_LTD * (A + tf.transpose(A))
                dwLTP_ = A_LTP * (B + tf.transpose(B))

                dwGap_ = dt * (dwLTP_ - dwLTD_) * tf.cast((sim_index > self.startPlast), tf.float32)
                wGap_ = tf.clip_by_value(wGap + dwGap_, clip_value_min=0, clip_value_max=10 ** 10)

            # monitoring
            with tf.name_scope('Monitoring'):
                vmean_ = tf.reduce_mean(v_)
                umean_ = tf.reduce_mean(u_)
                imean_ = tf.reduce_mean(I_)
                vvmean_ = tf.reduce_mean(tf.to_float(vv_))
                pmean_ = tf.reduce_mean(p_)
                gammamean_ = tf.reduce_mean(wGap_)


            with tf.name_scope('Raster_Plot'):
                spike_update = tf.scatter_update(self.spikes, sim_index, tf.reshape((vv_), (N,)))

            # Operation to update the state
            step = tf.group(
                v.assign(v_),
                u.assign(u_),
                vv.assign(vv_),

                iBack.assign(iBack_),
                # iEff.assign(iEff_),
                
                
                LowSp.assign(LowSp_),

                vmean.assign(vmean_),
                umean.assign(umean_),
                imean.assign(imean_),
                vvmean.assign(vvmean_),
                pmean.assign(pmean_),
                # p.assign(p_),
                iEffm.assign(tf.reduce_mean(iEff_)),
                iGapm.assign(tf.reduce_mean(iGap_)),
                iChemm.assign(tf.reduce_mean(iChem_)),
            )

            plast = tf.group(
                wGap.assign(wGap_),
                gammamean.assign(gammamean_),
            )

        self.sess.run(tf.initialize_all_variables())
        if 1:
            self.vm = np.empty(T)
            self.um = np.empty(T)
            self.vvm = np.empty(T)
            self.bursts = np.empty(T)
            self.im = np.empty(T)
            self.gamma = np.empty(T)
            self.iEff = np.empty(T)
            self.iGap = np.empty(T)
            self.iChem = np.empty(T)
            self.lowsp = np.empty((T,N))

            t0 = time.time()
            for i in range(T):
                # Step simulation

                # start without plasticity until t>startPlast
                if i < self.startPlast:
                    self.sess.run([step], feed_dict={dt: self.dt, tauv: self.tauv, sim_index: i})
                else:
                    self.sess.run([step, plast], feed_dict={dt: self.dt, tauv: self.tauv, sim_index: i})

                if self.spikeMonitor:
                    feed = {dt: self.dt, tauv: self.tauv, sim_index: i}
                    self.sess.run(spike_update, feed_dict=feed)
                # Visualize every 50 steps
                if i % 1 == 0:
                    if self.disp:
                        clear_output(wait=True)
                        self.DisplayArray(wGap.eval(), rng=[0, 1.5 * g0], text="%.2f ms" % (i * self.dt))
                    self.vm[i] = vmean.eval()
                    self.vvm[i] = (vvmean.eval())
                    self.um[i] = (umean.eval())
                    self.im[i] = (imean.eval())
                    
                    self.iEff[i] = (iEffm.eval())
                    self.iGap[i] = (iGapm.eval())
                    self.iChem[i] = (iChemm.eval())
                    
                    self.gamma[i] = (gammamean.eval() * (nbOfGaps**0.5))
                    self.bursts[i] = (pmean.eval())
                    self.lowsp[i] = LowSp.eval().reshape(N)

            self.raster = self.spikes.eval()
            self.burstingActivity = np.mean(self.bursts)
            self.spikingActivity = np.mean(self.vvm)


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
        self.connectTime = 0
        self.FACT = 10
        self.ratio = 1
        self.weight_step = 100
        self.nu = nu
        self.profiling = profiling
        if self.profiling:
            self.run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
            self.run_metadata = tf.RunMetadata()
        else:
            self.run_metadata = None
            self.run_options = None
        self.sess = tf.InteractiveSession(config=tf.ConfigProto(
            inter_op_parallelism_threads=NUM_CORES,
            intra_op_parallelism_threads=NUM_CORES,
            device_count={'GPU': (device[:4]=='/gpu')*1},
            gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memfraction),
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

            conn = tf.Variable(tf.ones((N,N)))
            conn0 = tf.Variable(tf.ones((N,N)))
            conn1 = tf.Variable(tf.ones((N,N)))
            conn2 = tf.Variable(tf.ones((N,N)))
            connS = tf.Variable(tf.ones((N,N)))

            synapses = tf.group(
                conn.assign(conn_),
                conn0.assign(conn0_),
                conn1.assign(conn1_),
                conn2.assign(conn2_),
                connS.assign(connS_)
            )
            self.sess.run(tf.initialize_all_variables())
            self.sess.run([synapses],
                          # options=self.run_options,
                          # run_metadata=self.run_metadata
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
            subnet = tf.concat(0, [tf.ones([N // 2, 1]), tf.zeros([N - N // 2, 1])])
            subnetout = tf.concat(0, [tf.zeros([N // 2, 1]), tf.ones([N - N // 2, 1])])
            tauvSubnet = tf.Variable(subnet * 15 + subnetout * self.tauv, "tauv")

            # debug
            wcontrol = tf.Variable(0, dtype='float32')
            LTPcontrol = tf.Variable(0, dtype='float32')
            LTDcontrol = tf.Variable(0, dtype='float32')
            dwcontrol = tf.Variable(0, dtype='float32')

            # monitoring variables
            self.spikes = self.init_float([T, N], "spikes")
            vvmN1 = self.init_float([T,1], "vv1")
            vvmN2 = self.init_float([T,1], "vv2")
            i1 = self.init_float([T,1], "i1")
            i2 = self.init_float([T,1], "i2")
            iEffm = self.init_float([T,1], "noise")
            plastON = self.init_float([T,1], "plasticity_is_ON")
            weight_step = self.weight_step
            g1m = self.init_float([T//weight_step,1], "gamma_N1")
            g2m = self.init_float([T//weight_step,1], "gamma_N2")
            gSm = self.init_float([T//weight_step,1], "gamma_NS")

            # currents
            iBack =self.init_float([N, 1], 'iBack')
            iEff =self.init_float([N, 1], 'iEff')
            iGap =self.init_float([N, 1], 'iGap')
            iChem =self.init_float([N, 1], 'iChem')


            conn1 = tf.constant(self.conn1, name="N1")
            conn2 = tf.constant(self.conn2, name="N2")
            connS = tf.constant(self.connS, name="NS")

            # connection matrices for before and after the connection is made
            allowedConnections = tf.constant(self.conn, name='with_shared')
            allowedConnections0 = tf.constant(self.conn0, name='without_shared')

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
            # connection and plasticity times
            connectTime = tf.constant(self.connectTime*1.0, name="connection_time")
            startPlast = tf.constant(self.startPlast*1.0, name="plasticity_starting_time")
            sim_index = tf.Variable(0.0, name="sim_index")
            one = tf.constant(1.0)
            plastON = tf.Variable(0.0, name='plast_ON')


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

                g0_S = (g0_1 + g0_2) / 2
                wGap_init_S = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                          seed=None, name=None) * (1 - 0.001) + 0.001) * g0_S
                wGap_init_S = tf.mul(wGap_init_S, connS)


                # wGap_init = wGap_init_1 + wGap_init_2 + wGap_init_S
                wGap_init0 = wGap_init_1 + wGap_init_2


            elif self.initWGap == -1:
                '''
                We initialize the gap junctions to arbitrary values, g1, g2, and (g1+g2)/2
                '''
                g_1 = self.g1
                g0_1 = g_1 / sumSubnetGap
                wGap_init_1 = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                          seed=None, name=None) * (1 - 0.001) + 0.001) * g0_1
                wGap_init_1 = tf.mul(wGap_init_1, conn1)

                g_2 = self.g2
                g0_2 = g_2 / sumSubnetGap
                wGap_init_2 = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                          seed=None, name=None) * (1 - 0.001) + 0.001) * g0_2
                wGap_init_2 = tf.mul(wGap_init_2, conn2)

                g0_S = (g0_1 + g0_2) / 2
                wGap_init_S = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                          seed=None, name=None) * (1 - 0.001) + 0.001) * g0_S
                wGap_init_S = tf.mul(wGap_init_S, connS)

                # wGap_init = wGap_init_1 + wGap_init_2 + wGap_init_S
                wGap_init0 = wGap_init_1 + wGap_init_2

            else:
                '''
                We initialize the gap junctions to a unique arbitrary values g0
                '''
                g0 = self.g / (N/2)
                random_mat = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                          seed=None, name=None) * (1 - 0.001) + 0.001) * g0
                # wGap_init = random_mat * (conn1+conn2+connS)
                wGap_init0 = tf.mul(random_mat, (conn1+conn2))
                wGap_init_S = tf.mul(random_mat, connS)

            # gmean_init = np.mean(wGap_init)
            wII_init = tf.ones((N, N), dtype=tf.float32) * tf.to_float(700 / (tf.reduce_sum(conn1)**0.5) / self.dt)

            # # just to check
            # self.wGap_init_S = wGap_init_S.eval()
            # self.wGap_init0 = wGap_init0.eval()

            wGap = tf.Variable(wGap_init0)
            wGapS = tf.Variable(wGap_init_S)
            WII = tf.Variable(tf.mul(wII_init, conn))
            zer = tf.Variable(tf.zeros((N, N), dtype=tf.float32))

            # because no if/else in tensorflow, we need to define functions that will be chosen
            # depending on the result of the boolean in tf.cond(bool, fn1, fn2)
            def fn1():
                return wGapS

            def fn2():
                return zer

            def fn3():
                return allowedConnections

            def fn4():
                return allowedConnections0
        #################################################################################
        ## Computation
        #################################################################################
        with tf.device(self.device):
            with tf.name_scope('Currents'):
                # Discretized PDE update rules
                wgap = tf.add(wGap, tf.cond(tf.equal(sim_index, connectTime), fn1, fn2))
                # divide the weights by 2 at connection time
                # wgap = wgap / (1 + tf.cast(tf.equal(sim_index, connectTime), tf.float32))
                iGap_ = tf.matmul(wgap, v) - tf.mul(tf.reshape(tf.reduce_sum(wgap, 0), (N, 1)), v)
                iChem_ = iChem + dt / 5 * (-iChem + tf.matmul(WII, tf.to_float(vv)))

                # current
                iBack_ = iBack + dt / 5 * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
                                                                     seed=None, name=None))
                input_ = tf.gather(input, tf.to_int32(sim_index))

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
                vvN1_ = tf.to_float(tf.greater(v_*subnet, 25.0))
                vvN2_ = tf.to_float(tf.greater(v_*subnetout, 25.0))
                # reset
                v_ = tf.mul(vv_, -40.0) + tf.mul((1 - vv_), v_)


                u_ = u_ + tf.mul(vv_, (50.0))

            # bursting
            with tf.name_scope('bursting'):
                LowSp_ = LowSp + dt / 8.0 * (vv_ * 8.0 / dt - LowSp)
                p_ = tf.to_float(tf.greater(LowSp_, 1.5))

            # plasticity
            with tf.name_scope('plasticity'):
                A = tf.matmul(p_, tf.ones((1, N)))  # bursts
                B = tf.matmul(vv_, tf.ones((1, N)))  # spikes

                dwLTD_ = A_LTD * (A + tf.transpose(A))
                dwLTP_ = A_LTP * (B + tf.transpose(B))

                plastON_ = tf.logical_or(tf.logical_and(sim_index > startPlast,sim_index < (connectTime + 1000)),
                                   sim_index > (connectTime + startPlast + 1000))
                plastON_ = tf.cast(plastON_, tf.float32)
                dwGap_ = dt * (dwLTP_ - dwLTD_) * plastON_
                '''
                multiply by 0 where there is no gap junction (upper right and lower left of the connection matrix
                '''
                wGap_ = tf.mul(tf.clip_by_value(wgap + dwGap_, clip_value_min=0, clip_value_max=10 ** 10),
                               tf.cond((sim_index >= connectTime), fn3, fn4))

            # debug
            with tf.name_scope('debug'):
                LTDcontrol_ = tf.reduce_sum(dwLTD_)
                LTPcontrol_ =  tf.reduce_sum(dwLTP_)
                wcontrol_ = tf.reduce_sum(wgap)
                dwcontrol_ = tf.reduce_sum(dwGap_)

            # monitoring
            with tf.name_scope('Monitoring'):
                vvmeanN1_ = tf.reduce_sum(tf.to_float(vvN1_))
                vvmeanN2_ = tf.reduce_sum(tf.to_float(vvN2_))
                imean1_ = tf.reduce_mean(I_ * subnet)
                imean2_ = tf.reduce_mean(I_ * subnetout)
                iEffm_ = tf.reduce_mean(iEff_)
                update = tf.group(
                    tf.scatter_update(vvmN1, tf.to_int32(sim_index), tf.reshape(tf.reduce_sum(tf.to_float(vvN1_)), (1,))),
                    tf.scatter_update(vvmN2, tf.to_int32(sim_index), tf.reshape(tf.reduce_sum(tf.to_float(vvN2_)), (1,))),
                    tf.scatter_update(i1, tf.to_int32(sim_index), tf.reshape(imean1_, (1,))),
                    tf.scatter_update(i2, tf.to_int32(sim_index), tf.reshape(imean2_, (1,))),
                    tf.scatter_update(iEffm, tf.to_int32(sim_index), tf.reshape(iEffm_, (1,))),
                )

            with tf.name_scope('Weights_monitoring'):
                g1m_ = tf.reduce_sum(wGap_*conn1)
                g2m_ = tf.reduce_sum(wGap_*conn2)
                gSm_ = tf.reduce_sum(wGap_*connS)
                update_weights = tf.group(
                    tf.scatter_update(g1m, tf.to_int32(sim_index/weight_step), tf.reshape(g1m_, (1,))),
                    tf.scatter_update(g2m, tf.to_int32(sim_index/weight_step), tf.reshape(g2m_, (1,))),
                    tf.scatter_update(gSm, tf.to_int32(sim_index/weight_step), tf.reshape(gSm_, (1,))),
                )

            with tf.name_scope('Raster_Plot'):
                spike_update = tf.scatter_update(self.spikes, tf.to_int32(sim_index), tf.reshape((vv_), (N,)))


            # Operation to update the state
            step = tf.group(
                sim_index.assign(tf.add(sim_index, one)),
                v.assign(v_),
                plastON.assign(plastON_),
                vv.assign(vv_),
                u.assign(u_),
                iBack.assign(iBack_),
                iEff.assign(iEff_),
                LowSp.assign(LowSp_),
                wGap.assign(wGap_),
            )

            # debug unstability
            debug = tf.group(
                wcontrol.assign(wcontrol_),
                LTPcontrol.assign(LTPcontrol_),
                LTDcontrol.assign(LTDcontrol_),
                dwcontrol.assign(dwcontrol_),
            )

            # plasticity
            plast = tf.group(
                            wGap.assign(wGap_),
                        )

        # initialize the graph
        self.sess.run(tf.initialize_all_variables())

        # initialize and fill the var arrays (TODO: do in on the device, not the host)

        self.gamma = []
        self.gammaN1 = []
        self.gammaN2 = []
        self.gammaNS = []
        self.gammaTest = []
        self.raster = []
        # self.plastON = []

        t0 = time.time()
        for i in range(T):
            # Step simulation
            if self.spikeMonitor:
                # we save the raster plot
                if i >= self.startPlast or i == self.connectTime:
                    self.sess.run([step, plast, update, spike_update])
                else:
                    self.sess.run([step, update, spike_update])
            else:
                # we don't save the raster plot
                if i >= self.startPlast or i == self.connectTime:
                    self.sess.run([step, plast, update],
                                  options=self.run_options,
                                  run_metadata=self.run_metadata
                                  )
                else:
                    self.sess.run([step, update],
                                  options=self.run_options,
                                  run_metadata=self.run_metadata
                                  )

            if self.debug:
                self.sess.run([debug],
                              options=self.run_options,
                              run_metadata=self.run_metadata
                              )

                if i==0:
                    self.wcontrol = []
                    self.dwcontrol = []
                    self.LTDcontrol = []
                    self.LTPcontrol = []

                self.wcontrol.append(wcontrol.eval())
                self.dwcontrol.append(dwcontrol.eval())
                self.LTDcontrol.append(LTDcontrol.eval())
                self.LTPcontrol.append(LTPcontrol.eval())

            # self.plastON.append(plastON.eval())

            # if i % 100 == 0:
            #     weights = wGap.eval()
            #     self.gammaN1.append(np.mean(weights[:nbInCluster - sG, :nbInCluster - sG])*N)
            #     self.gammaN2.append(np.mean(weights[nbInCluster + sG:, nbInCluster + sG:])*N)
            #     # self.gammaTest.append(np.mean(weights[:nbInCluster - sG, nbInCluster + sG:])*N)
            #     self.gammaNS.append(np.mean(weights[nbInCluster - sG:nbInCluster + sG,
            #                                 nbInCluster - sG:nbInCluster + sG]) * N)
            #
            #     self.gamma.append(gammamean.eval())

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
        self.i1 = i1.eval()
        self.i2 = i2.eval()
        self.iEff = iEffm.eval()
        self.plastON = plastON.eval()
        self.gammaN1 = g1m.eval() / np.sum(self.conn1)
        self.gammaN2 = g2m.eval() / np.sum(self.conn2)
        self.gammaNS = gSm.eval() / np.sum(self.connS)
        if self.spikeMonitor:
            self.raster = self.spikes.eval()

        # profiling information
        # Create the Timeline object, and write it to a json
        if self.profiling:
            tl = timeline.Timeline(self.run_metadata.step_stats)
            ctf = tl.generate_chrome_trace_format()
            with open('timeline.json', 'w') as f:
                f.write(ctf)

        print('\n%.2f\n' % (time.time() - t0))
        self.sess.close()
