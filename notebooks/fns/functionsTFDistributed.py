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
                 memfraction=0.95):
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
        self.weight_step = 100
        self.gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memfraction)


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
        listWorkers = ["localhost:%d"%(2224+i) for i in range(50)]

        cluster = tf.train.ClusterSpec({"local": listWorkers})
        # time.sleep(0)


        for k in range(len(self.params)):
            T, N, self.g, self.nu, i, device, filename = self.params[k]
            server = tf.train.Server(cluster, job_name="local", task_index=k)

            with tf.device(tf.train.replica_device_setter(
                    worker_device="/job:local/task:%d" % k,
                    cluster=cluster)):
            # with tf.device(device):
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
                    vm = self.init_float([T], "vm")
                    um = self.init_float([T], "um")
                    vvm = self.init_float([T], "vvm")
                    pm = self.init_float([T], "pm")
                    lowspm = self.init_float([T], "lowspm")
                    im = self.init_float([T], "im")
                    gm = self.init_float([T//self.weight_step], "gm")
                    iEffm = self.init_float([T], "iEffm")
                    spikes = self.init_float([T, N], "spikes")

                with tf.name_scope('synaptic_connections'):
                    # synaptics connection
                    conn = tf.constant(np.ones((N, N), dtype='float32') - np.diag(np.ones((N,), dtype='float32')))
                    nbOfGaps = N*(N-1)


                    if self.g0fromFile:
                        self.g = getGSteady(self.tauv, 5, 1000)
                    g0 = self.g / (nbOfGaps**0.5)
                    wGap_init = (tf.random_normal((N, N), mean=0.0, stddev=1.0, dtype=tf.float32,
                                                  seed=None, name=None) * (1 - 0.001) + 0.001) * g0
                    wII_init = tf.ones((N, N), dtype=tf.float32) * 700 / (nbOfGaps**0.5) / self.dt
                    wGap = tf.Variable(tf.mul(wGap_init, conn))
                    WII = tf.Variable(tf.mul(wII_init, conn))

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
                spike_index = tf.Variable(0.0, name="sim_index")
                one = tf.Variable(1.0)
                ones = tf.ones((1, N))

            #################################################################################
            ## Computation
            #################################################################################
                with tf.name_scope('Currents'):
                    # Discretized PDE update rules
                    iChem_ = iChem + dt / 5 * (-iChem + tf.matmul(WII, tf.to_float(vv)))

                    # current
                    iBack_ = iBack + dt / 5 * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
                                                                         seed=None, name=None))
                    input_ = tf.gather(input, tf.to_int32(sim_index))

                    # input to network: colored noise + external input
                    iEff_ = iBack_ * scaling + input_ + TImean

                    iGap_ = tf.matmul(wGap, v) - tf.mul(tf.reshape(tf.reduce_sum(wGap, 0), (N, 1)), v)

                    I_ = iGap_ + iChem_ + iEff_

                # IZHIKEVICH
                with tf.name_scope('Izhikevich'):
                    # voltage
                    v_ = v + dt / tauv * (tf.mul((v + 60), (v + 50)) - 20 * u + 8 * I_)
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
                    p_ = tf.to_float(tf.greater(LowSp_, 1.5))

                # plasticity
                with tf.name_scope('plasticity'):
                    A = tf.matmul(p_, ones, name="bursts")  # bursts
                    B = tf.matmul(vv_, ones, name="spikes")  # spikes

                    dwLTD_ = A_LTD * tf.add(A, tf.transpose(A, name="tr_bursts"))
                    dwLTP_ = A_LTP * tf.add(B, tf.transpose(B, name="tr_spikes"))

                    dwGap_ = dt * tf.sub(dwLTP_, dwLTD_)
                    wGap_ = tf.clip_by_value(wGap + dwGap_, clip_value_min=0, clip_value_max=10 ** 10)

                # monitoring
                with tf.name_scope('Monitoring'):
                    vvmean_ = tf.reduce_sum(vv_)
                    vmean_ = tf.reduce_mean(v_)
                    umean_ = tf.reduce_mean(u_)
                    pmean_ = tf.reduce_mean(p_)
                    lowspmean_ = tf.reduce_mean(LowSp_)
                    imean_ = tf.reduce_mean(I_)
                    iEffm_ = tf.reduce_mean(iEff_)
                    update = tf.group(
                        tf.scatter_update(vvm, tf.to_int32(sim_index), vvmean_),
                        tf.scatter_update(vm, tf.to_int32(sim_index), vmean_),
                        tf.scatter_update(um, tf.to_int32(sim_index), umean_),
                        tf.scatter_update(pm, tf.to_int32(sim_index), pmean_),
                        tf.scatter_update(lowspm, tf.to_int32(sim_index), lowspmean_),
                        tf.scatter_update(im, tf.to_int32(sim_index), imean_),
                        tf.scatter_update(iEffm, tf.to_int32(sim_index), iEffm_),
                    )

                with tf.name_scope('Weights_monitoring'):
                    gm_ = tf.reduce_sum(wGap*conn)
                    update_weights = tf.group(
                        tf.scatter_update(gm, tf.to_int32(sim_index / weight_step), gm_),
                    )

                with tf.name_scope('Raster_Plot'):
                    spike_update = tf.group(
                        tf.scatter_update(spikes, tf.to_int32(spike_index), tf.reshape((vv_), (N,))),
                        spike_index.assign(tf.add(spike_index, one)),
                    )

                # Operation to update the state
                step = tf.group(
                    sim_index.assign(tf.add(sim_index, one)),
                    v.assign(v_),
                    vv.assign(vv_),
                    u.assign(u_),
                    iBack.assign(iBack_),
                    LowSp.assign(LowSp_),
                )

                plast = tf.group(
                    wGap.assign(wGap_),
                )

                self.sess = tf.Session(config=tf.ConfigProto(
                    inter_op_parallelism_threads=1,
                    intra_op_parallelism_threads=1,
                    gpu_options=self.gpu_options,
                    # device_count={'GPU': (device[:4] == '/gpu') * 1}
                )
                )

                # initialize the graph
                tf.global_variables_initializer().run(session=self.sess)

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

                    # if i % weight_step == 0:
                    #     self.sess.run([update_weights])

                    # Visualize every X steps
                    if i % 1 == 0:
                        if self.disp:
                            clear_output(wait=True)
                            self.DisplayArray(wGap.eval(), rng=[0, 1.5 * g0], text="%.2f ms" % (i * self.dt))

                    if i==0:
                        self.w0 = wGap.eval(session=self.sess)
                    elif i==T-1:
                        self.wE = wGap.eval(session=self.sess)

                    # monitoring variables
                    self.vvm = vvm.eval(session=self.sess)
                    self.p = pm.eval(session=self.sess)
                    self.lowsp = lowspm.eval(session=self.sess)
                    self.im = im.eval(session=self.sess)
                    self.iEff = iEffm.eval(session=self.sess)
                    self.gamma = gm.eval(session = self.sess) / np.sum(nbOfGaps)
                    if self.spikeMonitor:
                        self.raster = spikes.eval(session=self.sess)
                    self.burstingActivity = np.mean(self.p)
                    self.spikingActivity = np.mean(self.vvm)

                with open(filename, 'wb') as f:
                    np.savez(f, vvm=self.vvm, i=self.im, burst=self.burstingActivity, spike=self.spikingActivity)

        print('%.2f' % (time.time() - t0))

        self.sess.close()
