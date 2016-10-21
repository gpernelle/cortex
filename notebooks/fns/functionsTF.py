# from fns.utils import *
#
# import PIL.Image
# from io import BytesIO
# from IPython.display import clear_output, Image, display
#
# import tensorflow as tf
# import time
#
#
# flags = tf.app.flags
# FLAGS = flags.FLAGS
# flags.DEFINE_string('summaries_dir', '/tmp/tensorflow_logs', 'Summaries directory')
# flags.DEFINE_string('data_dir', '/tmp/data', 'Directory for storing data')
#
#
# def getGSteady(tauv, k, N=100):
#     '''
#     Get steady state value of the gap junction strenght
#     '''
#     df = pd.read_csv('gSteady.csv')
#     df2 = df[(df['tauv']==tauv) & (df['k']==k) & (df['N']==N)]
#     return df2['gSteady'].values[0]
#
# #
# #
# # class Tfnet:
# #     # DEVICE = '/gpu:0'
# #
# #
# #     def __init__(self, N=400, T=400, disp=False, spikeMonitor=False, input=None, tauv=15,
# #                  sG=10, device='/gpu:0', both=False, NUM_CORES=1, g0=7, startPlast=500):
# #         tf.reset_default_graph()
# #         self.N = N
# #         self.T = T
# #         self.disp = disp
# #         self.spikeMonitor = spikeMonitor
# #         self.tauv = tauv
# #         self.sG = sG
# #         self.g = g0
# #         self.device = device
# #         self.both = both
# #         self.initWGap = True
# #         self.startPlast = startPlast
# #         self.raster = []
# #         self.showProgress = False
# #         self.sess = tf.InteractiveSession(config=tf.ConfigProto(
# #             inter_op_parallelism_threads=NUM_CORES,
# #             intra_op_parallelism_threads=NUM_CORES,
# #             device_count={'GPU': (device[:4]=='/gpu')*1})
# #         )
# #         if input is None:
# #             self.input = np.ones((T,1), dtype='int32')
# #
# #     def to_JSON(self):
# #         return json.dumps(self, default=lambda o: o.__dict__,
# #                           sort_keys=True, indent=4)
# #
# #     def DisplayArray(self, a, fmt='jpeg', rng=[0,1], text=""):
# #         """Display an array as a picture."""
# #         a = (a - rng[0])/float(rng[1] - rng[0])*255
# #         a = np.uint8(np.clip(a, 0, 255))
# #         f = BytesIO()
# #         PIL.Image.fromarray(np.uint8(plt.cm.YlGnBu_r(a)*255)).save(f, fmt)
# #         display(Image(data=f.getvalue()))
# #         print(text)
# #
# #     def init_float(self, shape, name):
# #     #     return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)
# #         return tf.Variable(tf.zeros(shape), name=name)
# #
# #
# #     def variable_summaries(self, var, name):
# #         """Attach a lot of summaries to a Tensor."""
# #         with tf.name_scope('summaries'):
# #             mean = tf.reduce_mean(var)
# #             tf.scalar_summary('mean/' + name, mean)
# #             with tf.name_scope('stddev'):
# #                 stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
# #             tf.scalar_summary('sttdev/' + name, stddev)
# #             tf.scalar_summary('max/' + name, tf.reduce_max(var))
# #             tf.scalar_summary('min/' + name, tf.reduce_min(var))
# #             tf.histogram_summary(name, var)
# #         return 0
# #
# #
# #     def runTFSimul(self):
# #         #################################################################################
# #         ### INITIALISATION
# #         #################################################################################
# #         N = self.N
# #         T = self.T
# #         with tf.device(self.device):
# #             dt = tf.placeholder(tf.float32, shape=(), name='dt')
# #             tauv = tf.placeholder(tf.float32, shape=(), name='tauv')
# #             sim_index = tf.placeholder(tf.int32, shape=(), name='sim_index')
# #
# #             scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70
# #
# #             # Create variables for simulation state
# #             u =self.init_float([N, 1], 'u')
# #             v =self.init_float([N, 1], 'v')
# #             t = tf.Variable(0, dtype='float32')
# #             ind = tf.Variable(0, dtype='float32')
# #
# #             LowSp =self.init_float([N, 1], 'bursting')
# #             vv =self.init_float([N, 1], 'spiking')
# #
# #             vmean = tf.Variable(0, dtype='float32')
# #             umean = tf.Variable(0, dtype='float32')
# #             vvmean = tf.Variable(0, dtype='float32')
# #             vvmeanN1 = tf.Variable(0, dtype='float32')
# #             vvmeanN2 = tf.Variable(0, dtype='float32')
# #             imean = tf.Variable(0, dtype='float32')
# #             gammamean = tf.Variable(0, dtype='float32')
# #
# #             # currents
# #             iBack =self.init_float([N, 1], 'iBack')
# #             iEff =self.init_float([N, 1], 'iEff')
# #             iGap =self.init_float([N, 1], 'iGap')
# #             iChem =self.init_float([N, 1], 'iChem')
# #             # synaptics connection
# #             conn = np.zeros([N, N], dtype='float32')
# #             conn1 = np.zeros([N, N], dtype='float32')
# #             conn2 = np.zeros([N, N], dtype='float32')
# #             connS = np.zeros([N, N], dtype='float32')
# #             sG = self.sG
# #             nbInCluster = N//2
# #             for i in range(N):
# #                 for j in range(N):
# #                     conn[i][j] = (((i < (nbInCluster + sG)) & (j < (nbInCluster + sG))) \
# #                                  or ((i >= (nbInCluster - sG)) & (j >= (nbInCluster - sG)))) and (i!=j)
# #             for i in range(N):
# #                 for j in range(N):
# #                     conn1[i][j] = ((i < nbInCluster-sG) and (j < nbInCluster-sG) and (i!=j))
# #                     # conn1 = np.float32(conn1 != 0)
# #
# #             for i in range(N):
# #                 for j in range(N):
# #                     conn2[i][j] = ((i > nbInCluster+sG) and (j > nbInCluster+sG) and (i!=j))
# #                     # conn2 = np.float32(conn2 != 0)
# #
# #             for i in range(N):
# #                 for j in range(N):
# #                     connS[i][j] = ((i >= (nbInCluster - sG) and i <= (nbInCluster + sG)) \
# #                             or (j >= (nbInCluster - sG) and j <= (nbInCluster + sG))) and (i!=j)
# #                     # connS = np.float32(connS!=0)
# #
# #
# #             conn = conn1 + conn2 + connS
# #             conn =  np.float32(conn != 0)
# #             self.conn = conn
# #             self.conn1 = conn1
# #             self.conn2 = conn2
# #             self.connS = connS
# #
# #             allowedConnections = tf.Variable(conn)
# #             nbOfGaps = np.sum(conn)
# #
# #             input = tf.cast(tf.constant(self.input), tf.float32)
# #
# #             if self.initWGap:
# #                 g_1 = getGSteady(15, 5, 1000)
# #                 g0_1 = g_1 / nbOfGaps ** 0.5
# #                 wGap_init_1 = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_1
# #                 wGap_init_1 *= conn1
# #
# #                 g_2 = getGSteady(self.tauv, 5, 1000)
# #                 g0_2 = g_2 / nbOfGaps ** 0.5
# #                 wGap_init_2 = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_2
# #                 wGap_init_2 *= conn2
# #
# #                 g_S = (g_1 + g_2) / 2
# #                 g0_S = g_S / nbOfGaps ** 0.5
# #                 wGap_init_S = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_S
# #                 wGap_init_S *= connS
# #
# #                 wGap_init = wGap_init_1 + wGap_init_2 + wGap_init_S
# #
# #             else:
# #                 g0 = self.g / nbOfGaps ** 0.5
# #                 wGap_init = (np.random.random_sample([N, N]).astype(np.float32)*(1-0.001)+0.001) * g0
# #                 wGap_init *= conn
# #
# #             gmean_init = np.mean(wGap_init)
# #             wII_init = np.ones([N, N], dtype=np.float32) * 500 / N / 0.25
# #
# #             wGap = tf.Variable(wGap_init)
# #             WII = tf.Variable(wII_init * conn)
# #
# #             FACT = 100
# #             ratio = 15
# #             A_LTD = 1e-0 * 4.7e-6 * FACT * 400/N
# #             A_LTP = ratio * A_LTD
# #
# #             # low constant noise
# #             TImean_init = tf.ones([N, 1]) * 30
# #
# #             # stimulation
# #             TImean = 130.0
# #             TImean_simul = tf.ones([N, 1], dtype='float32') * TImean
# #
# #             self.spikes = self.init_float([T, N], "spikes")
# #
# #             subnet = tf.concat(0, [tf.ones([N//2, 1]), tf.zeros([N-N//2,1])])
# #             subnetout = tf.concat(0, [tf.zeros([N//2, 1]), tf.ones([N-N//2,1])])
# #             tauvSubnet = subnet * 15 + subnetout * self.tauv
# #
# #         #################################################################################
# #         ## Computation
# #         #################################################################################
# #         with tf.device(self.device):
# #             with tf.name_scope('Currents'):
# #                 # Discretized PDE update rules
# #                 iChem_ = iChem + dt / 5 * (-iChem + tf.matmul(WII, tf.to_float(vv)))
# #
# #                 # current
# #                 iBack_ = iBack + dt / 2 * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
# #                                                                      seed=None, name=None))
# #                 # iEff_ = iBack_ * scaling + tf.select(tf.greater(tf.ones([N, 1]) * t, 300), TImean_simul, TImean_init)
# #                 input_ = tf.gather(input, sim_index)
# #
# #                 if self.both:
# #                     # input to both subnet
# #                     iEff_ = iBack_ * scaling + tf.select(tf.greater(tf.ones([N, 1]) * input_, 0), TImean_simul, TImean_init)
# #                 else:
# #                     # input first subnet
# #                     iEff_ = iBack_ * scaling + tf.select(tf.greater(subnet * input_, 0), TImean_simul, TImean_init)
# #
# #                 iGap_ = tf.matmul(wGap, v) - tf.mul(tf.reshape(tf.reduce_sum(wGap, 0), (N, 1)), v)
# #
# #                 I_ = iGap_ + iChem_ + iEff_
# #
# #             # IZHIKEVICH
# #             with tf.name_scope('Izhikevich'):
# #                 ind_ = ind + 1
# #                 # voltage
# #                 v_ = v + dt / tauvSubnet * (tf.mul((v + 60), (v + 50)) - 20 * u + 8 * I_)
# #                 # adaptation
# #                 u_ = u + dt * 0.044 * (v_ + 55 - u)
# #                 # spikes
# #                 vv_ = tf.to_float(tf.greater(v_, 25.0))
# #                 vvN1_ = tf.to_float(tf.greater(v_*subnet, 25.0))
# #                 vvN2_ = tf.to_float(tf.greater(v_*subnetout, 25.0))
# #                 # reset
# #                 v_ = tf.mul(vv_, -40.0) + tf.mul((1 - vv_), v_)
# #                 u_ = u_ + tf.mul(vv_, (50.0))
# #
# #             # bursting
# #             with tf.name_scope('bursting'):
# #                 LowSp_ = LowSp + dt / 10.0 * (vv_ * 10.0 / dt - LowSp)
# #                 p = tf.to_float(tf.greater(LowSp_, 1.3))
# #
# #             # plasticity
# #             with tf.name_scope('plasticity'):
# #                 A = tf.matmul(p, tf.ones([1, N]))
# #                 dwLTD_ = A_LTD * (A + tf.transpose(A))
# #
# #                 # dwLTP_ = A_LTP * tf.mul(tf.to_float(vv_),(g0 - wGap))
# #
# #                 B = tf.matmul(vv_, tf.ones([1, N]))
# #                 g0 = 7
# #                 dwLTP_ = A_LTP * (tf.mul(tf.ones([N, N]) - 1 / g0 * wGap, B + tf.transpose(B)))
# #                 # dwLTD_ = A_LTD * p
# #                 dwGap_ = dt * (dwLTP_ - dwLTD_) * tf.cast((sim_index>self.startPlast), tf.float32)
# #                 wGap_ = tf.mul(tf.clip_by_value(wGap + dwGap_, clip_value_min=0, clip_value_max=10 ** 10), allowedConnections)
# #
# #             # monitoring
# #             with tf.name_scope('Monitoring'):
# #                 vmean_ = tf.reduce_mean(v_)
# #                 umean_ = tf.reduce_mean(u_)
# #                 imean_ = tf.reduce_mean(I_)
# #                 vvmean_ = tf.reduce_sum(tf.to_float(vv_))
# #                 vvmeanN1_ = tf.reduce_sum(tf.to_float(vvN1_))
# #                 vvmeanN2_ = tf.reduce_sum(tf.to_float(vvN2_))
# #                 gammamean_ = tf.reduce_mean(wGap_)
# #
# #             with tf.name_scope('Raster_Plot'):
# #                 spike_update = tf.scatter_update(self.spikes, sim_index, tf.reshape((vv_), (N,)))
# #
# #             # Operation to update the state
# #             step = tf.group(
# #                 v.assign(v_),
# #                 vv.assign(vv_),
# #                 u.assign(u_),
# #                 iBack.assign(iBack_),
# #                 iEff.assign(iEff_),
# #                 LowSp.assign(LowSp_),
# #                 wGap.assign(wGap_),
# #                 vmean.assign(vmean_),
# #                 umean.assign(umean_),
# #                 imean.assign(imean_),
# #                 vvmean.assign(vvmean_),
# #                 vvmeanN1.assign(vvmeanN1_),
# #                 vvmeanN2.assign(vvmeanN2_),
# #             )
# #
# #         plast = tf.group(
# #                         wGap.assign(wGap_),
# #                     )
# #
# #         # with tf.name_scope('Summaries'):
# #         #     variable_summaries(v_, 'Monitoring/v')
# #         #     variable_summaries(u_, 'u')
# #         #     variable_summaries(I_, 'I')
# #         #     variable_summaries(vv_, 'vv')
# #         #     variable_summaries(wGap_, 'gamma')
# #         #     w_hist = tf.histogram_summary("weights", wGap_)
# #
# #
# #         ## Simulation
# #         # Initialize state to initial conditions
# #         # merged = tf.merge_all_summaries()
# #         # train_writer = tf.train.SummaryWriter(FLAGS.summaries_dir + '/train', sess.graph)
# #
# #         # sess = tf.Session()
# #         self.sess.run(tf.initialize_all_variables())
# #         # with sess.as_default():
# #         if 1:
# #             self.vm = []
# #             self.um = []
# #             self.vvm = []
# #             self.vvmN1 = []
# #             self.vvmN2 = []
# #             self.im = []
# #             self.gamma = []
# #             self.gammaN1 = []
# #             self.gammaN2 = []
# #             self.iEff = []
# #             t0 = time.time()
# #             bar = pyprind.ProgBar(T)
# #             for i in range(T):
# #                 # Step simulation
# #
# #                 if self.showProgress:
# #                     bar.update()
# #
# #                 if i > self.startPlast:
# #                     self.sess.run([step, plast], feed_dict={dt: 0.25, tauv: self.tauv, sim_index: i})
# #                 else:
# #                     self.sess.run([step], feed_dict={dt: 0.25, tauv: self.tauv, sim_index: i})
# #
# #                 if self.spikeMonitor:
# #                     feed = {dt: 0.25, tauv: self.tauv, sim_index: i}
# #                     self.sess.run(spike_update, feed_dict=feed)
# #                 self.vvm.append(vvmean.eval())
# #                 self.vvmN1.append(vvmeanN1.eval())
# #                 self.vvmN2.append(vvmeanN2.eval())
# #
# #                 # Visualize every x steps
# #                 if i % 1 == 0:
# #                     pass
# #                     #             summary = sess.run(merged, feed_dict={dt: dtVal, tauv: 15})
# #                     #             train_writer.add_summary(summary, i)
# #                     weights = wGap.eval()
# #                     if self.disp and i % 40 == 0:
# #                         clear_output(wait=True)
# #                         self.DisplayArray(weights, rng=[0, 1.5 * gmean_init], text="%.2f ms" % (i*0.25))
# #                     self.vm.append(vmean.eval())
# #                     self.um.append(umean.eval())
# #                     self.im.append(imean.eval())
# #                     self.iEff.append(iEff.eval())
# #
# #                     self.gammaN1.append(np.mean(weights[:nbInCluster-sG, :nbInCluster-sG]) * nbOfGaps ** 0.5)
# #                     self.gammaN2.append(np.mean(weights[nbInCluster+sG:, nbInCluster+sG:]) * nbOfGaps ** 0.5)
# #                     self.gamma.append(gammamean.eval())
# #             self.raster = self.spikes.eval()
# #         # if i % 100 == 0:
# #         #             run_options = tf.RunOptions(trace_level=tf.RunOptions.FULL_TRACE)
# #         #             run_metadata = tf.RunMetadata()
# #         #             train_writer.add_run_metadata(run_metadata, 'step%03d' % i)
# #
# #         print('%.2f' % (time.time() - t0))
# #         self.sess.close()
#
# '''
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
# '''
#
# class TfSingleNet:
#     # DEVICE = '/gpu:0'
#
#
#     def __init__(self, N=400,
#                  T=400,
#                  disp=False,
#                  spikeMonitor=False,
#                  input=None,
#                  tauv=15,
#                  device='/gpu:0',
#                  NUM_CORES=1,
#                  g0=7,
#                  nu=100,
#                  startPlast=500,
#                  memfraction=1):
#         tf.reset_default_graph()
#         self.N = N
#         self.T = T
#         self.disp = disp
#         self.spikeMonitor = spikeMonitor
#         self.tauv = tauv
#         self.g = g0
#         self.g0fromFile = False
#         self.device = device
#         self.startPlast = startPlast
#         self.raster = []
#         self.nu = nu
#         gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=memfraction)
#
#         self.sess = tf.InteractiveSession(config=tf.ConfigProto(
#             inter_op_parallelism_threads=NUM_CORES,
#             intra_op_parallelism_threads=NUM_CORES,
#             gpu_options=gpu_options,
#             device_count={'GPU': (device[:4] == '/gpu') * 1})
#         )
#         if input is None:
#             self.input = np.ones((T, 1), dtype='int32')
#
#     def to_JSON(self):
#         return json.dumps(self, default=lambda o: o.__dict__,
#                           sort_keys=True, indent=4)
#
#     def DisplayArray(self, a, fmt='jpeg', rng=[0, 1], text=""):
#         """Display an array as a picture."""
#         a = (a - rng[0]) / float(rng[1] - rng[0]) * 255
#         a = np.uint8(np.clip(a, 0, 255))
#         f = BytesIO()
#         PIL.Image.fromarray(np.uint8(plt.cm.YlGnBu_r(a) * 255)).save(f, fmt)
#         display(Image(data=f.getvalue()))
#         print(text)
#
#     def init_float(self, shape, name):
#         #     return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)
#         return tf.Variable(tf.zeros(shape), name=name)
#
#     def variable_summaries(self, var, name):
#         """Attach a lot of summaries to a Tensor."""
#         with tf.name_scope('summaries'):
#             mean = tf.reduce_mean(var)
#             tf.scalar_summary('mean/' + name, mean)
#             with tf.name_scope('stddev'):
#                 stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
#             tf.scalar_summary('sttdev/' + name, stddev)
#             tf.scalar_summary('max/' + name, tf.reduce_max(var))
#             tf.scalar_summary('min/' + name, tf.reduce_min(var))
#             tf.histogram_summary(name, var)
#         return 0
#
#     def runTFSimul(self):
#         #################################################################################
#         ### INITIALISATION
#         #################################################################################
#         N = self.N
#         T = self.T
#         with tf.device(self.device):
#             dt = tf.placeholder(tf.float32, shape=(), name='dt')
#             tauv = tf.placeholder(tf.float32, shape=(), name='tauv')
#             sim_index = tf.placeholder(tf.int32, shape=(), name='sim_index')
#
#             scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70
#
#             # Create variables for simulation state
#             u = self.init_float([N, 1], 'u')
#             v = self.init_float([N, 1], 'v')
#             p = self.init_float([N, 1], 'v')
#
#             LowSp = self.init_float([N, 1], 'bursting')
#             vv = self.init_float([N, 1], 'spiking')
#
#             vmean = tf.Variable(0, dtype='float32')
#             umean = tf.Variable(0, dtype='float32')
#             vvmean = tf.Variable(0, dtype='float32')
#             pmean = tf.Variable(0, dtype='float32')
#             imean = tf.Variable(0, dtype='float32')
#             gammamean = tf.Variable(0, dtype='float32')
#             iEffm = tf.Variable(0, dtype='float32')
#
#             # currents
#             iBack = self.init_float([N, 1], 'iBack')
#             iEff = self.init_float([N, 1], 'iEff')
#             iChem = self.init_float([N, 1], 'iChem')
#
#             # synaptics connection
#             conn = np.ones([N, N], dtype='float32') - np.diag(np.ones(N, dtype='float32'))
#             nbOfGaps = np.sum(conn)
#
#             input = tf.cast(tf.constant(self.input), tf.float32)
#
#             if self.g0fromFile:
#                 self.g = getGSteady(self.tauv, 5, 1000)
#             g0 = self.g / (nbOfGaps**0.5)
#             wGap_init = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0
#             wII_init = np.ones([N, N], dtype=np.float32) * 700 / (nbOfGaps**0.5) / 0.25
#
#             wGap = tf.Variable(wGap_init)
#             WII = tf.Variable(wII_init)
#
#             FACT = 10
#             ratio = 15
#             A_LTD = 1e-0 * 4.7e-6 * FACT * 400 / N
#             A_LTP = ratio * A_LTD
#
#             # stimulation
#             TImean = self.nu
#
#             self.spikes = self.init_float([T, N], "spikes")
#
#             net = tf.ones([N, 1])
#             tauvSubnet = net * self.tauv
#
#         #################################################################################
#         ## Computation
#         #################################################################################
#         with tf.device(self.device):
#             with tf.name_scope('Currents'):
#                 # Discretized PDE update rules
#                 iChem_ = iChem + dt / 5 * (-iChem + tf.matmul(WII, tf.to_float(vv)))
#
#                 # current
#                 iBack_ = iBack + dt / 5 * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
#                                                                      seed=None, name=None))
#                 input_ = tf.gather(input, sim_index)
#
#                 # input to network: colored noise + external input
#                 iEff_ = iBack_ * scaling + input_ + TImean
#
#                 iGap_ = tf.matmul(wGap, v) - tf.mul(tf.reshape(tf.reduce_sum(wGap, 0), (N, 1)), v)
#
#                 I_ = iGap_ + iChem_ + iEff_
#
#             # IZHIKEVICH
#             with tf.name_scope('Izhikevich'):
#                 # voltage
#                 v_ = v + dt / tauvSubnet * (tf.mul((v + 60), (v + 50)) - 20 * u + 8 * I_)
#                 # adaptation
#                 u_ = u + dt * 0.044 * (v_ + 55 - u)
#                 # spikes
#                 vv_ = tf.to_float(tf.greater(v_, 25.0))
#                 # reset
#                 v_ = tf.mul(vv_, -40.0) + tf.mul((1 - vv_), v_)
#                 u_ = u_ + tf.mul(vv_, (50.0))
#
#             # bursting
#             with tf.name_scope('bursting'):
#                 LowSp_ = LowSp + dt / 10.0 * (vv_ * 10.0 / dt - LowSp)
#                 p_ = tf.to_float(tf.greater(LowSp_, 1.3))
#
#             # plasticity
#             with tf.name_scope('plasticity'):
#                 A = tf.matmul(p_, tf.ones([1, N]))
#                 dwLTD_ = A_LTD * (A + tf.transpose(A))
#                 B = tf.matmul(vv_, tf.ones([1, N]))
#                 # change g0 by 7
#                 dwLTP_ = A_LTP * (tf.mul(tf.ones([N, N]) - 1 / (7/ (nbOfGaps**0.5)) * wGap, B + tf.transpose(B)))
#                 dwGap_ = dt * (dwLTP_ - dwLTD_) * tf.cast((sim_index > self.startPlast), tf.float32)
#                 wGap_ = tf.clip_by_value(wGap + dwGap_, clip_value_min=0, clip_value_max=10 ** 10)
#
#             # monitoring
#             with tf.name_scope('Monitoring'):
#                 vmean_ = tf.reduce_mean(v_)
#                 umean_ = tf.reduce_mean(u_)
#                 imean_ = tf.reduce_mean(I_)
#                 vvmean_ = tf.reduce_mean(tf.to_float(vv_))
#                 pmean_ = tf.reduce_mean(p_)
#                 gammamean_ = tf.reduce_mean(wGap_)
#
#
#             with tf.name_scope('Raster_Plot'):
#                 spike_update = tf.scatter_update(self.spikes, sim_index, tf.reshape((vv_), (N,)))
#
#             # Operation to update the state
#             step = tf.group(
#                 v.assign(v_),
#                 u.assign(u_),
#                 vv.assign(vv_),
#
#                 iBack.assign(iBack_),
#                 iEff.assign(iEff_),
#                 LowSp.assign(LowSp_),
#
#                 vmean.assign(vmean_),
#                 umean.assign(umean_),
#                 imean.assign(imean_),
#                 vvmean.assign(vvmean_),
#                 pmean.assign(pmean_),
#                 # p.assign(p_),
#                 iEffm.assign(tf.reduce_mean(iEff_))
#             )
#
#             plast = tf.group(
#                 wGap.assign(wGap_),
#                 gammamean.assign(gammamean_),
#             )
#
#         self.sess.run(tf.initialize_all_variables())
#         if 1:
#             self.vm = []
#             self.um = []
#             self.vvm = []
#             self.bursts = []
#             self.im = []
#             self.gamma = []
#             self.iEff = []
#             self.lowsp = []
#             t0 = time.time()
#             for i in range(T):
#                 # Step simulation
#
#                 # start without plasticity until t>startPlast
#                 if i < self.startPlast:
#                     self.sess.run([step], feed_dict={dt: 0.25, tauv: self.tauv, sim_index: i})
#                 else:
#                     self.sess.run([step, plast], feed_dict={dt: 0.25, tauv: self.tauv, sim_index: i})
#
#                 if self.spikeMonitor:
#                     feed = {dt: 0.25, tauv: self.tauv, sim_index: i}
#                     self.sess.run(spike_update, feed_dict=feed)
#                 # Visualize every 50 steps
#                 if i % 1 == 0:
#                     if self.disp:
#                         clear_output(wait=True)
#                         self.DisplayArray(wGap.eval(), rng=[0, 1.5 * g0], text="%.2f ms" % (i * 0.25))
#                     # self.vm.append(vmean.eval())
#                     self.vvm.append(vvmean.eval())
#                     # self.um.append(umean.eval())
#                     self.im.append(imean.eval())
#                     # self.iEff.append(iEffm.eval())
#                     self.gamma.append(gammamean.eval() * (nbOfGaps**0.5))
#                     self.bursts.append(pmean.eval())
#                     self.lowsp.append(LowSp.eval())
#
#             self.raster = self.spikes.eval()
#             self.burstingActivity = np.mean(self.bursts)
#             self.spikingActivity = np.mean(self.vvm)
#
#
#         print('%.2f' % (time.time() - t0))
#         self.sess.close()
#
#
# '''
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
#
# '''
# #
# # class TfEvolveNet:
# #     # DEVICE = '/gpu:0'
# #
# #
# #     def __init__(self, N=400, T=400, disp=False, spikeMonitor=False, input=None, tauv=15,
# #                  sG=10, device='/gpu:0', both=False, NUM_CORES=1, g0=7, startPlast=500):
# #         tf.reset_default_graph()
# #         self.N = N
# #         self.T = T
# #         self.disp = disp
# #         self.spikeMonitor = spikeMonitor
# #         self.tauv = tauv
# #         self.sG = sG
# #         self.g = g0
# #         self.device = device
# #         self.both = both
# #         self.initWGap = True
# #         self.startPlast = startPlast
# #         self.raster = []
# #         self.showProgress = False
# #         self.sess = tf.InteractiveSession(config=tf.ConfigProto(
# #             inter_op_parallelism_threads=NUM_CORES,
# #             intra_op_parallelism_threads=NUM_CORES,
# #             device_count={'GPU': (device[:4]=='/gpu')*1})
# #         )
# #         if input is None:
# #             self.input = np.ones((T,1), dtype='int32')
# #
# #     def to_JSON(self):
# #         return json.dumps(self, default=lambda o: o.__dict__,
# #                           sort_keys=True, indent=4)
# #
# #     def DisplayArray(self, a, fmt='jpeg', rng=[0,1], text=""):
# #         """Display an array as a picture."""
# #         a = (a - rng[0])/float(rng[1] - rng[0])*255
# #         a = np.uint8(np.clip(a, 0, 255))
# #         f = BytesIO()
# #         PIL.Image.fromarray(np.uint8(plt.cm.YlGnBu_r(a)*255)).save(f, fmt)
# #         display(Image(data=f.getvalue()))
# #         print(text)
# #
# #     def init_float(self, shape, name):
# #     #     return tf.Variable(tf.random_normal(shape, stddev=0.01), name=name)
# #         return tf.Variable(tf.zeros(shape), name=name)
# #
# #
# #     def variable_summaries(self, var, name):
# #         """Attach a lot of summaries to a Tensor."""
# #         with tf.name_scope('summaries'):
# #             mean = tf.reduce_mean(var)
# #             tf.scalar_summary('mean/' + name, mean)
# #             with tf.name_scope('stddev'):
# #                 stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
# #             tf.scalar_summary('sttdev/' + name, stddev)
# #             tf.scalar_summary('max/' + name, tf.reduce_max(var))
# #             tf.scalar_summary('min/' + name, tf.reduce_min(var))
# #             tf.histogram_summary(name, var)
# #         return 0
# #
# #
# #     def runTFSimul(self):
# #         #################################################################################
# #         ### INITIALISATION
# #         #################################################################################
# #         N = self.N
# #         T = self.T
# #         with tf.device(self.device):
# #             dt = tf.placeholder(tf.float32, shape=(), name='dt')
# #             tauv = tf.placeholder(tf.float32, shape=(), name='tauv')
# #             sim_index = tf.placeholder(tf.int32, shape=(), name='sim_index')
# #
# #             scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70
# #
# #             # Create variables for simulation state
# #             u =self.init_float([N, 1], 'u')
# #             v =self.init_float([N, 1], 'v')
# #             t = tf.Variable(0, dtype='float32')
# #             ind = tf.Variable(0, dtype='float32')
# #
# #             LowSp =self.init_float([N, 1], 'bursting')
# #             vv =self.init_float([N, 1], 'spiking')
# #
# #             vmean = tf.Variable(0, dtype='float32')
# #             umean = tf.Variable(0, dtype='float32')
# #             vvmean = tf.Variable(0, dtype='float32')
# #             vvmeanN1 = tf.Variable(0, dtype='float32')
# #             vvmeanN2 = tf.Variable(0, dtype='float32')
# #             imean = tf.Variable(0, dtype='float32')
# #             gammamean = tf.Variable(0, dtype='float32')
# #             iEffm = tf.Variable(0, dtype='float32')
# #
# #
# #             # currents
# #             iBack =self.init_float([N, 1], 'iBack')
# #             iEff =self.init_float([N, 1], 'iEff')
# #             iGap =self.init_float([N, 1], 'iGap')
# #             iChem =self.init_float([N, 1], 'iChem')
# #             # synaptics connection
# #             conn = np.zeros([N, N], dtype='float32')
# #             conn1 = np.zeros([N, N], dtype='float32')
# #             conn2 = np.zeros([N, N], dtype='float32')
# #             connS = np.zeros([N, N], dtype='float32')
# #             sG = self.sG
# #             nbInCluster = N//2
# #             for i in range(N):
# #                 for j in range(N):
# #                     conn[i][j] = (((i < (nbInCluster + sG)) & (j < (nbInCluster + sG))) \
# #                                  or ((i >= (nbInCluster - sG)) & (j >= (nbInCluster - sG)))) and (i!=j)
# #             for i in range(N):
# #                 for j in range(N):
# #                     conn1[i][j] = ((i < nbInCluster-sG) and (j < nbInCluster-sG) and (i!=j))
# #
# #             for i in range(N):
# #                 for j in range(N):
# #                     conn2[i][j] = ((i > nbInCluster+sG) and (j > nbInCluster+sG) and (i!=j))
# #
# #             for i in range(N):
# #                 for j in range(N):
# #                     connS[i][j] = ((i >= (nbInCluster - sG) and i <= (nbInCluster + sG)) \
# #                             or (j >= (nbInCluster - sG) and j <= (nbInCluster + sG))) and (i!=j)
# #
# #             conn = conn1 + conn2 + connS
# #             conn = np.float32(conn != 0)
# #             self.conn = conn
# #             self.conn1 = conn1
# #             self.conn2 = conn2
# #             self.connS = connS
# #
# #             allowedConnections = tf.Variable(conn)
# #             nbOfGaps = np.sum(conn)
# #
# #             input = tf.cast(tf.constant(self.input), tf.float32)
# #
# #             if self.initWGap:
# #                 g_1 = getGSteady(15, 5, 1000)
# #                 g0_1 = g_1 / (np.sum(conn1)**0.5) *2
# #                 wGap_init_1 = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_1
# #                 wGap_init_1 *= conn1
# #
# #                 g_2 = getGSteady(self.tauv, 5, 1000)
# #                 g0_2 = g_2 / (np.sum(conn2)**0.5) *2
# #                 wGap_init_2 = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_2
# #                 wGap_init_2 *= conn2
# #
# #                 g_S = (g_1 + g_2) / 2
# #                 g0_S = g_S / (np.sum(conn1)**0.5) *2
# #                 wGap_init_S = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_S
# #                 wGap_init_S *= connS
# #
# #                 wGap_init = wGap_init_1 + wGap_init_2 + wGap_init_S
# #
# #             else:
# #                 g0 = self.g / (N/2)
# #                 wGap_init = (np.random.random_sample([N, N]).astype(np.float32)*(1-0.001)+0.001) * g0
# #                 wGap_init *= conn
# #
# #             gmean_init = np.mean(wGap_init)
# #             wII_init = np.ones([N, N], dtype=np.float32) * 500 / (np.sum(conn1)**0.5) / 0.25
# #
# #             wGap = tf.Variable(wGap_init)
# #             WII = tf.Variable(wII_init * (conn1 + conn2))
# #
# #             FACT = 10
# #             ratio = 15
# #             A_LTD = 1e-0 * 4.7e-6 * FACT * 400 / (np.sum(conn1)**0.5)
# #             A_LTP = ratio * A_LTD
# #
# #             # low constant noise
# #             TImean_init = tf.ones([N, 1]) * 30
# #
# #             # stimulation
# #             TImean = 100.0
# #             TImean_simul = tf.ones([N, 1], dtype='float32') * TImean
# #
# #             self.spikes = self.init_float([T, N], "spikes")
# #
# #             subnet = tf.concat(0, [tf.ones([N//2, 1]), tf.zeros([N-N//2,1])])
# #             subnetout = tf.concat(0, [tf.zeros([N//2, 1]), tf.ones([N-N//2,1])])
# #             tauvSubnet = subnet * 15 + subnetout * self.tauv
# #
# #         #################################################################################
# #         ## Computation
# #         #################################################################################
# #         with tf.device(self.device):
# #             with tf.name_scope('Currents'):
# #                 # Discretized PDE update rules
# #                 iChem_ = iChem + dt / 5 * (-iChem + tf.matmul(WII, tf.to_float(vv)))
# #
# #                 # current
# #                 iBack_ = iBack + dt / 5 * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
# #                                                                      seed=None, name=None))
# #                 input_ = tf.gather(input, sim_index)
# #
# #                 if self.both:
# #                     # input to both subnet
# #                     # iEff_ = iBack_ * scaling + tf.select(tf.greater(tf.ones([N, 1]) * input_, 0), TImean_simul, TImean_init)
# #                     iEff_ = iBack_ * scaling + input_ + TImean
# #                 else:
# #                     # input first subnet
# #                     iEff_ = iBack_ * scaling + tf.select(tf.greater(subnet * input_, 0), TImean_simul, TImean_init)
# #                     iEff_ = iBack_ * scaling + input_ * subnet + TImean
# #
# #
# #
# #                 iGap_ = tf.matmul(wGap, v) - tf.mul(tf.reshape(tf.reduce_sum(wGap, 0), (N, 1)), v)
# #
# #                 I_ = iGap_ + iChem_ + iEff_
# #
# #             # IZHIKEVICH
# #             with tf.name_scope('Izhikevich'):
# #                 ind_ = ind + 1
# #                 # voltage
# #                 v_ = v + dt / tauvSubnet * (tf.mul((v + 60), (v + 50)) - 20 * u + 8 * I_)
# #                 # adaptation
# #                 u_ = u + dt * 0.044 * (v_ + 55 - u)
# #                 # spikes
# #                 vv_ = tf.to_float(tf.greater(v_, 25.0))
# #                 vvN1_ = tf.to_float(tf.greater(v_*subnet, 25.0))
# #                 vvN2_ = tf.to_float(tf.greater(v_*subnetout, 25.0))
# #                 # reset
# #                 v_ = tf.mul(vv_, -40.0) + tf.mul((1 - vv_), v_)
# #                 u_ = u_ + tf.mul(vv_, (50.0))
# #
# #             # bursting
# #             with tf.name_scope('bursting'):
# #                 LowSp_ = LowSp + dt / 10.0 * (vv_ * 10.0 / dt - LowSp)
# #                 p = tf.to_float(tf.greater(LowSp_, 1.3))
# #
# #             # plasticity
# #             with tf.name_scope('plasticity'):
# #                 A = tf.matmul(p, tf.ones([1, N]))
# #                 dwLTD_ = A_LTD * (A + tf.transpose(A))
# #
# #                 B = tf.matmul(vv_, tf.ones([1, N]))
# #                 dwLTP_ = A_LTP * (tf.mul(tf.ones([N, N]) - (1 / (7/(N/2)) * wGap), B + tf.transpose(B)))
# #                 # dwLTD_ = A_LTD * p
# #                 dwGap_ = dt * (dwLTP_ - dwLTD_) * tf.cast((sim_index>self.startPlast), tf.float32)
# #                 wGap_ = tf.mul(tf.clip_by_value(wGap + dwGap_, clip_value_min=0, clip_value_max=10 ** 10), allowedConnections)
# #
# #             # monitoring
# #             with tf.name_scope('Monitoring'):
# #                 vvmean_ = tf.reduce_sum(tf.to_float(vv_))
# #                 vvmeanN1_ = tf.reduce_sum(tf.to_float(vvN1_))
# #                 vvmeanN2_ = tf.reduce_sum(tf.to_float(vvN2_))
# #             with tf.name_scope('Raster_Plot'):
# #                 spike_update = tf.scatter_update(self.spikes, sim_index, tf.reshape((vv_), (N,)))
# #
# #             # Operation to update the state
# #             step = tf.group(
# #                 v.assign(v_),
# #                 vv.assign(vv_),
# #                 u.assign(u_),
# #                 iBack.assign(iBack_),
# #                 iEff.assign(iEff_),
# #                 LowSp.assign(LowSp_),
# #                 wGap.assign(wGap_),
# #                 vvmean.assign(vvmean_),
# #                 vvmeanN1.assign(vvmeanN1_),
# #                 vvmeanN2.assign(vvmeanN2_),
# #                 iEffm.assign(tf.reduce_mean(iEff_)),
# #                 i1mean.assign(tf.reduce_mean(I_*subnet)),
# #                 i2mean.assign(tf.reduce_mean(I_*subnetout)),
# #             )
# #
# #         plast = tf.group(
# #                         wGap.assign(wGap_),
# #                     )
# #
# #         self.sess.run(tf.initialize_all_variables())
# #         if 1:
# #             self.vm = []
# #             self.um = []
# #             self.vvm = []
# #             self.vvmN1 = []
# #             self.vvmN2 = []
# #             self.im = []
# #             self.i1 = []
# #             self.i2 = []
# #             self.gamma = []
# #             self.gammaN1 = []
# #             self.gammaN2 = []
# #             self.iEff = []
# #             t0 = time.time()
# #             for i in range(T):
# #                 # Step simulation
# #
# #                 if i > self.startPlast:
# #                     self.sess.run([step, plast], feed_dict={dt: 0.25, tauv: self.tauv, sim_index: i})
# #                 else:
# #                     self.sess.run([step], feed_dict={dt: 0.25, tauv: self.tauv, sim_index: i})
# #
# #                 if self.spikeMonitor:
# #                     feed = {dt: 0.25, tauv: self.tauv, sim_index: i}
# #                     self.sess.run(spike_update, feed_dict=feed)
# #                 self.vvm.append(vvmean.eval())
# #                 self.vvmN1.append(vvmeanN1.eval())
# #                 self.vvmN2.append(vvmeanN2.eval())
# #                 self.iEff.append(iEffm.eval())
# #                 self.i1.append(i1mean.eval())
# #                 self.i2.append(i2mean.eval())
# #                 if i % 40 == 0:
# #                     weights = wGap.eval()
# #                     self.gammaN1.append(np.mean(weights[:nbInCluster - sG, :nbInCluster - sG])*N/2)
# #                     self.gammaN2.append(np.mean(weights[nbInCluster + sG:, nbInCluster + sG:])*N/2)
# #                     self.gamma.append(gammamean.eval())
# #
# #
# #         print('\n%.2f\n' % (time.time() - t0))
# #         self.sess.close()
#
#
# '''
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# ////////////////////////////////////////////////////////////////////////////////////////////////////////////////////////
# '''
#
# class TfConnEvolveNet:
#     # DEVICE = '/gpu:0'
#
#     def __init__(self, N=400, T=400, disp=False, spikeMonitor=False, input=None, tauv=15,
#                  sG=10, device='/gpu:0', both=False, NUM_CORES=1, g0=7, startPlast=500, nu=0):
#         tf.reset_default_graph()
#         self.N = N
#         self.T = T
#         self.debug = False
#         self.disp = disp
#         self.spikeMonitor = spikeMonitor
#         self.tauv = tauv
#         self.sG = sG
#         self.g = g0
#         self.device = device
#         self.both = both
#         self.initWGap = True
#         self.startPlast = startPlast
#         self.raster = []
#         self.showProgress = False
#         self.connectTime = 0
#         self.FACT = 10
#         self.nu = nu
#         self.sess = tf.InteractiveSession(config=tf.ConfigProto(
#             inter_op_parallelism_threads=NUM_CORES,
#             intra_op_parallelism_threads=NUM_CORES,
#             device_count={'GPU': (device[:4]=='/gpu')*1})
#         )
#         if input is None:
#             self.input = np.ones((T,1), dtype='int32')
#
#     def to_JSON(self):
#         return json.dumps(self, default=lambda o: o.__dict__,
#                           sort_keys=True, indent=4)
#
#     def DisplayArray(self, a, fmt='jpeg', rng=[0,1], text=""):
#         """Display an array as a picture."""
#         a = (a - rng[0])/float(rng[1] - rng[0])*255
#         a = np.uint8(np.clip(a, 0, 255))
#         f = BytesIO()
#         PIL.Image.fromarray(np.uint8(plt.cm.YlGnBu_r(a)*255)).save(f, fmt)
#         display(Image(data=f.getvalue()))
#         print(text)
#
#     def init_float(self, shape, name):
#         return tf.Variable(tf.zeros(shape), name=name)
#
#
#     def variable_summaries(self, var, name):
#         """Attach a lot of summaries to a Tensor."""
#         with tf.name_scope('summaries'):
#             mean = tf.reduce_mean(var)
#             tf.scalar_summary('mean/' + name, mean)
#             with tf.name_scope('stddev'):
#                 stddev = tf.sqrt(tf.reduce_sum(tf.square(var - mean)))
#             tf.scalar_summary('sttdev/' + name, stddev)
#             tf.scalar_summary('max/' + name, tf.reduce_max(var))
#             tf.scalar_summary('min/' + name, tf.reduce_min(var))
#             tf.histogram_summary(name, var)
#         return 0
#
#
#     def runTFSimul(self):
#         #################################################################################
#         ### INITIALISATION
#         #################################################################################
#         N = self.N
#         T = self.T
#         with tf.device(self.device):
#             dt = tf.placeholder(tf.float32, shape=(), name='dt')
#             tauv = tf.placeholder(tf.float32, shape=(), name='tauv')
#             sim_index = tf.placeholder(tf.int32, shape=(), name='sim_index')
#
#             scaling = 1 / (1 / (2 * 2 / 0.25)) ** 0.5 * 70
#
#             # Create variables for simulation state
#             u =self.init_float([N, 1], 'u')
#             v =self.init_float([N, 1], 'v')
#             ind = tf.Variable(0, dtype='float32')
#
#             LowSp =self.init_float([N, 1], 'bursting')
#             vv =self.init_float([N, 1], 'spiking')
#
#
#             # debut
#             wcontrol = tf.Variable(0, dtype='float32')
#             LTPcontrol = tf.Variable(0, dtype='float32')
#             LTDcontrol = tf.Variable(0, dtype='float32')
#             dwcontrol = tf.Variable(0, dtype='float32')
#
#             vvmean = tf.Variable(0, dtype='float32')
#             vvmeanN1 = tf.Variable(0, dtype='float32')
#             vvmeanN2 = tf.Variable(0, dtype='float32')
#             i1mean = tf.Variable(0, dtype='float32')
#             i2mean = tf.Variable(0, dtype='float32')
#             gammamean = tf.Variable(0, dtype='float32')
#             iEffm = tf.Variable(0, dtype='float32')
#
#
#             # currents
#             iBack =self.init_float([N, 1], 'iBack')
#             iEff =self.init_float([N, 1], 'iEff')
#             iGap =self.init_float([N, 1], 'iGap')
#             iChem =self.init_float([N, 1], 'iChem')
#             # synaptics connection
#             conn = np.zeros([N, N], dtype='float32')
#             conn1 = np.zeros([N, N], dtype='float32')
#             conn2 = np.zeros([N, N], dtype='float32')
#             connS = np.zeros([N, N], dtype='float32')
#             sG = self.sG
#             nbInCluster = N//2
#             for i in range(N):
#                 for j in range(N):
#                     conn[i][j] = (((i < (nbInCluster + sG)) & (j < (nbInCluster + sG))) \
#                                  or ((i >= (nbInCluster - sG)) & (j >= (nbInCluster - sG)))) and (i!=j)
#             for i in range(N):
#                 for j in range(N):
#                     conn1[i][j] = ((i < nbInCluster-sG) and (j < nbInCluster-sG) and (i!=j))
#                     # conn1 = np.float32(conn1 != 0)
#
#             for i in range(N):
#                 for j in range(N):
#                     conn2[i][j] = ((i > nbInCluster+sG) and (j > nbInCluster+sG) and (i!=j))
#                     # conn2 = np.float32(conn2 != 0)
#
#             for i in range(N):
#                 for j in range(N):
#                     connS[i][j] = ((i >= (nbInCluster - sG) and i <= (nbInCluster + sG)) \
#                             or (j >= (nbInCluster - sG) and j <= (nbInCluster + sG))) and (i!=j)
#                     # connS = np.float32(connS!=0)
#
#
#             conn = conn1 + conn2 + connS
#             conn =  np.float32(conn != 0)
#             conn0 = conn1 + conn2
#             conn0 = np.float32(conn0 != 0)
#             self.conn = conn
#             self.conn1 = conn1
#             self.conn2 = conn2
#             self.connS = connS
#
#             allowedConnections = tf.Variable(conn)
#             allowedConnections0 = tf.Variable(conn0)
#
#             input = tf.cast(tf.constant(self.input), tf.float32)
#
#             if self.initWGap==True:
#                 g_1 = getGSteady(15, 5, 1000)
#                 g0_1 = g_1 / (np.sum(conn1)**0.5) * 2
#                 wGap_init_1 = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_1
#                 wGap_init_1 *= conn1
#
#                 g_2 = getGSteady(self.tauv, 5, 1000)
#                 g0_2 = g_2 / (np.sum(conn2)**0.5) * 2
#                 wGap_init_2 = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_2
#                 wGap_init_2 *= conn2
#
#                 g0_S = (g0_1 + g0_2) / 2
#                 wGap_init_S = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_S
#                 wGap_init_S *= connS
#
#                 wGap_init = wGap_init_1 + wGap_init_2 + wGap_init_S
#                 wGap_init0 = wGap_init_1 + wGap_init_2
#
#             elif self.initWGap == -1:
#                 g_1 = self.g1
#                 g0_1 = g_1 / (np.sum(conn1) ** 0.5) * 2
#                 wGap_init_1 = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_1
#                 wGap_init_1 *= conn1
#
#                 g_2 = self.g2
#                 g0_2 = g_2 / (np.sum(conn1) ** 0.5) * 2
#                 wGap_init_2 = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_2
#                 wGap_init_2 *= conn2
#
#                 g0_S = (g0_1 + g0_2) / 2
#                 wGap_init_S = (np.random.random_sample([N, N]).astype(np.float32) * (1 - 0.001) + 0.001) * g0_S
#                 wGap_init_S *= connS
#
#                 wGap_init = wGap_init_1 + wGap_init_2 + wGap_init_S
#                 wGap_init0 = wGap_init_1 + wGap_init_2
#
#             else:
#                 g0 = self.g / (N/2)
#                 wGap_init = (np.random.random_sample([N, N]).astype(np.float32)*(1-0.001)+0.001) * g0
#                 wGap_init *= conn
#
#             gmean_init = np.mean(wGap_init)
#             wII_init = np.ones([N, N], dtype=np.float32) * 700 / (np.sum(conn1)**0.5) / 0.25
#
#             wGap = tf.Variable(wGap_init0)
#             wGap0 = tf.constant(wGap_init0)
#             wGapS = tf.constant(wGap_init_S)
#             WII = tf.Variable(wII_init * conn)
#             zer = tf.constant(np.zeros([N, N], dtype='float32'))
#
#
#             FACT = self.FACT
#             ratio = 15
#             A_LTD = 1e-0 * 4.7e-6 * FACT * 400 / (np.sum(conn1)**0.5)
#             A_LTP = ratio * A_LTD
#
#             # stimulation
#             # TImean = 100.0
#             TImean = self.nu
#
#             self.spikes = self.init_float([T, N], "spikes")
#
#             subnet = tf.concat(0, [tf.ones([N//2, 1]), tf.zeros([N-N//2,1])])
#             subnetout = tf.concat(0, [tf.zeros([N//2, 1]), tf.ones([N-N//2,1])])
#             tauvSubnet = subnet * 15 + subnetout * self.tauv
#
#             def fn1():
#                 return wGapS
#
#             def fn2():
#                 return zer
#
#             def fn3():
#                 return allowedConnections
#
#             def fn4():
#                 return allowedConnections0
#         #################################################################################
#         ## Computation
#         #################################################################################
#         with tf.device(self.device):
#             with tf.name_scope('Currents'):
#                 # Discretized PDE update rules
#                 wgap = tf.add(wGap, tf.cond(tf.equal(sim_index, self.connectTime), fn1, fn2))
#                 wgap = wgap / (1 + tf.cast(tf.equal(sim_index, self.connectTime), tf.float32))
#                 iGap_ = tf.matmul(wgap, v) - tf.mul(tf.reshape(tf.reduce_sum(wgap, 0), (N, 1)), v)
#                 iChem_ = iChem + dt / 5 * (-iChem + tf.matmul(WII, tf.to_float(vv)))
#
#                 # current
#                 iBack_ = iBack + dt / 5 * (-iBack + tf.random_normal((N, 1), mean=0.0, stddev=1.0, dtype=tf.float32,
#                                                                      seed=None, name=None))
#                 input_ = tf.gather(input, sim_index)
#
#                 if self.both:
#                     # input to both subnet
#                     iEff_ = iBack_ * scaling + input_ + TImean
#                 else:
#                     # input first subnet
#                     iEff_ = iBack_ * scaling + input_ * subnet + TImean
#
#                 # sum all currents
#                 I_ = iGap_ + iChem_ + iEff_
#
#             # IZHIKEVICH
#             with tf.name_scope('Izhikevich'):
#                 ind_ = ind + 1
#                 # voltage
#                 v_ = v + dt / tauvSubnet * (tf.mul((v + 60), (v + 50)) - 20 * u + 8 * I_)
#                 # adaptation
#                 u_ = u + dt * 0.044 * (v_ + 55 - u)
#                 # spikes
#                 vv_ = tf.to_float(tf.greater(v_, 25.0))
#                 vvN1_ = tf.to_float(tf.greater(v_*subnet, 25.0))
#                 vvN2_ = tf.to_float(tf.greater(v_*subnetout, 25.0))
#                 # reset
#                 v_ = tf.mul(vv_, -40.0) + tf.mul((1 - vv_), v_)
#                 u_ = u_ + tf.mul(vv_, (50.0))
#
#             # bursting
#             with tf.name_scope('bursting'):
#                 LowSp_ = LowSp + dt / 10.0 * (vv_ * 10.0 / dt - LowSp)
#                 p = tf.to_float(tf.greater(LowSp_, 1.3))
#
#             # plasticity
#             with tf.name_scope('plasticity'):
#                 # depression
#                 A = tf.matmul(p, tf.ones([1, N]))
#                 dwLTD_ = A_LTD * (A + tf.transpose(A))
#
#                 # potentiation
#                 B = tf.matmul(vv_, tf.ones([1, N]))
#                 dwLTP_ = A_LTP * (tf.mul(tf.ones([N, N]) - (1 / (7/(N/2)) * wGap), B + tf.transpose(B)))
#
#                 dwGap_ = dt * (dwLTP_ - dwLTD_) * tf.cast((sim_index > self.startPlast), tf.float32)
#                 wGap_ = tf.mul(tf.clip_by_value(wgap + dwGap_, clip_value_min=0, clip_value_max=10 ** 10),
#                                tf.cond((sim_index >= self.connectTime), fn3, fn4))
#
#             # debug
#             with tf.name_scope('debug'):
#                 LTDcontrol_ = tf.reduce_sum(dwLTD_)
#                 LTPcontrol_ =  tf.reduce_sum(dwLTP_)
#                 wcontrol_ = tf.reduce_sum(wgap)
#                 dwcontrol_ = tf.reduce_sum(dwGap_)
#
#
#             # monitoring
#             with tf.name_scope('Monitoring'):
#                 vvmean_ = tf.reduce_sum(tf.to_float(vv_))
#                 vvmeanN1_ = tf.reduce_sum(tf.to_float(vvN1_))
#                 vvmeanN2_ = tf.reduce_sum(tf.to_float(vvN2_))
#             with tf.name_scope('Raster_Plot'):
#                 spike_update = tf.scatter_update(self.spikes, sim_index, tf.reshape((vv_), (N,)))
#
#             # Operation to update the state
#             step = tf.group(
#                 v.assign(v_),
#                 vv.assign(vv_),
#                 u.assign(u_),
#                 iBack.assign(iBack_),
#                 iEff.assign(iEff_),
#                 LowSp.assign(LowSp_),
#                 wGap.assign(wGap_),
#                 vvmean.assign(vvmean_),
#                 vvmeanN1.assign(vvmeanN1_),
#                 vvmeanN2.assign(vvmeanN2_),
#                 iEffm.assign(tf.reduce_mean(iEff_)),
#                 i1mean.assign(tf.reduce_mean(I_ * subnet)),
#                 i2mean.assign(tf.reduce_mean(I_ * subnetout)),
#             )
#
#             debug = tf.group(
#                 wcontrol.assign(wcontrol_),
#                 LTPcontrol.assign(LTPcontrol_),
#                 LTDcontrol.assign(LTDcontrol_),
#                 dwcontrol.assign(dwcontrol_)
#             )
#
#             plast = tf.group(
#                             wGap.assign(wGap_),
#                         )
#
#
#         self.sess.run(tf.initialize_all_variables())
#         if 1:
#             self.vm = []
#             self.um = []
#             self.vvm = []
#             self.vvmN1 = []
#             self.vvmN2 = []
#             self.im = []
#             self.i1 = []
#             self.i2 = []
#             self.gamma = []
#             self.gammaN1 = []
#             self.gammaN2 = []
#             self.gammaNS = []
#             self.raster = []
#             self.iEff = []
#
#             t0 = time.time()
#             for i in range(T):
#                 # Step simulation
#                 if i > self.startPlast or i == self.connectTime:
#                     self.sess.run([step, plast], feed_dict={dt: 0.25, tauv: self.tauv, sim_index: i})
#                 else:
#                     self.sess.run([step], feed_dict={dt: 0.25, tauv: self.tauv, sim_index: i})
#
#                 if self.spikeMonitor:
#                     # feed = {dt: 0.25, tauv: self.tauv, sim_index: i}
#                     # self.sess.run(spike_update, feed_dict=feed)
#                     self.raster.append(vv.eval())
#
#                 if self.debug:
#                     self.sess.run([debug], feed_dict={dt: 0.25, tauv: self.tauv, sim_index: i})
#
#                     if i ==0:
#                         self.wcontrol = []
#                         self.dwcontrol = []
#                         self.LTDcontrol = []
#                         self.LTPcontrol = []
#
#                     self.wcontrol.append(wcontrol.eval())
#                     self.dwcontrol.append(dwcontrol.eval())
#                     self.LTDcontrol.append(LTDcontrol.eval())
#                     self.LTPcontrol.append(LTPcontrol.eval())
#
#
#                 self.vvm.append(vvmean.eval())
#                 self.vvmN1.append(vvmeanN1.eval())
#                 self.vvmN2.append(vvmeanN2.eval())
#                 self.iEff.append(iEffm.eval())
#                 self.i1.append(i1mean.eval())
#                 self.i2.append(i2mean.eval())
#                 if i % 40 == 0:
#                     weights = wGap.eval()
#                     self.gammaN1.append(np.mean(weights[:nbInCluster - sG, :nbInCluster - sG])*N)
#                     self.gammaN2.append(np.mean(weights[nbInCluster + sG:, nbInCluster + sG:])*N)
#                     self.gammaNS.append(np.mean(weights[nbInCluster - sG:nbInCluster + sG,
#                                                 nbInCluster - sG:nbInCluster + sG]) * N)
#
#                     self.gamma.append(gammamean.eval())
#
#                 if i==0:
#                     self.w0 = weights
#                 elif i==T-1:
#                     self.wE = weights
#                 # self.raster = self.spikes.eval()
#
#
#         print('\n%.2f\n' % (time.time() - t0))
#         self.sess.close()