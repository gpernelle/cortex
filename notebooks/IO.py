from functions import *



# c = ipp.Client(profile='cluster')
# lview = c.load_balanced_view()
# lview.block = True

def xax(gamma, duration):
    x= np.arange(0,(duration-1)/1000,(duration)/1000/len(gamma) )
    return x



class IO(object):
    def __init__(self):
        self.computer = ""
        self.executable_path = ""
        self.data_path = ""
        self.workstation_path = ""
        self.executable_name = ""

        self.N = 100
        self.r = 0
        self.i = 0
        self.g = 5
        self.S = 100
        self.d1 = 100
        self.d2 = 1000
        self.d3 = 100
        self.WII = 1400
        self.FACT = 1
        self.ratio = 15
        self.LTD = 1e-0 * 4.7e-6 * self.FACT * self.N
        self.LTP = self.ratio * self.LTD
        self.tauv = 15  # in ms, for 40Hz resonance

        self.glob = 0  # 0 for local GJ, 1 otherwise
        self.plast = 1
        self.sG = 0  # number of shared GJ
        self.sWII = 0  # number of shared chemical synapses
        self.model = "gp-izh-subnetworks"  # neuron model

        self.TImean = 30
        self.sigma = 60  # colored noise mean

        self.with_currents = False  # set to True to output LFP as current mean

    def initTime(self):
        self.before = self.d1
        self.after = self.d2 + self.d3
        self.T = self.d1 + self.d2 + self.d3

    def getHist(self, data, binsize):
        binnb = 0
        val = []
        for ind, i in enumerate(data):
            if (ind % binsize) == 0:
                if binnb > 0:
                    val.append(accumulator)
                accumulator = 0
                binnb += 1
            else:
                accumulator += i
        return val

    # def check_connection(self):
    #     print('Number of workers: %d' % len(c.ids))

    def runPSTH(self, it=8):
        Parallel(n_jobs=num_cores)(delayed(self.runSimulation)(i) for i in range(it))

    def readPSTH(self, it,
                 binsize, coeff, RON='izh', tau_m=10):
        gr = GRAPH()
        listSSP1 = Parallel(n_jobs=num_cores)(delayed(self.readSimulationSSP1)(i) for i in range(it))
        if RON == 'izh':
            listS = Parallel(n_jobs=num_cores)(delayed(gr.readoutSpikes)(ssp1, coeff, tau_m) for ssp1 in listSSP1)
        else:
            listS = Parallel(n_jobs=num_cores)(delayed(gr.readoutSpikesIAF)(ssp1, coeff, tau_m) for ssp1 in listSSP1)

        tot = np.sum(listS, axis=0)
        totSSP1 = np.sum(listSSP1, axis=0)
        total = np.sum(listS)
        h = self.getHist(tot, binsize)
        hSSP1 = self.getHist(totSSP1, binsize)
        spikes_x, spikes_y, spikes_x_tc, spikes_y_tc, gamma, correlation, ssp1, stimulation, p, q, lowsp, vm = self.readSimulation(
            0)

        return h, stimulation, total, hSSP1, totSSP1

    def runSimulation(self, i=0, tauv=None):
        ext = "_%d.txt" % i
        sh.cd(self.executable_path)
        if tauv is None:
            tauv = self.tauv

        commandStr = [self.executable_name, '-N', str(self.N), '-ext', str(ext),
                      '-d1', str(self.d1), '-d2', str(self.d2), '-d3', str(self.d3),
                      '-before', str(self.before), '-after', str(self.after),
                      '-S', str(self.S), '-G', str(self.g), '-s', str(self.sigma),
                      '-WII', str(self.WII), '-LTP', str(self.LTP), '-LTD', str(self.LTD),
                      '-model', self.model, '-r', str(self.r), '-global', str(self.glob),
                      '-sG', str(self.sG), '-sWII', str(self.sWII), '-tauv', str(tauv), '-plast', str(self.plast)]
        print(' '.join(commandStr))
        subprocess.check_output(commandStr)

    def readSimulationSSP1(self, i=0):

        DIRECTORY = self.data_path
        ext = "_%d.txt" % i
        # compute the paths of data files.
        extension = "_g-%.6g_TImean-%d_T-%d_Glob-%d_dt-0.25_N-%d_r-%.2g_S-%d_WII-%d_LTD-%.6g_LTP-%.6g_model-%s_sG-%d_sWII-%d_tauv-%d" \
                    % (self.g, self.TImean, self.T, self.glob, self.N, self.r, self.S, self.WII, self.LTD, self.LTP,
                       self.model, self.sG, self.sWII, self.tauv)
        extension += ext
        ssp1_p = DIRECTORY + "sspE" + extension
        ssp1 = np.fromfile(ssp1_p, dtype='double', count=-1, sep=" ")
        return ssp1

    def readMatrix(self, i=0, type="GAP", extension=None, workstation=False):

        if not workstation:
            DIRECTORY = self.data_path
        else:
            DIRECTORY = self.workstation_path
        # simulation parameters
        ext = "_%d.txt" % i
        # compute the paths of data files.
        if extension is None:
            extension = "_g-%.6g_TImean-%d_T-%d_Glob-%d_dt-0.25_N-%d_r-%.2g_S-%d_WII-%d_LTD-%.6g_LTP-%.6g_model-%s_sG-%d_sWII-%d_tauv-%d" \
                        % (self.g, self.TImean, self.T, self.glob, self.N, self.r, self.S, self.WII, self.LTD, self.LTP,
                           self.model, self.sG, self.sWII, self.tauv)
            extension += ext
        path_GAP = DIRECTORY + type + extension

        with open(path_GAP) as file:
            array2d = [[float(digit) for digit in line.split()] for line in file]

        return array2d

    def readToCSV(self, i=0, filename="data.csv", tauv=0, sWII=0, LTD=0, LTP=0, sG=0, save=True, workstation=False):
        print(tauv, sWII, sG)
        T = self.d1 + self.d2 + self.d3

        if not workstation:
            DIRECTORY = self.data_path
        else:
            DIRECTORY = self.workstation_path
        # simulation parameters
        TImean = 30
        ext = "_%d.txt" % i
        # compute the paths of data files.
        extension = "_g-%.6g_TImean-%d_T-%d_Glob-%d_dt-%.2g_N-%d_r-%.2g_S-%d_WII-%d_LTD-%.6g_LTP-%.6g_model-%s_sG-%d_sWII-%d_tauv-%d" \
                    % (self.g, TImean, T, self.glob, self.dt, self.N, self.r, self.S, self.WII, LTD, LTP,
                       self.model,
                       sG, sWII, tauv)
        extension += ext


        # path_g = DIRECTORY + "gamma" + extension
        path_gN1 = DIRECTORY + "gammaN1" + extension
        path_gN2 = DIRECTORY + "gammaN2" + extension
        path_gNShared = DIRECTORY + "gammaNshared" + extension

        i1_p = DIRECTORY + "current1" + extension
        i2_p = DIRECTORY + "current2" + extension

        try:
            gammaN1 = np.fromfile(path_gN1, dtype='double', count=-1, sep=" ")
        except:
            print('can\'t find:\t ' + path_gN1)
        try:
            gammaN2 = np.fromfile(path_gN2, dtype='double', count=-1, sep=" ")
        except:
            print('can\'t find:\t ' + path_gN2)
        try:
            gammaNShared = np.fromfile(path_gNShared, dtype='double', count=-1, sep=" ")
        except:
            print('can\'t find:\t ' + path_gNShared)
        try:
            i1 = np.fromfile(i1_p, dtype='double', count=-1, sep=" ")
        except:
            print('can\'t find:\t ' + i1_p)
        try:
            i2 = np.fromfile(i2_p, dtype='double', count=-1, sep=" ")
        except:
            print('can\'t find:\t ' + i2_p)

        start = 0
        end = 4000
        f, Pxy = signal.csd(i1[start:end], i2[start:end], fs=1 / 0.00025, nperseg=1024)
        f2, Pxy2 = signal.csd(i1[-end:], i2[-end:-1], fs=1 / 0.00025, nperseg=1024)

        maxBegin = np.max(np.abs(Pxy))
        argmaxBegin = np.argmax(np.abs(Pxy))
        maxEnd = np.max(np.abs(Pxy2))
        argmaxEnd = np.argmax(np.abs(Pxy2))

        csd = {'maxBegin': maxBegin, 'argmaxBegin': argmaxBegin, 'maxEnd': maxEnd, 'argmaxEnd': argmaxEnd}

        csd['f1Begin'] = fourier(i1[start:end])
        csd['f2Begin'] = fourier(i2[start:end])
        csd['f1End'] = fourier(i1[-end:-1])
        csd['f2End'] = fourier(i2[-end:-1])
        csd['fBothBegin'] = fourier(np.array(i1[start:end]) + np.array(i2[start:end]))
        csd['fBothEnd'] = fourier(np.array(i1[-end:-1]) + np.array(i2[-end:-1]))

        if self.tauv <= 35 and sG <= 10:
            key = -2
        elif self.tauv > 35 and sG <= 10:
            key = -1
        elif self.tauv <= 35 and sG > 10:
            key = 2
        elif self.tauv > 35 and sG > 10:
            key = 1
        else:
            key = 0

        mat = np.array(self.readMatrix(extension=extension, workstation=workstation))
        matN1 = mat[0:(self.N - sG) // 2 - 1, 0:(self.N - sG) // 2 - 1]
        matN2 = mat[-(self.N - sG) // 2 + 1 :, -(self.N - sG) // 2 + 1:]
        matNshared = mat[(self.N - sG) // 2 +1:(self.N + sG) // 2-1, (self.N - sG) // 2+1:(self.N + sG) // 2-1]
        gn1 = np.mean(matN1[np.nonzero(matN1)])
        gn2 = np.mean(matN2[np.nonzero(matN2)])
        gnshared = np.mean(matNshared[np.nonzero(matNshared)])

        mat = np.array(self.readMatrix(type="GAP0", extension=extension, workstation=workstation))
        matN1 = mat[0:(self.N - sG) // 2 - 1, 0:(self.N - sG) // 2 - 1]
        matN2 = mat[-(self.N - sG) // 2 + 1:, -(self.N - sG) // 2 + 1:]
        matNshared = mat[(self.N - sG) // 2 + 1:(self.N + sG) // 2 - 1, (self.N - sG) // 2 + 1:(self.N + sG) // 2 - 1]
        _gn1 = np.mean(matN1[np.nonzero(matN1)])
        _gn2 = np.mean(matN2[np.nonzero(matN2)])
        _gnshared = np.mean(matNshared[np.nonzero(matNshared)])
        _initmean = np.mean(mat[np.nonzero(mat)])

        data = [tauv, self.d2, sWII, sG, LTD > 1e-8,
                     csd['maxBegin'], csd['argmaxBegin'], csd['maxEnd'], csd['argmaxEnd'],
                     csd['f1Begin'][0], csd['f1Begin'][1], csd['f2Begin'][0], csd['f2Begin'][1],
                     csd['f1End'][0], csd['f1End'][1], csd['f2End'][0], csd['f2End'][1],
                     csd['fBothBegin'][0], csd['fBothBegin'][1], csd['fBothEnd'][0], csd['fBothEnd'][1],
                    _gn1, _gn2, _gnshared, _initmean,
                    _gn1/_initmean, _gn2/_initmean, _gnshared/_initmean,
                    gn1, gn2, gnshared,
                    gn1 / _initmean, gn2 / _initmean, gnshared / _initmean,
                     key]

        # append data file
        if save:
            f_handle = open(filename, 'ab')
            try:
                np.savetxt(f_handle, np.array(data).reshape(1,len(data)), delimiter=",", fmt='%.5g')
            except:
                pass
            f_handle.close()

            return 0
        else:
            return data

    def readSimulation(self, i=0, workstation=False):

        T = self.d1 + self.d2 + self.d3
        if not workstation:
            DIRECTORY = self.data_path
        else:
            DIRECTORY = self.workstation_path
        # simulation parameters
        TImean = 30
        ext = "_%d.txt" % i
        # compute the paths of data files.
        extension = "_g-%.6g_TImean-%d_T-%d_Glob-%d_dt-%.2g_N-%d_r-%.2g_S-%d_WII-%d_LTD-%.6g_LTP-%.6g_model-%s_sG-%d_sWII-%d_tauv-%d" \
                    % (self.g, TImean, T, self.glob, self.dt, self.N, self.r, self.S, self.WII, self.LTD, self.LTP, self.model,
                       self.sG, self.sWII, self.tauv)
        extension += ext

        path_x = DIRECTORY + "spike_x" + extension
        path_x_tc = DIRECTORY + "spike_x_tc" + extension
        path_y = DIRECTORY + "spike_y" + extension
        path_y_tc = DIRECTORY + "spike_y_tc" + extension
        path_g = DIRECTORY + "gamma" + extension
        path_gN1 = DIRECTORY + "gammaN1" + extension
        path_gN2 = DIRECTORY + "gammaN2" + extension
        path_c = DIRECTORY + "correlation" + extension
        ssp1_p = DIRECTORY + "sspE" + extension
        p_p = DIRECTORY + "p" + extension
        q_p = DIRECTORY + "q" + extension
        lowsp_p = DIRECTORY + "LowSp" + extension
        vm_p = DIRECTORY + "vm" + extension
        i1_p = DIRECTORY + "current1" + extension
        i2_p = DIRECTORY + "current2" + extension
        v_p = DIRECTORY + "v" + extension
        u_p = DIRECTORY + "u" + extension

        # RON_I_p = DIRECTORY +"RON_I"+ extension
        # RON_V_p = DIRECTORY +"RON_V"+ extension
        # V_p = DIRECTORY +"V"+ extension

        stimulation_p = DIRECTORY + "stimulation" + extension
        #     print path_g

        try:
            self.spikes_x = np.fromfile(path_x, dtype='uint', count=-1, sep=" ")
            # self.spikes_x_tc = np.fromfile(path_x_tc, dtype='uint', count=-1, sep=" ")
            self.spikes_y = np.fromfile(path_y, dtype='uint', count=-1, sep=" ")
            # self.spikes_y_tc = np.fromfile(path_y_tc, dtype='uint', count=-1, sep=" ")
            self.gamma = np.fromfile(path_g, dtype='double', count=-1, sep=" ")
            self.correlation = np.fromfile(path_c, dtype='double', count=-1, sep=" ")
            self.ssp1 = np.fromfile(ssp1_p, dtype='double', count=-1, sep=" ")
            self.p = np.fromfile(p_p, dtype='double', count=-1, sep=" ")
            self.q = np.fromfile(q_p, dtype='double', count=-1, sep=" ")
            self.LowSp = np.fromfile(lowsp_p, dtype='double', count=-1, sep=" ")
            self.vm = np.fromfile(vm_p, dtype='double', count=-1, sep=" ")
            if self.with_currents:
                self.i1 = np.fromfile(i1_p, dtype='double', count=-1, sep=" ")
                self.i2 = np.fromfile(i2_p, dtype='double', count=-1, sep=" ")
            self.stimulation = np.fromfile(stimulation_p, dtype='double', count=-1, sep=" ")

        except:
            print('can\'t find:\t ' + path_g)

        try:
            self.gammaN1 = np.fromfile(path_gN1, dtype='double', count=-1, sep=" ")
            self.gammaN2 = np.fromfile(path_gN2, dtype='double', count=-1, sep=" ")
        except:
            pass

        # try:
        self.voltage = np.fromfile(v_p, dtype='double', count=-1, sep=" ")
        self.adaptation = np.fromfile(u_p, dtype='double', count=-1, sep=" ")
        # except:
        #     pass

        return 0


class Cortex(IO):
    def __init__(self, *args, **kwargs):
        super(Cortex, self).__init__(*args, **kwargs)

        if os.uname()[1]=='OSX':
            self.computer = "guillaume"
            self.executable_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/" % self.computer
            self.data_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/cortex/data/" % self.computer
            self.executable_name = './cortex'

        elif os.uname()[1] != "CNL-Brain1":
            self.computer = "GP1514"
            self.executable_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/" % self.computer
            self.data_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/cortex/data/" % self.computer
            self.executable_name = './cortex'
        else:
            self.computer = "gp1514"
            self.executable_path = "/mnt/DATA/gp1514/Projects/github/cortex/cortex/"
            self.data_path = "/mnt/DATA/gp1514/Projects/github/cortex/data/"
            self.executable_name = './cortex'
        self.workstation_path = '/Users/GP1514/cnl-brain1/Projects/github/cortex/data/'


class TRN(IO):
    def __init__(self):
        self.computer = "GP1514"
        self.executable_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/TRN/TRN/" % self.computer
        self.data_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/TRN/data/" % self.computer
        self.executable_name = './trn'


class GRAPH():
    def __init__(self, cortex):
        self.cortex = cortex
        self.figsize11 = (6,6)
        self.figsize21 = (12,5)
        self.figsize22 = (8,8)
        self.figsize12 = (5,8)
        self.axes12 = [0.85, 0.25, 0.02, 0.5]
        self.axes11 = [0.89, 0.135, 0.023, 0.70]
        self.axes21 = [0.81, 0.12, 0.010, 0.72]
        self.axes22 = [0.85, 0.15, 0.02, 0.7]
        self.ext = '_%d_%d.svg'%(self.cortex.sWII, self.cortex.N)
        ## to print figures
        # self.figsize11 = (8, 6)
        # self.figsize21 = (14, 6)
        # self.figsize22 = (14, 12)
        # self.figsize12 = (8, 12)

    def readoutneuroncortex(self, ax, ssp1, Rm, W=-200, x=None):
        dt = 0.25
        bTh = 1.3
        T = len(ssp1) / 4  # total time to simulate (msec)

        time = np.arange(0, T, dt)
        n = IzhNeuron("Burst Mode", a=0.02, b=0.25, c=-50, d=2, v0=-70)
        s1 = IzhSim(n, T=T, dt=dt)

        for i, t in enumerate(time):
            s1.stim[i] = W * ssp1[i]

        res = s1.integrate()

        ax.set_ylim([-150, 30])
        ax.spines['left'].set_position('zero')
        ax.spines['right'].set_color('none')
        # ax.spines['bottom'].set_position('zero')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        if x:
            ax.set_xlabel('Time (msec)')
        else:
            ax.set_xticks([])

        ax.plot(s1.t, res[0], label='membrane voltage [mV]')
        return ax

    def readoutSpikes(self, ssp1, W=-200):
        dt = 0.25
        T = len(ssp1) / 4  # total time to simulate (msec)

        time = np.arange(0, T, dt)
        # n = IzhNeuron("Burst Mode", a=0.02, b=0.25, c=-50, d=2, v0=-70)
        n = IzhNeuron("(A) tonic spiking", a=0.02, b=0.2, c=-65, d=6, v0=-70)

        s1 = IzhSim(n, T=T, dt=dt)

        for i, t in enumerate(time):
            s1.stim[i] = W * ssp1[i]

        res = s1.integrate()
        spikes = res[0] > 0

        return spikes

    def readoutSpikesIAF(self, ssp1, W, tau_m=10):
        ## setup parameters and state variables
        T = len(ssp1) / 4  # total time to simulate (msec)
        dt = 0.25  # simulation time step (msec)
        time = np.arange(0, T, dt)  # time array
        t_rest = 0  # initial refractory time

        ## LIF properties
        Vm = np.zeros(len(time))  # potential (V) trace over time
        Rm = W  # resistance (kOhm)
        Cm = 7  # capacitance (uF)
        tau_m = tau_m  # time constant (msec)
        tau_ref = 4  # refractory period (msec)
        Vth = 1  # spike threshold (V)
        V_spike = 0.5  # spike delta (V)

        ## Stimulus
        I = ssp1  # input current (A)

        ## iterate over each time step
        for i, t in enumerate(time):
            if t > t_rest:
                Vm[i] = Vm[i - 1] + (-Vm[i - 1] + I[i] * Rm) / tau_m * dt
            if Vm[i] >= Vth:
                Vm[i] += V_spike
                t_rest = t + tau_ref
        # plt.plot(Vm)
        spikes = Vm > 1.1
        return spikes

    def findindex(self, val, spikes_x, start=0):
        result = spikes_x[-1]
        for i in range(start, len(spikes_x)):
            if spikes_x[i] == val or spikes_x[i] > val:
                result = i
                break
        return result

    def plotraster(self, ax, spikes_x, spikes_y, i1, i2, x=0):
        spx = spikes_x[i1:i2]
        spy = spikes_y[i1:i2]
        ax.spines['right'].set_color('none')
        ax.spines['top'].set_color('none')
        ax.xaxis.set_ticks_position('bottom')
        ax.yaxis.set_ticks_position('left')
        #     ax.set_ylim([0,100])
        if x == 0:
            ax.set_xticks([])
        else:
            ax.set_xlabel('Time (msec)')
        ax.plot(spx, spy, '.', markersize=2)



    def plotPTSH(self, fig, before, after, binsize, h, s, it, DIRECTORY, S, N):
        T = before + after
        simsize = T / 0.25
        x2 = np.arange(0, (simsize / 4 - 1) / 1000, (simsize / len(s) / 4) / 1000)

        # fig = plt.figure(figsize=(9,5))
        gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1])
        ax0 = plt.subplot(gs[0])
        # ax0.set_ylim([0,10])
        ax0.bar(np.arange(0, simsize / 1000, simsize / len(h) / 1000), np.array(h) / it, binsize / 1000)
        ax0.set_title('PSTH - Input: %s' % S)
        ax0.set_xticks([])

        # plot stimulation s
        ax = plt.subplot(gs[1])
        ax.set_ylim([30, 110])
        ax.set_yticks([30, 100])
        ax.set_xlabel('Time [s]')
        plt.plot(x2, s)
        plt.tight_layout()
        extension = "_S-%d_N-%d_T-%d" % (S, N, T)
        print(DIRECTORY + extension + '_PTSH.pdf')
        plt.savefig(DIRECTORY + extension + '_PTSH.pdf')

    def plotDiagram2(self, figure, ax, dataframe, title, column, filename, save=True, front=False, gridsize=60):
        ax.set_ylim(ymin=50, ymax=250)
        # Set color transparency (0: transparent; 1: solid)
        a = 1
        # Create a colormap
        customcmap = [(x / 24.0, x / 48.0, 0.05) for x in range(len(dataframe))]

        dataframe.plot(kind="hexbin",
                       y='nuEI', x='gammaC', C=column, gridsize=gridsize, ax=ax, alpha=a, legend=False, colormap=cx4,
                       # edgecolor='w',
                       title=title)
        ax.set_title(title, y=1.03)
        # Customize title, set position, allow space on top of plot for title
        # ax.set_title(ax.get_title(), fontsize=36, alpha=a)
        # ax.title.set_fontsize(36)
        plt.subplots_adjust(top=0.9)
        # ax.title.set_position((0,1.08))
        # Set x axis label on top of plot, set label text
        ax.xaxis.set_label_position('bottom')
        ax.set_xlim([0, 7])
        xlab = r'$\gamma_C$'
        ylab = r'$\nu_{EI}$'
        ax.set_xlabel(xlab, alpha=a)
        ax.set_ylabel(ylab, alpha=a)
        ax.set_xticklabels(ax.get_xticks(), alpha=a)
        ax.set_yticklabels(ax.get_yticks(), alpha=a)

        # plasticity trajectory
        # x = spikingSimulation.gamma*spikingSimulation.N_I
        # y = (spikingSimulation.stimulation+neuronI.N_mean)/(neuronI.V_th - neuronI.vReset)
        # plt.plot(x,y)
        if front:
            dataframe['logburst'] = (dataframe['burst']).apply(np.log10)
            df_sliced = dataframe[(dataframe['logburst'] > -6) & (dataframe['logburst'] < -5.8)]
            contour = df_sliced[['gammaC', 'nuEI']].get_values()
            yvals, xvals = self.bezier_curve(contour, nTimes=100)
            frontier = np.array([xvals, yvals])
            ax.plot(frontier[1, :], frontier[0, :], '-w', linewidth=3)
        if save:
            plt.tight_layout()
            plt.savefig(DIRECTORY + filename)
        return ax

    def plotDiagram(self, figure, ax, dataframe, title, column, filename, save=True, front=False, gridsize=60,
                    extent=[0, 7.0, 0, 200], cmap=cx4, bad=False, format=None):
        ax.set_ylim(ymin=extent[2], ymax=extent[3])
        # Set color transparency (0: transparent; 1: solid)
        a = 1
        # Create a colormap
        customcmap = [(x / 24.0, x / 48.0, 0.05) for x in range(len(dataframe))]
        dataframe = dataframe.fillna(0)
        nbVal = len(pd.unique(dataframe.nuEI.ravel()))
        da = np.array(dataframe[['gammaC', 'nuEI', column]].sort_values(['nuEI', 'gammaC'], ascending=[0, 1]))
        z = da[:, 2]
        # print(z.shape)
        try:
            zr = z.reshape(nbVal, len(z) / nbVal)
            print(zr.shape)
            if bad:
                cmap.set_bad('white')
                plt.imshow(np.ma.masked_values(zr, 0), cmap=cmap)
            image = ax.imshow(zr, extent=extent, cmap=cmap, aspect=(extent[1] - extent[0]) / (
            extent[3] - extent[2]))  # , cmap =cx4)# drawing the function
            if format == None:
                plt.colorbar(image, format='%.2g')
            else:
                plt.colorbar(image, format=format)
            # ax.plot(frontier[1,:], frontier[0,:], '-w', linewidth=2)
            ax.set_title(title, y=1.03)
            # Customize title, set position, allow space on top of plot for title
            # ax.set_title(ax.get_title(), fontsize=36, alpha=a)
            # ax.title.set_fontsize(36)
            plt.subplots_adjust(top=0.9)
            # ax.title.set_position((0,1.08))
            # Set x axis label on top of plot, set label text
            ax.xaxis.set_label_position('bottom')

            maxX = math.floor(extent[1] * 10) / 10
            maxY = math.ceil(extent[3] / 10) * 10

            ax.set_xlim([extent[0], maxX])
            ax.set_yticks([extent[2], (maxY - extent[2]) / 2, maxY])
            ax.set_xticks([extent[0], (maxX - extent[0]) / 2, maxX])
            xlab = r'Gap-junctions $\gamma_C$'
            ylab = r'Mean drive $\nu_{EI}$'
            ax.set_xlabel(xlab, alpha=a)
            ax.set_ylabel(ylab, alpha=a)
            ax.set_xticklabels(ax.get_xticks(), alpha=a)
            ax.set_yticklabels(ax.get_yticks(), alpha=a)

            # plasticity trajectory
            # x = spikingSimulation.gamma*spikingSimulation.N_I
            # y = (spikingSimulation.stimulation+neuronI.N_mean)/(neuronI.V_th - neuronI.vReset)
            # plt.plot(x,y)
            if front:
                dataframe['logburst'] = (dataframe['burst']).apply(np.log10)
                df_sliced = dataframe[(dataframe['logburst'] > -6) & (dataframe['logburst'] < -5.8)]
                contour = df_sliced[['gammaC', 'nuEI']].get_values()
                yvals, xvals = self.bezier_curve(contour, nTimes=100)
                frontier = np.array([xvals, yvals])
                ax.plot(frontier[1, :], frontier[0, :], '-w', linewidth=3)
            if save:
                plt.tight_layout()
                plt.savefig(DIRECTORY + filename)
        except:
            print('Error reshaping array')
        return ax

    def plotDiagramCSD(self, figure, ax, dataframe, title, column, filename, save=False, front=False, gridsize=60,
                        extent=[0, 7.0, 0, 200], cmap=cx4, bad=False, format=None, vmin=None, vmax=None, plast=False):
        '''
        to check output image orientation
        -1   1

        -2   2
        '''

        ax.set_ylim(ymin=extent[2], ymax=extent[3])
        # Set color transparency (0: transparent; 1: solid)
        a = 1
        # Create a colormap
        customcmap = [(x / 24.0, x / 48.0, 0.05) for x in range(len(dataframe))]
        dataframe = dataframe.fillna(0)
        nbVal = len(pd.unique(dataframe.sG.ravel()))
        da = np.array(dataframe[['tauv', 'sG', column]].sort_values(['sG', 'tauv'], ascending=[1, 0]))
        z = da[:, 2]

        print(z.shape)
        try:
            zr = z.reshape(nbVal, int(len(z) / nbVal)).transpose()
            print(zr.shape)
            if bad:
                cmap.set_bad('white')
                plt.imshow(np.ma.masked_values(zr, 0), cmap=cmap)
            image = ax.imshow(zr, extent=extent, cmap=cmap, interpolation='nearest', vmin=vmin, vmax=vmax,
                              aspect=(extent[1] - extent[0]) / (extent[3] - extent[2]))  # , cmap =cx4)# drawing the function
            # if format == None:
                # plt.colorbar(image, format='%.2g', cax=ax)
            # else:
                # plt.colorbar(image, format=format)
            ax.set_title(title, y=1.03)
            plt.subplots_adjust(top=0.9)

            ax.xaxis.set_label_position('bottom')

            maxX = math.floor(extent[1] * 10) / 10
            maxY = math.ceil(extent[3] / 10) * 10

            ax.set_xlim([extent[0], maxX])
            ax.set_yticks([extent[2], (maxY - extent[2]) / 2, maxY])
            ax.set_xticks([extent[0], (maxX - extent[0]) / 2, maxX])
            xlab = r'Nb of shared Gap-junctions'
            ylab = r'Time constant $\tau_{v}$ from N2'
            ax.set_xlabel(xlab, alpha=a)
            ax.set_ylabel(ylab, alpha=a)
            ax.set_xticklabels(ax.get_xticks(), alpha=a)
            ax.set_yticklabels(ax.get_yticks(), alpha=a)

            if save:
                plt.tight_layout()
                plt.savefig(DIRECTORY + filename)
        except:
            print('Error reshaping array')
        return ax, image

    def plotDiagramChangeCSD(self, figure, ax, dataframe, title, column, filename, save=True, front=False,
                           extent=[0, 7.0, 0, 200], cmap=cx4, bad=False,
                             format=None, vmin=None, vmax=None, both=None, plast=None, sWII=10):


        '''
        to check output image orientation
        -1   1

        -2   2
        '''



        ax.set_ylim(ymin=extent[2], ymax=extent[3])
        # Set color transparency (0: transparent; 1: solid)
        a = 1
        # Create a colormap
        customcmap = [(x / 24.0, x / 48.0, 0.05) for x in range(len(dataframe))]
        dataframe = dataframe.fillna(0)
        nbVal = len(pd.unique(dataframe.sG.ravel()))

        if not plast:
            daBegin = np.array(dataframe[['tauv', 'sG', column+'Begin']].sort_values(['sG', 'tauv'], ascending=[1, 0]))
            daEnd = np.array(dataframe[['tauv', 'sG', column+'End']].sort_values(['sG', 'tauv'], ascending=[1, 0]))

        else:
            df = dataframe
            dfplast = df[(df['LTD'] == True) & (df['sWII'] == sWII)]
            dfnoplast = df[(df['LTD'] == False) & (df['sWII'] == sWII)]
            daBegin = np.array(
                dfnoplast[['tauv', 'sG', column + 'End']].sort_values(['sG', 'tauv'], ascending=[1, 0]))
            daEnd = np.array(dfplast[['tauv', 'sG', column + 'End']].sort_values(['sG', 'tauv'], ascending=[1, 0]))

        zBegin = daBegin[:, 2]
        zEnd = daEnd[:, 2]
        zrBegin = zBegin.reshape(nbVal, int(len(zBegin) / nbVal)).transpose()
        zrEnd = zEnd.reshape(nbVal, int(len(zEnd) / nbVal)).transpose()
        zr = zrEnd / zrBegin

        if both:
            if not plast:
                da2Begin = np.array(
                    dataframe[['tauv', 'sG', column[:-1] + '2Begin']].sort_values(['sG', 'tauv'],
                                                                                  ascending=[1, 0]))
                da2End = np.array(
                    dataframe[['tauv', 'sG', column[:-1] + '2End']].sort_values(['sG', 'tauv'],
                                                                                           ascending=[1, 0]))
            else:
                da2Begin = np.array(
                    dfnoplast[['tauv', 'sG', column[:-1] + '2End']].sort_values(['sG', 'tauv'],
                                                                                  ascending=[1, 0]))
                da2End = np.array(dfplast[['tauv', 'sG', column[:-1] + '2End']].sort_values(['sG', 'tauv'],
                                                                                              ascending=[1, 0]))
            z2Begin = da2Begin[:, 2]
            z2End = da2End[:, 2]
            zr2Begin = z2Begin.reshape(nbVal, len(z2Begin) // nbVal).transpose()
            zr2End = z2End.reshape(nbVal, len(z2End) // nbVal).transpose()
            zr2 = zr2End / zr2Begin
            # vmax = max(np.max(zr), np.max(zr2))
            if vmax is None:
                vmax = max(np.percentile(zr,98), np.percentile(zr2,98))
                vmin = max(0,2 - vmax)

        elif vmax is None:
            vmax = np.max(zr)
            # 1 = m = (a+b)/2 - > a = 2-b
            vmin = max(0,2 - vmax)


        # print(z.shape)
        try:
            if bad:
                cmap.set_bad('white')
                plt.imshow(np.ma.masked_values(zr, 0), cmap=cmap)
            image = ax.imshow(zr, extent=extent, cmap=cmap, interpolation='nearest', norm=MidpointNormalize(midpoint=1.), vmin=vmin, vmax=vmax,
                              aspect=(extent[1] - extent[0]) / (extent[3] - extent[2]))  # , cmap =cx4)# drawing the function
            # if format == None:
            # plt.colorbar(image, format='%.2g', cax=ax)
            # else:
            # plt.colorbar(image, format=format)
            ax.set_title(title, y=1.03)
            plt.subplots_adjust(top=0.9)

            ax.xaxis.set_label_position('bottom')

            maxX = math.floor(extent[1] * 10) / 10
            maxY = math.ceil(extent[3] / 10) * 10

            ax.set_xlim([extent[0], maxX])
            ax.set_yticks([extent[2], (maxY - extent[2]) / 2, maxY])
            ax.set_xticks([extent[0], (maxX - extent[0]) / 2, maxX])
            xlab = r'Nb of shared Gap-junctions'
            ylab = r'Time constant $\tau_{v}$ from N2'
            ax.set_xlabel(xlab, alpha=a)
            ax.set_ylabel(ylab, alpha=a)
            ax.set_xticklabels(ax.get_xticks(), alpha=a)
            ax.set_yticklabels(ax.get_yticks(), alpha=a)

            if save:
                plt.tight_layout()
                plt.savefig(DIRECTORY + filename)
        except:
            print('Error reshaping array')
        return ax, image



    def savePTSH(self, before, after, h, s, it, DIRECTORY, S, N):
        '''
        Save PTSH data
        '''
        T = before + after
        simsize = len(s)
        print(T / 1000, simsize, T / 1000 / simsize)
        x2 = np.arange(0, T / 1000, T / 1000 / simsize)
        extension = "_S-%d_N-%d_T-%d" % (S, N, T)
        np.save(DIRECTORY + extension + '_data.npy', np.array([np.arange(len(h)), np.array(h) / it]))

    def func(self, x, a, b, c):
        return a * np.exp(-b * x) + c

    def plotRaster(self, spikes_x, spikes_y, ax=None, titlestr=""):
        if not ax:
            f = plt.figure(figsize=(4, 3))
            ax = f.add_subplot(111)
        # ax.set_xticks([])
        # ax.set_yticks([0,300])
        # ax.set_xlabel('Time [1.5s]')
        # ax.set_ylabel('Neuron indices [0-300]')
        # ax.set_title('Neuronal Activity')
        if ax == None:
            plt.plot(spikes_x, spikes_y, '.', markersize=1, color='grey')
            plt.title(titlestr, y=1.08)
        else:
            ax.plot(spikes_x, spikes_y, '.', markersize=1, color='grey')
            ax.set_title(titlestr, y=1.08)
        return ax
        # plt.savefig(DIRECTORY + extension + '_raster.pdf')
        # plt.savefig(DIRECTORY + extension + '_raster.png')

    def plotRasterGPU(self, spikes_x, spikes_y, titlestr="", saveImg=0):
        '''
        Take advantage of WebGL to draw raster plot
        '''
        # output_file("spikesGPU.html", title="Neural activity")

        p = figure(plot_width=1000, plot_height=500, webgl=True, title=titlestr)
        if len(spikes_x) > 100000:
            p.scatter(spikes_x[0:100000], spikes_y[0:100000], alpha=0.5)
        else:
            p.scatter(spikes_x, spikes_y, alpha=0.5)

        save(p, filename=titlestr) if saveImg else show(p)

    def fourier(self, signal):
        f_val, p_val = self.maxPowerFreq(signal[int(signal.shape[0] / 2):], 0.25 / 1000)
        return [f_val, p_val]

    def maxPowerFreq(self, y, dt):
        # return the max power of the signal and its associated frequency
        fs = 1. / dt
        y = y - np.mean(y)
        t = np.arange(0, y.shape[0], 1)
        p1, f1 = psd(y, NFFT=len(t), pad_to=len(t), Fs=fs)
        powerVal = 10 * np.log10(max(p1))
        powerFreq = np.argmax(p1) * np.max(f1) / len(f1)

        return powerFreq, powerVal

    def bernstein_poly(self, i, n, t):
        """
         The Bernstein polynomial of n, i as a function of t
        """

        return comb(n, i) * (t ** (n - i)) * (1 - t) ** i

    def bezier_curve(self, points, nTimes=1000):
        """
           Given a set of control points, return the
           bezier curve defined by the control points.

           points should be a list of lists, or list of tuples
           such as [ [1,1],
                     [2,3],
                     [4,5], ..[Xn, Yn] ]
            nTimes is the number of time steps, defaults to 1000

            See http://processingjs.nihongoresources.com/bezierinfo/
        """

        nPoints = len(points)
        xPoints = np.array([p[0] for p in points])
        yPoints = np.array([p[1] for p in points])

        t = np.linspace(0.0, 1.0, nTimes)

        polynomial_array = np.array([self.bernstein_poly(i, nPoints - 1, t) for i in range(0, nPoints)])

        xvals = np.dot(xPoints, polynomial_array)
        yvals = np.dot(yPoints, polynomial_array)

        return xvals, yvals

    def frontgamma(self, nuEI, frontier, tol=10):
        '''
        Return the value of GAMMA for a value of NUEI
        '''
        indices = np.where(np.logical_and(frontier[0] <= nuEI + tol, frontier[0] >= nuEI - tol))
        return np.mean(frontier[1][indices[0]])

    def find_nearest(self, array, value, dataframe):
        gamma = (pd.unique(dataframe['gammaC']))
        idx = (np.abs(array - value)).argmin()
        return array[idx], idx, gamma[idx]

    def fixpoint(self, df, nuEI, ratio, rule, g0=5):
        '''
        Search and return the fix point for a plasticity rule, or 0 if none

        :param df:
        :param nuEI:
        :param ratio:
        :param rule: [1 hardbound
        2 softbound passive
        3 softbound spiking no bursting
        4 hardbound passive
        5 hardbound spiking non bursting]
        :param g0:
        :return:
        '''
        LTP = 10
        LTD = LTP * ratio
        df_sliced = df[df['nuEI'] == nuEI]
        df_sliced = df_sliced.sort_values(by=['gammaC'])
        bg = df_sliced[['burst', 'gammaC']].get_values()
        sg = df_sliced[['spike', 'gammaC']].get_values()
        rg = df_sliced[['non-burst', 'gammaC']].get_values()
        df_sliced['passive'] = df_sliced['burst'].apply(lambda x: 1 - x)
        passive = df_sliced[['passive', 'gammaC']].get_values()
        #     sg = 1 - np.array(bg)
        cg = df_sliced[['corI', 'gammaC']].get_values()
        if rule == 1:
            dG = LTP * sg[:, 0] - LTD * bg[:, 0]
        elif rule == 4:
            dG = LTP * passive[:, 0] - LTD * bg[:, 0]
        elif rule == 5:
            dG = LTP * rg[:, 0] - LTD * bg[:, 0]
        elif rule == 0:
            dG = LTP * (g0 - sg[:, 1]) / g0 * sg[:, 0] - LTD * bg[:, 0]
        elif rule == 2:
            dG = LTP * (g0 - passive[:, 1]) / g0 * passive[:, 0] - LTD * bg[:, 0]
        elif rule == 3:
            dG = LTP * (g0 - rg[:, 1]) / g0 * rg[:, 0] - LTD * bg[:, 0]
        res = 0
        dg2 = np.array(dG)
        if (dg2 > 0).all() or (dg2 < 0).all():
            res = 0
        else:
            minval, idx, _ = self.find_nearest(dg2, 0, df)
            res = sg[idx, 1]
        return res

    def overunder(self, df, frontier, rule):
        '''
        return 0 if fixpoint over in the SYNC regime, 1 otherwise
        '''
        if rule in [0, 1, 5, 3]:
            col = np.arange(3, 5, 0.05)  # ratio for active rule
        else:
            col = np.arange(60, 180, 3)  # ratio for passive rule
        row = np.arange(0, 200, 1)  # nu
        Z = np.zeros(shape=(len(row), len(col)))
        for i, ratio in enumerate(col):
            for j, nuEI in enumerate(row):
                f = self.fixpoint(df, nuEI, ratio, rule)
                border = self.frontgamma(nuEI, frontier)
                Z[len(row) - 1 - j, i] = abs(f - border) * ((f < border) and (f > 0.15) and (nuEI > 48)) * 1.0
        return Z

    def plotStability(self, ax, Z, ylabel, title, extent):
        cx_blue = cubehelix.cmap(reverse=False, start=3., rot=0)
        ax.set_xlabel('ratio')
        ax.set_ylabel(r'Excitatory input $\nu$')
        ax.set_title('Hard. all-spiking', y=1.08)
        image = ax.imshow(Z, interpolation='nearest', extent=extent, cmap=cx_blue,
                          aspect=2 / 200)  # , cmap =cx4)# drawing the function
        plt.colorbar(image)
        return ax

    def plotWeights(self, tauv=15):
        c = self.cortex
        c.tauv = tauv
        c.readSimulation()

        titlestr = r'$N=%d$  $\frac{\alpha_{LTP}}{\alpha_{LTD}}=%d$  $g_0=%.1f$  $\nu=%d$ $sG=%d$ $sW_{II}=%d$ $LTD=%.6g$ $\tau_v=%d$' \
                   % (c.N, c.r, c.g, c.sigma, c.sG, c.sWII, c.LTD, c.tauv)

        GAP2D = c.readMatrix(type="GAP")
        GAP2D0 = c.readMatrix(type="GAP0")
        WII2D = c.readMatrix(type='WII')
        fig = plt.figure(figsize=(20, 14))

        fontsize = 10
        matplotlib.rc('xtick', labelsize=fontsize)
        matplotlib.rc('ytick', labelsize=fontsize)
        matplotlib.rc('axes', labelsize=fontsize)
        matplotlib.rc('axes', titlesize=fontsize)

        ax0 = fig.add_subplot(421)
        ax0.plot(xax(c.gamma, c.T), c.gamma, color='g')
        ax0.plot(xax(c.gammaN1, c.T), c.gammaN1*(1+0.0*c.sG), color='r')
        ax0.plot(xax(c.gammaN2, c.T), c.gammaN2*(1+0.0*c.sG), color='c')
        ax0.set_title('Mean gap junction coupling')

        ax01 = fig.add_subplot(422)
        T = 2000
        dt = 1
        t = np.arange(0, T, dt)
        F = np.logspace(0.5, 2.3, 200)
        mod = resonanceFS(F, tauv=tauv)
        mod15 = resonanceFS(F, tauv=15)
        ax01.semilogx(F, mod / np.nanmax(mod), label='%.1f' % (F[np.argmax(mod)]), color='c')
        ax01.semilogx(F, mod15 / np.nanmax(mod15), label='%.1f' % (F[np.argmax(mod15)]), color='r')
        ax01.set_ylim([0, 1.05])
        # plt.legend()
        ax01.set_xlabel('Stimulus Frequency [Hz]')
        ax01.set_ylabel('Normalised Response Amplitude')
        ax01.set_xlim([0, 200])
        ax01.set_title('Subthreshold Resonant Property', y=1.08)
        ax01.legend()

        # --------------------------------------------------------------------------------
        # PLot weights
        # --------------------------------------------------------------------------------
        ax1 = plt.subplot(434)
        # plt.imshow(np.array(GAP2D)[100:150,1:20], interpolation='nearest')
        im1 = ax1.imshow(np.array(GAP2D0), interpolation='nearest')
        ax1.set_title(r'$\gamma(t = 0)$')
        fig.colorbar(im1)
        # plt.figure(figsize=(10,10))
        ax2 = plt.subplot(435)
        im2 = ax2.imshow(np.array(GAP2D), interpolation='nearest')
        ax2.set_title(r'$\gamma(t = END)$')
        fig.colorbar(im2)
        # plt.figure(figsize=(10,10))
        ax3 = plt.subplot(436)
        im3 = ax3.imshow(np.array(WII2D), interpolation='nearest')
        ax3.set_title(r'$W_{II}$')

        #--------------------------------------------------------------------------------
        # RASTER PLOT begin and end of simulation
        #--------------------------------------------------------------------------------
        ax1 = fig.add_subplot(425)
        ax2 = fig.add_subplot(426)
        self.plotRaster(c.spikes_x[0:4000], c.spikes_y[0:4000], ax=ax1)
        self.plotRaster(c.spikes_x[-5000:-1000], c.spikes_y[-5000:-1000], ax=ax2)

        #--------------------------------------------------------------------------------
        # Cross Spectrum Gaph
        #--------------------------------------------------------------------------------
        ax7 = fig.add_subplot(427)
        ax8 = fig.add_subplot(428)
        ax7 = self.plotCSD(c.i1, c.i2, start=1000, end=5000, ax=ax7, sharey=True)
        ax8 = self.plotCSD(c.i1, c.i2, start=-4000, end=-1, ax=ax8, sharey=True)

        plt.suptitle(titlestr, y=0.93)
        plt.savefig(DIRECTORY + "full_sWII-%d_sG-%d_WII-%d_G-%d_N-%d_t-%d_LTD-%d_tauv-%d" % (
            c.sWII, c.sG, c.WII, c.g, c.N, c.T, c.LTD > 1e-8, c.tauv))
        del fig
        # gr.plotRasterGPU(spikes_x[:],spikes_y[:], "test_%s.html"%(str(G)), saveImg=0)
        return 0

    def plotCSD(self, i1, i2, start=0, end=8000, ax=None, sharey = None):
        f, Pxy = signal.csd(i1[start:end], i2[start:end], fs=1 / 0.00025, nperseg=1024)

        if sharey:
            f2, Pxy2 = signal.csd(i1[-end:-1], i2[-end:-1], fs=1 / 0.00025, nperseg=1024)
            ax.plot(f, np.abs(Pxy) )
            ax.set_ylim([0, max(np.max(np.abs(Pxy2)), np.max(np.abs(Pxy)))])
        else:
            ax.plot(f, np.abs(Pxy))
        ax.set_xlabel('frequency [Hz]')
        ax.set_ylabel('CSD [V**2/Hz]')
        ax.set_xlim([0, 300])
        ax.set_title('Cross-spectrum between LFPs of 2 networks')
        return ax

    def valCSD(self, start=0, end=8000):
        '''
        Return the max value of the CSD at the beginning and at the end of the simulation
        :param i1:
        :param i2:
        :return:
        '''
        dictRes = {}
        c = self.cortex
        f, Pxy = signal.csd(c.i1[start:end], c.i2[start:end], fs=1 / 0.00025, nperseg=1024)
        f2, Pxy2 = signal.csd(c.i1[-end:-1], c.i2[-end:-1], fs=1 / 0.00025, nperseg=1024)

        maxBegin = np.max(np.abs(Pxy))
        argmaxBegin = np.argmax(np.abs(Pxy))
        maxEnd = np.max(np.abs(Pxy2))
        argmaxEnd = np.argmax(np.abs(Pxy2))

        dictRes = {'maxBegin': maxBegin, 'argmaxBegin': argmaxBegin, 'maxEnd': maxEnd, 'argmaxEnd': argmaxEnd}

        dictRes['f1Begin'] = self.fourier(c.i1[start:end])
        dictRes['f2Begin'] = self.fourier(c.i2[start:end])
        dictRes['f1End'] = self.fourier(c.i1[-end:-1])
        dictRes['f2End'] = self.fourier(c.i2[-end:-1])
        dictRes['fBothBegin'] = self.fourier(np.array(c.i1[start:end])+np.array(c.i2[start:end]))
        dictRes['fBothEnd'] = self.fourier(np.array(c.i1[-end:-1])+np.array(c.i2[-end:-1]))

        return dictRes

    def plotCoherenceEvolution(self, dataframe, kind='max', sWII=10, LTD=True,
                               vmin = None, vmax=None):
        fig = plt.figure(figsize=self.figsize21)

        df = dataframe
        df = df[(df['LTD'] == LTD) & (df['sWII'] == sWII)]
        extent = [np.min(df['sG']), np.max(df['sG']), np.min(df['tauv']), np.max(df['tauv'])]

        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)

        cmap = plt.cm.RdBu_r
        if vmax is None:
            if kind == 'max':
                vmin = min(list(map(np.min, [df.maxBegin, df.maxEnd])))
                vmax = max(list(map(np.max, [df.maxBegin, df.maxEnd])))
            elif kind == 'argmax':
                vmin = min(list(map(np.min, [df.argmaxBegin, df.argmaxEnd])))
                vmax = max(list(map(np.max, [df.argmaxBegin, df.argmaxEnd])))
            elif kind == 'fBoth':
                vmin = min(list(map(np.min, [df.fBothBegin, df.fBothEnd])))
                vmax = max(list(map(np.max, [df.fBothBegin, df.fBothEnd])))
            elif kind == 'pBoth':
                vmin = min(list(map(np.min, [df.pBothBegin, df.pBothEnd])))
                vmax = max(list(map(np.max, [df.pBothBegin, df.pBothEnd])))

        title = r'Network %s: start' % kind
        column = kind + 'Begin'
        filename = column + '%s' % self.ext
        ax1, im = self.plotDiagramCSD(fig, ax1, df, title, column, filename, extent=extent, cmap=cmap,
                                    vmin=vmin, vmax=vmax)

        title = r'Network %s: end' % kind
        column = kind + 'End'
        filename = column + '%s' % self.ext
        ax2, im = self.plotDiagramCSD(fig, ax2, df, title, column, filename, extent=extent, cmap=cmap,
                                    vmin=vmin, vmax=vmax)

        ax2.set_yticks([])
        ax2.set_ylabel('')

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes(self.axes21)
        fig.colorbar(im, cax=cbar_ax)

        plt.savefig(DIRECTORY + kind + '_cluster_plast%s%s' % (str(LTD), self.ext))
        return 0

    def plotEvolution(self, dataframe, kind='frequency', sWII=10, LTD=True, vmin=None, vmax=None):
        fig = plt.figure(figsize=self.figsize22)

        df = dataframe
        df = df[(df['LTD'] == LTD) & (df['sWII'] == sWII)]
        extent = [np.min(df['sG']), np.max(df['sG']), np.min(df['tauv']), np.max(df['tauv'])]

        ax1 = fig.add_subplot(221)
        ax2 = fig.add_subplot(222)
        ax3 = fig.add_subplot(223)
        ax4 = fig.add_subplot(224)

        cmap = plt.cm.RdBu_r
        if vmax == None:
            if kind == 'frequency':
                vmax = max(list(map(np.max, [df.f1Begin, df.f2Begin, df.f1End, df.f2End])))
            else:
                vmax = max(list(map(np.max, [df.p1Begin, df.p2Begin, df.p1End, df.p2End])))

        if vmin ==None:
            if kind == 'frequency':
                vmin = min(list(map(np.min, [df.f1Begin, df.f2Begin, df.f1End, df.f2End])))
            else:
                vmin = min(list(map(np.min, [df.p1Begin, df.p2Begin, df.p1End, df.p2End])))

        title = r'Network %s N1: start' % kind
        column = kind[0] + '1Begin'
        filename = column + '%s'%self.ext
        ax1, _ = self.plotDiagramCSD(fig, ax1, df, title, column, filename, extent=extent, cmap=cmap,
                                   vmin=vmin, vmax=vmax)
        ax1.set_xticks([])
        ax1.set_xlabel('')

        title = r'Network %s N1: end' % kind
        column = kind[0] + '1End'
        filename = column + '.%s'%self.ext
        ax2, _ = self.plotDiagramCSD(fig, ax2, df, title, column, filename, extent=extent, cmap=cmap,
                                   vmin=vmin, vmax=vmax)
        ax2.set_xticks([])
        ax2.set_xlabel('')
        ax2.set_yticks([])
        ax2.set_ylabel('')

        title = r'Network %s N2: start' % kind
        column = kind[0] + '2Begin'
        filename = column + '%s'%self.ext
        ax3, _ = self.plotDiagramCSD(fig, ax3, df, title, column, filename, extent=extent, cmap=cmap,
                                   vmin=vmin, vmax=vmax)

        title = r'Network %s N2: end' % kind
        column = kind[0] + '2End'
        filename = column + '%s'%self.ext
        ax4, im = self.plotDiagramCSD(fig, ax4, df, title, column, filename, extent=extent, cmap=cmap,
                                    vmin=vmin, vmax=vmax)
        ax4.set_yticks([])
        ax4.set_ylabel('')

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes(self.axes22)
        fig.colorbar(im, cax=cbar_ax)

        plt.savefig(DIRECTORY + kind + '_cluster%s%s' % (str(LTD), self.ext))
        return 0

    def plotChange(self, dataframe, kind='max', network='', both=None, vmin=None, vmax=None,
                   sWII=10, title="", LTD = True):
        fig = plt.figure(figsize=self.figsize11)

        df = dataframe
        df = df[(df['LTD'] == LTD) & (df['sWII'] == sWII)]
        extent = [np.min(df['sG']), np.max(df['sG']), np.min(df['tauv']), np.max(df['tauv'])]

        ax1 = fig.add_subplot(111)

        cmap = plt.cm.RdYlGn

        title = r'%s' % (title)
        column = kind + 'Begin'
        filename = column + '.pdf'
        ax1, im = self.plotDiagramChangeCSD(fig, ax1, df, title,
                                          kind, filename, extent=extent, cmap=cmap, both=both,
                                          vmin=vmin, vmax=vmax)

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes(self.axes11)
        cbar = fig.colorbar(im, cax=cbar_ax)

        if title == "":
            plt.savefig(DIRECTORY + kind + 'change_cluster%s%s' % (str(LTD), self.ext))
        else:
            plt.savefig(DIRECTORY + kind + '%s%s' % (str(str.replace(title, " ", "_")), self.ext))

    def plotChange2(self, dataframe, kind='max', network='', title='', both=None, vmin=None, vmax=None, sWII=10, LTD = True):
        fig = plt.figure(figsize=self.figsize12)

        df = dataframe
        df = df[(df['LTD'] == LTD) & (df['sWII'] == sWII)]
        extent = [np.min(df['sG']), np.max(df['sG']), np.min(df['tauv']), np.max(df['tauv'])]

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        cmap = plt.cm.RdYlGn

        title = r'%s' % (title)
        filename = "".join(title.split(' ')) + '.pdf'
        ax1, im = self.plotDiagramChangeCSD(fig, ax1, df, title, kind + '1', filename, extent=extent, cmap=cmap,
                                          both=both,
                                          vmin=vmin, vmax=vmax)

        title = r'%s' % (title)
        filename = "".join(title.split(' ')) + '2.pdf'
        ax2, im = self.plotDiagramChangeCSD(fig, ax2, df, title, kind + '2', filename, extent=extent, cmap=cmap,
                                          both=both,
                                          vmin=vmin, vmax=vmax)

        ax2.set_title('')
        ax1.set_xticks([])
        ax1.set_xlabel("")
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes(self.axes12)
        cbar = fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()
        plt.savefig(DIRECTORY + kind + 'both_change_cluster%s%s' % (str(LTD), self.ext))
        return 0

    def plotChangePlast(self, dataframe, kind='max', network='', title='', both=None, vmax=None, sWII=10):

        fig = plt.figure(figsize=self.figsize11)

        df = dataframe
        df = df[(df['sWII'] == sWII)]
        extent = [np.min(df['sG']), np.max(df['sG']), np.min(df['tauv']), np.max(df['tauv'])]

        ax1 = fig.add_subplot(111)

        cmap = plt.cm.RdYlGn

        title = r'%s' % (title)
        column = kind + 'Begin'
        filename = column + '.pdf'
        ax1, im = self.plotDiagramChangeCSD(fig, ax1, df, title,
                                            kind, filename, extent=extent, cmap=cmap, both=both,
                                            vmax=vmax, plast=True, sWII=sWII)

        fig.subplots_adjust(right=0.85)
        cbar_ax = fig.add_axes(self.axes11)
        cbar = fig.colorbar(im, cax=cbar_ax)

        plt.savefig(DIRECTORY + kind + '_changePLAST_cluster%s'% self.ext)


    def plotChangePlast2(self, dataframe, kind='max', network='', title='', both=None,
                        vmin=None, vmax=None, sWII=10):
        fig = plt.figure(figsize=self.figsize12)

        df = dataframe
        extent = [np.min(df['sG']), np.max(df['sG']), np.min(df['tauv']), np.max(df['tauv'])]

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        cmap = plt.cm.RdYlGn

        title = r'%s' % (title)
        filename = "".join(title.split(' ')) + '%s' % self.ext
        ax1, im = self.plotDiagramChangeCSD(fig, ax1, df, title, kind + '1', filename, extent=extent, cmap=cmap,
                                          both=both,
                                          vmin=0, vmax=vmax, sWII=sWII, plast=True, save=False)

        title = r'%s' % (title)
        filename = "".join(title.split(' ')) + '2%s' % self.ext
        ax2, im = self.plotDiagramChangeCSD(fig, ax2, df, title, kind + '2', filename, extent=extent, cmap=cmap,
                                          both=both,
                                          vmin=vmin, vmax=vmax, sWII=sWII, plast=True, save=False)

        ax2.set_title('')
        ax1.set_xticks([])
        ax1.set_xlabel("")
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes(self.axes12)
        cbar = fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()
        plt.savefig(DIRECTORY + kind + 'both_changePLAST_cluster%s'%self.ext)
        return 0

    def plotGamma(self, dataframe, kind='max', network='', title='', both=None,
                             vmin=None, vmax=None, sWII=10):

        fig = plt.figure(figsize=self.figsize12)

        df = dataframe
        extent = [np.min(df['sG']), np.max(df['sG']), np.min(df['tauv']), np.max(df['tauv'])]

        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        cmap = plt.cm.RdYlGn

        title = r'%s' % (title)
        filename = "".join(title.split(' ')) + '%s' % self.ext
        ax1, im = self.plotDiagramCSD(fig, ax1, df, title, kind + '1', filename, extent=extent, cmap=cmap,
                                            vmin=0, vmax=vmax, plast=True, save=False)

        title = r'%s' % (title)
        filename = "".join(title.split(' ')) + '2%s' % self.ext
        ax2, im = self.plotDiagramCSD(fig, ax2, df, title, kind + '2', filename, extent=extent, cmap=cmap,
                                            vmin=vmin, vmax=vmax, plast=True, save=False)
        ax2.set_title('')
        ax1.set_xticks([])
        ax1.set_xlabel("")
        fig.subplots_adjust(right=0.9)
        cbar_ax = fig.add_axes(self.axes12)
        cbar = fig.colorbar(im, cax=cbar_ax)
        plt.tight_layout()
        plt.savefig(DIRECTORY + kind + '_gamma_cluster%s' % self.ext)


    def plotSGCoherence(self, df, tauv=0, ax=None, sWII=10, LTD=True):
        df_sliced = df[(df['LTD'] == LTD) & (df['tauv'] == tauv) & (df['sWII'] == sWII)]
        data = df_sliced[['sG', 'maxBegin', 'maxEnd']].get_values()
        if ax is None:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
        im = ax.plot(data[:, 0], data[:, 1], label='start')
        im = ax.plot(data[:, 0], data[:, 2], label='end')
        ax.legend()
        ax.set_title('coherence vs sG, tauv=%d' % tauv)
        return 0

    def plotCoherence(self, df, sG=0, ax=None, sWII=10, LTD=True):
        df_sliced = df[(df['LTD'] == LTD) & (df['sG'] == sG) & (df['sWII'] == sWII)]
        data = df_sliced[['tauv', 'maxBegin', 'maxEnd']].get_values()
        if ax is None:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
        im = ax.scatter(np.ones(len(data)), data[:, 1], c=np.arange(len(data)))
        ax.scatter(np.ones(len(data)) * 2, data[:, 2], c=np.arange(len(data)))
        ax.set_title('peak coherence sG=%d' % sG)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['start', 'end'], rotation='vertical')
        return 0

    def plotArgCoherence(self, df, sG=0, ax=None, LTD=True):
        df_sliced = df[(df['LTD'] == LTD) & (df['sG'] == sG)]
        data = df_sliced[['tauv', 'argmaxBegin', 'argmaxEnd']].get_values()
        if ax is None:
            fig = plt.figure(figsize=(4, 4))
            ax = fig.add_subplot(111)
        ax.scatter(np.ones(len(data)), data[:, 1], c=np.arange(len(data)))
        ax.scatter(np.ones(len(data)) * 2, data[:, 2], c=np.arange(len(data)))
        ax.set_title('peak frequency sG=%d' % sG)
        ax.set_xticks([1, 2])
        ax.set_xticklabels(['start', 'end'], rotation='vertical')
        return ax


################################################################################
# Classes
################################################################################
class IzhNeuron:
    def __init__(self, label, a, b, c, d, v0, u0=None):
        self.label = label

        self.a = a
        self.b = b
        self.c = c
        self.d = d

        self.v = v0
        self.u = u0 if u0 is not None else b * v0


class IzhSim:
    def __init__(self, n, T, dt=0.1):
        self.neuron = n
        self.dt = dt
        self.t = t = np.arange(0, T + dt, dt)
        self.stim = np.zeros(len(t))
        self.x = 5
        self.y = 140
        self.du = lambda a, b, v, u: a * (b * v - u)

    def integrate(self, n=None):
        if n is None: n = self.neuron
        trace = np.zeros((3, len(self.t)))
        b = 2
        p = 0
        tau_p = 10
        for i, j in enumerate(self.stim):
            #       n.v += self.dt * (0.04*n.v**2 + self.x*n.v + self.y - n.u + self.stim[i])
            n.v += self.dt * 1 / 40.0 * (0.25 * (n.v ** 2 + 110 * n.v + 45 * 65) - n.u + self.stim[i])
            #       n.u += self.dt * self.du(n.a,n.b,n.v,n.u)
            n.u += self.dt * 0.015 * (b * (n.v + 65) - n.u)
            p += self.dt / tau_p * ((n.v > 0) * tau_p / self.dt - p)
            if n.v > 0:
                trace[0, i] = 30
                n.v = -55
                n.u += 50
            else:
                trace[0, i] = n.v
                trace[1, i] = n.u

            if n.v < -65:
                b = 10
            else:
                b = 2
            trace[2, i] = p
        return trace

class MidpointNormalize(colors.Normalize):
    def __init__(self, vmin=None, vmax=None, midpoint=None, clip=False):
        self.midpoint = midpoint
        colors.Normalize.__init__(self, vmin, vmax, clip)

    def __call__(self, value, clip=None):
        # I'm ignoring masked values and all kinds of edge cases to make a
        # simple example...
        x, y = [self.vmin, self.midpoint, self.vmax], [0, 0.5, 1]
        return np.ma.masked_array(np.interp(value, x, y))