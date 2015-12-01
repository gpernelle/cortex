from utils import *

class IO:
    def __init__(self):
        self.computer = ""
        self.executable_path = ""
        self.data_path = ""
        self.executable_name = ""

    def runSimulation(self, N, i, G ,S, d1, d2, d3, before, after, s, WII, tq, thq):
        ext = "_%d.txt"%i
        sh.cd(self.executable_path)
        subprocess.check_output([self.executable_name, '-N', str(N), '-ext', str(ext),
                                '-d1', str(d1), '-d2', str(d2), '-d3', str(d3) ,
                                '-before', str(before), '-after', str(after),
                                 '-S', str(S),  '-G', str(G), '-s', str(s),
                                 '-WII', str(WII),   '-tq', str(tq),  '-thq', str(thq)])


    def readSimulationSSP1(self, N, i, G,S, d1, d2, d3, before, after):
        T = d1+d2+d3
        DIRECTORY = self.data_path
        # simulation parameters
        g = G
        TImean = 50
        glob = 1
        N = N
        ext = "_%d.txt"%i
        # compute the paths of data files.
        extension = "_g-%d_TImean-%d_T-%d_Glob-%d_dt-0.25_N-%d_S-%d_WII-%d" % (g, TImean,T, glob, N, S)
        extension += ext

        ssp1_p = DIRECTORY + "ssp"+ extension
        ssp1 = np.fromfile(ssp1_p, dtype='double', count = -1, sep=" ")
        return ssp1

    def readSimulation(self, N, i, G,S, d1, d2, d3, before, after, WII):
        T = d1+d2+d3
        DIRECTORY = self.data_path
        # simulation parameters
        g = G
        TImean = 50
        glob = 1
        N = N
        dt = 0.25
        ext = "_%d.txt"%i
        # compute the paths of data files.
        extension = "_g-%.1f_TImean-%d_T-%d_Glob-%d_dt-0.25_N-%d_S-%d_WII-%d" % (g, TImean,T, glob, N, S, WII)
        extension += ext

        path_x  = DIRECTORY + "spike_x" + extension
        path_x_tc  = DIRECTORY + "spike_x_tc" + extension
        path_y  = DIRECTORY + "spike_y"+ extension
        path_y_tc  = DIRECTORY + "spike_y_tc"+ extension
        path_g  = DIRECTORY + "gamma"+ extension
        path_c  = DIRECTORY + "correlation"+ extension
        ssp1_p = DIRECTORY + "ssp"+ extension
        p_p = DIRECTORY + "p"+ extension
        q_p = DIRECTORY + "q"+ extension
        lowsp_p = DIRECTORY + "LowSp"+ extension
        vm_p = DIRECTORY + "vm"+ extension
        #ssp2_p = DIRECTORY +"ssp2"+ extension
        #ssp3_p = DIRECTORY +"ssp3"+ extension

        RON_I_p = DIRECTORY +"RON_I"+ extension
        RON_V_p = DIRECTORY +"RON_V"+ extension
        V_p = DIRECTORY +"V"+ extension

        stimulation_p = DIRECTORY +"stimulation"+ extension
    #     print path_g

        spikes_x = np.fromfile(path_x, dtype='uint',count =  -1, sep =" ")
        spikes_x_tc = np.fromfile(path_x_tc, dtype='uint',count =  -1, sep =" ")
        spikes_y = np.fromfile(path_y, dtype='uint',count =  -1, sep =" ")
        spikes_y_tc = np.fromfile(path_y_tc, dtype='uint', count = -1, sep=" ")
        gamma = np.fromfile(path_g, dtype='double', count = -1, sep=" ")
        correlation = np.fromfile(path_c, dtype='double', count = -1, sep=" ")
        ssp1 = np.fromfile(ssp1_p, dtype='double', count = -1, sep=" ")
        p = np.fromfile(p_p, dtype='double', count = -1, sep=" ")
        q = np.fromfile(q_p, dtype='double', count = -1, sep=" ")
        LowSp = np.fromfile(lowsp_p, dtype='double', count = -1, sep=" ")
        vm = np.fromfile(vm_p, dtype='double', count = -1, sep=" ")
        #ssp2 = np.fromfile(ssp2_p, dtype='double', count = -1, sep=" ")
        #ssp3 = np.fromfile(ssp3_p, dtype='double', count = -1, sep=" ")
        stimulation = np.fromfile(stimulation_p, dtype='double', count = -1, sep=" ")
        return spikes_x, spikes_y, spikes_x_tc, spikes_y_tc, gamma, correlation, ssp1, stimulation, p, q, LowSp, vm

class Cortex(IO):
    def __init__(self):
        self.computer = "GP1514"
        self.executable_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/cortex/cortex/" % self.computer
        self.data_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/cortex/data/" % self.computer
        self.executable_name = './cortex'

class TRN(IO):
    def __init__(self):
        self.computer = "GP1514"
        self.executable_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/TRN/TRN/" % self.computer
        self.data_path = "/Users/%s/Dropbox/ICL-2014/Code/C-Code/TRN/data/" % self.computer
        self.executable_name = './trn'