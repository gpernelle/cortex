//
//  utils.cpp
//  cortex
//
//  Created by GP1514 Pernelle on 22/09/2015.
//  Copyright (c) 2015 GP1514 Pernelle. All rights reserved.
//

#include "utils.h"

void Simulation::initData() {
//    PLAST_RULE = "nonbursting";
    PLAST_RULE = "spiking";
    CLUSTER = true;
    localWII = true;
//    PLAST_RULE = "passive";
    SPINDLE_LOWPASS = false;
    COMPUTE_PLAST = true;
    MIN_PLAST_THRESH = false;
    CORRELATION = false;
    FOURIER = true;
    SPIKES = true;
    CONSOLE = false;
    SOFT = true;
    GLOB = false;
    BOTH = true;
    RESONANCE = false;
    DEBUG = false;
    simName= "";

    LTD = 0;
    LTP = 0;

//    dt = 0.25;
    dt = 0.1;
//    dt = 1.00;
    T = 1000 / dt;
    connectTime = 10000;
    before = T;
    after = T;
    d1 = T;
    d2 = T;
    d3 = T;
    T1 = (int) d1/dt;
    T2 = (int) d1/dt + d2/dt;
    T3 = (int) d1/dt + d2/dt + d3/dt;
    NI = (int) 4;
    NE = (int) 20;
    N = NE + NI;
    r = 0; //when r = 0, simulation with only inhibitory neurons
    stimulation = 70;
    sharedWII = 0;
    sharedG = 0;

    // GET HOSTNAME TO ADJUT PATH in function of host
    char hostname[1024];
    hostname[1023] = '\0';
    gethostname(hostname,1023);
    
    csv_directory = "/Dropbox/0000_PhD/csv/";

    #ifdef __APPLE__
        if(!strcmp(hostname,"dyn1147-170.insecure.ic.ac.uk") or !strcmp(hostname,"ip-static-94-242-199-231.server.lu") or (1==1) ){
            root = "/Users/";
            computer = "GP1514";
            directory = "/Dropbox/ICL-2014/Code/C-Code/cortex/data/newdata/";
        }
        else if (!strcmp(hostname,"CNL-Brain1")){
            root = "/mnt/DATA/";
            computer = "gp1514";
            directory = "/Projects/github/cortex/data/newdata/";
        }
        else {
            root = "/Users/";
            computer = "GP1514";
            directory = "/Projects/github/cortex/data/";
        }
    #elif __linux__
        if (!strcmp(hostname,"CNL-Brain1")){
                root = "/mnt/DATA/";
                computer = "gp1514";
                directory = "/Projects/github/cortex/data/newdata/";
        }
        else {
            root = "/home/";
            computer = "gp1514";
            directory = "/Projects/github/cortex/data/newdata/";
        }

    #endif

    ext = "";
    path = root + computer + directory;
    csv_path = root + computer + csv_directory;

    TsigI = 70.0; // Variance of current in the inhibitory neurons
    TsigE = 72; // Variance of current in the inhibitory neurons
    tau_I = 2.0; // Time constant to filter the synaptic inputs
    tau_syn = 5.0;
    tauv = 15; // Time constant of the voltage for the Izhikevich's model

    GammaII = 500.00; // I to I connectivity
    GammaIE = -1000.00; // I to E connectivity
    GammaEE = 500.00; // E to E connectivity
    GammaEI = 1000.00; // E to I connectivity

//    TC_Tsig = 70.0;
//    TC_tau_I = .0;
//    TC_tau_syn = 5.0;

    C_Tsig = 75.0;
    C_tau_I = 1.0;
    C_tau_syn = 3.0;


    gamma_c = 3;    // Initial gap junction strength
//    TImean = 50.0; // Mean imput current in inhibitory neurons.
    TImean = 30.0; // Mean imput current in inhibitory neurons.
    TEmean = 20.0; // Mean input current in inhibitory neurons.
    TIMeanIN = TImean;
    TEMeanIN = TEmean;

//    cout << "Data initialized " << endl;


}

void Simulation::initDuration() {

    T1 = (int) d1/dt;
    T2 = (int) d1/dt + d2/dt;
    T3 = (int) d1/dt + d2/dt + d3/dt;
    T = (int) T3;
//    cout << "***** durations ****"<< endl;
//    cout << T1 << "\t" << T2 << "\t" << T3 <<  endl;
//    cout << T <<endl;
//    cout << std::string(100, '*')<< endl;
}

bool Simulation::saveTime(int t) {
     return (t < T1 + 10000/dt and t > T1 - 10000/dt) or (t < T2 + 10000/dt and t > T2 - 10000/dt) or (t > T - 2*10000/dt);
    }

void Plasticity::initData() {
    // params for plast
//    cout << std::string(100, '*')<< endl;
//    cout <<  "Init plasticity "<< endl;
    FACT = 400.0;
    tau_lowsp = 8.0;
    tau_q = 1.3;
    A_gapD = 2.45e-5 * FACT;
    th_lowsp = 1.5;
    th_q = 1.5;
//    th_lowsp = 2.0;
//    th_q = 2.0; // threshold to differentiate between bursts and spikes
//    A_gapP = A_gapD * 0.6;
//    A_gapP = A_gapD * 0.05;
    A_gapP = A_gapD * 0.5;

    initConnections();

    sim.nbOfGapJunctions = 0;
    for (int m = 0; m < sim.NI; ++m) {
        for (int n = 0; n < sim.NI; ++n) {
            sim.nbOfGapJunctions += int(allowedConnections[m][n]);
        }
    }





    VgapIN = 7.0 / sqrt(sim.nbOfGapJunctions);
    Vgap = sim.gamma_c / sqrt(sim.nbOfGapJunctions);

    VgapLocal = new double *[sim.NI];

    WIILocal = new double *[sim.NI];

    // Synaptic weights
    //
    WII = sim.GammaII / sim.NI / sim.dt;
    WIE = sim.GammaIE / sim.NI / sim.dt;
    if(sim.NE>0) {
        WEE = sim.GammaEE / sim.NE / sim.dt;
        WEI = sim.GammaEI / sim.NE / sim.dt;
    }
    else {
        WEE = 0;
        WEI = 0;
    }


//    cout <<  "Plasticity initialized: WII: "<< WII << "\t NI:" << sim.NI <<  endl;
//    cout << std::string(100, '*')<< endl;
}

/*****************************************************************************
 * CONNECTIONS
 *****************************************************************************/
void Plasticity::initConnections() {
    allowedConnections = new double *[sim.NI];
    allowedConnections0 = new double *[sim.NI];
//    VgapIN = sim.gamma_c / sim.NI;
    for (int i = 0; i < sim.NI; ++i) {
        allowedConnections[i] = new double[sim.NI];
        allowedConnections0[i] = new double[sim.NI];
    }
    for (int m = 0; m < sim.NI; ++m) {
        for (int n = 0; n < sim.NI; ++n) {
            allowedConnections[m][n] = (bool(m<(sim.nbInClusters+sim.sharedG) and n<(sim.nbInClusters+sim.sharedG))
                                        or bool((m>=(sim.nbInClusters-sim.sharedG) )and (n>=(sim.nbInClusters-sim.sharedG)))) and (m!=n);
            allowedConnections0[m][n] = (bool(m<(sim.nbInClusters-sim.sharedG) and n<(sim.nbInClusters-sim.sharedG))
                                        or bool((m>=(sim.nbInClusters+sim.sharedG) )and (n>=(sim.nbInClusters+sim.sharedG)))) and (m!=n);
        }
    }
}

/*****************************************************************************
 * PASTICITY
 *****************************************************************************/
void Plasticity::plasticity(double *burstTh, bool *topotentiate, int t) {
    if (t * sim.dt > 100) {
        double Pmean = getAvg<double>(burstTh, sim.NI);
        if (sim.SOFT) {
            Vgap = Vgap + VgapIN * sim.dt * (-A_gapD * Pmean +
                                             A_gapP * (VgapIN - Vgap) / VgapIN * getSum<bool>(topotentiate, sim.NI) /
                                             (sim.NI * 1.0));
        }
        else {
            Vgap = Vgap + VgapIN * sim.dt * (-A_gapD * Pmean + A_gapP * getSum<bool>(topotentiate, sim.NI) / (sim.NI * 1.0));
        }
        if (sim.MIN_PLAST_THRESH) {
            Vgap = max(1.1/sim.NI, Vgap);
        } else {
            Vgap = max(0.0, Vgap);
        }
    }
}

void Plasticity::plasticityLocal(double *burstTh, bool *topotentiate, int t) {
//    double instantMeanG = 0;
    double dG = 0;
    if (t * sim.dt > 100) {
        //        #pragma omp parallel for num_threads(NUM_THREADS)
        for (int i = 0; i < sim.NI; i++) {
            for (int j = 0; j < i; j++) {
                // TYPO: no VgapIN as factor?
                dG = (i != j) * ( sim.dt * (-A_gapD * burstTh[i] +
                                                    A_gapP * (topotentiate[i]*1.0)));

                if (t * sim.dt > sim.connectTime) {
                    VgapLocal[i][j] = max(0.0, VgapLocal[i][j] + 2 * dG) * allowedConnections[i][j];
                } else {
                    VgapLocal[i][j] = max(0.0, VgapLocal[i][j] + 2 * dG) * allowedConnections0[i][j];
                }
                VgapLocal[j][i] = VgapLocal[i][j];
            }
        }
    }
}

    void MovingAverage::compute(double instantMeanG, int t, int T) {
    if (t % int(T / 1000) == 0 and t>0) {
        meanG /= (counter * 1.0);
        outputVec.push_back(meanG);
        meanG = 0.0;
        counter = 0;
    }
    else if (t * sim.dt > 100) {
        counter++;
        meanG += instantMeanG * sqrt(sim.nbOfGapJunctions);
    }
}

/*****************************************************************************
 * UTILITY FNs
 *****************************************************************************/

double getAvg2D(double **tosum, int N) {
    double result = 0;
    for (int i = 0; i < N; i++) {
        for (int j = 0; j < i; j++) {
            result += tosum[i][j];
        }
    }
    return result / ((N - 1) / 2);
}

void fileExists(const char *pathname) {
    struct stat info;

    if (stat(pathname, &info) != 0)
        printf("cannot access %s\n", pathname);
    else if (info.st_mode & S_IFDIR)  // S_ISDIR() doesn't exist on my windows
        printf("%s is a directory\n", pathname);
    else
        printf("%s is no directory\n", pathname);
}

/*****************************************************************************
 * FFT Fns
 *****************************************************************************/

void Fourier::fft(std::vector<double> &inp, std::vector<double> &out, bool forward = true) {
    fftw_plan plan = fftw_plan_dft_1d(inp.size() / 2,
                                      (fftw_complex *) &inp[0],
                                      (fftw_complex *) &out[0],
                                      forward ? FFTW_FORWARD : FFTW_BACKWARD,
                                      FFTW_ESTIMATE);
    fftw_execute(plan);
    fftw_destroy_plan(plan);
}


void Fourier::computeFFT(vector<double> vm) {

    const int n = int(sim.T);
    double delta = 0.00025;

    vector<double> inp(2 * n, 0);
    vector<double> out(2 * n, 0);

    // DATA formatting
    for (int i = 0; i < n; i++) {
        inp[2 * i] += vm[i];
    }

    // Now, we do the FFT:
    fft(inp, out, true);

    for (int i = 0; i < n; i++) {
        double f = i / (n * delta);
        amp.push_back(out[2 * i]);
        freq.push_back(f);
    }

    std::vector<double>::iterator result;
    int distance = 0;
    result = std::max_element(amp.begin(), amp.end() - n / 2);
    distance = std::distance(amp.begin(), result);

    fftPower = pow(amp[distance] / (sim.T / sim.dt), 2);
    fftFreq = freq[distance];

}

/*****************************************************************************
 * CORRELATION
 *****************************************************************************/

double Corr::correlation(deque<int> list1, deque<int> list2, double dt, double sig) {
    double res = 0;
    if (list1.size() >= 20 and list2.size() >= 20) {
        for (int i = 0; i < list1.size(); i++) {
            int upper = 0;
            int lower = 0;
            for (int k = 0; k < list2.size(); k++) {
                if (list2[k] <= list1[i]) {
                    lower = k;
                }
            }
            upper = lower + 1;

            double d1 = abs(list1[i] - list2[lower]);
            double d2 = abs(list1[i] - list2[upper]);
            double distance = min(d1, d2);
            double val = exp(-pow(distance * dt, 2.0) / (2 * pow(sig, 2)));
            res += val;
        }
        res /= list1.size();
    }

    return res;
}

double Corr::avgCorrelation(int N1, int N2, deque<int> *spikeTimesCor, double sig, double dt) {
    double t = 0;
    for (int i = N1; i < N2 - 1; i++) {
        t += correlation(spikeTimesCor[i], spikeTimesCor[i + 1], dt, sig);

    }
    return t / pow((N2 - N1), 2);
}

double Corr::computeCorrelation(deque<int> *spikeTimes, double dt) {
    return avgCorrelation(sim.NI - 100, sim.NI, spikeTimes, 3, dt);
}

double gaussian(double x, double mu, double sig, double dt) {
    if (abs(x - mu) / sig > 10 / dt) {
        return 0;
    }
    else {
        return exp(-pow(x - mu, 2) / (2 * pow(sig, 2)));
    }
}

double multigaussian(deque<int> list1, int x, double sig, double dt) {
    double max = 0;
    for (int i = 0; i < list1.size(); i++) {
        if (list1[i] * dt > 10 + x * dt) {
            break;
        }
        double val = gaussian(x, list1[i], sig, dt);
        if (val > max) {
            max = val;
        }
    }
    return max;
}

double Corr::correlationPairwise(deque<int> list1, deque<int> list2, double sig, double dt) {
    if (list1.size() > 0 and list2.size() > 0) {
        double sumG = 0;
        double gcache = 0;
        for (int i = 0; i < list2.size(); i++) { //int(list2.size())- 0
            gcache = multigaussian(list1, list2[i], sig, dt);
            if (list2[i] > (sim.NI - 100) / dt and gcache > 0) {
                sumG += gcache;
            }
        }
        return sumG /= list2.size();
    }
    else {
        return 0;
    }
}

void pl(bool CONSOLE, int line) {
    if(CONSOLE){
        cout << line << endl;
    }
}