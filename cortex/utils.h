//
//  utils.h
//  cortex
//
//  Created by GP1514 Pernelle on 22/09/2015.
//  Copyright (c) 2015 GP1514 Pernelle. All rights reserved.
//

#ifndef __cortex__utils__
#define __cortex__utils__

#include "libs.h"

using namespace std;
typedef std::vector<int>::iterator iter;

// function template declarations

class Simulation
{
public:
    string PLAST_RULE;
    bool SPINDLE_LOWPASS;
    bool COMPUTE_PLAST;
    bool CLUSTER;
    bool localWII;
    bool MIN_PLAST_THRESH;
    bool GLOB;
    bool BOTH; // input to both subnetworks
    bool CORRELATION;
    bool FOURIER;
    bool SPIKES;
    bool CONSOLE;
    bool SOFT;
    bool RESONANCE;
    bool DEBUG;
    int d1, d2, d3;
    int T;
    int T1;
    int T2;
    int T3;
    int before;
    int after;
    double dt;
    int N;
    double r;
    int NI;
    int NE;
    int stimulation;
    int sharedWII;
    int sharedG;
    int sharedGTemp;
    std::string root;
    std::string computer;
    std::string directory;
    std::string csv_directory;
    std::string ext;
    std::string path;
    std::string csv_path;
    std::string model;
    std::string simName;

    double TsigI;
    double TsigE;
    double tau_I;
    double tau_syn;
    double GammaII;

    double LTD; //only used for writing data name
    double LTP; //only used for writing data name

    double GammaIE;
    double GammaEE ;
    double GammaEI ;

    double TC_Tsig ;
    double TC_tau_I;
    double TC_tau_syn;

    double C_Tsig;
    double C_tau_I;
    double C_tau_syn;
    int tauv;

    double gamma_c;
    double TImean;
    double TEmean;
    double TIMeanIN;
    double TEMeanIN;

    int nbInClusters;

    int nbOfGapJunctions;


    void initData();
    void initDuration();
    bool saveTime(int t);

};

class Util {
public:
    Simulation sim;
    //    int writedata(string name ,vector<double> towrite);
    //    int writedataint(string name ,vector<int> towrite);

    template <class T>
    int writedata(string name, vector<T> towrite) {
        /*
         * WRITE 1D array to file
         */

        stringstream sstm;
        sstm = makeName(name);
        //    //test if path exists
        //    fileExists(sim.path.c_str());
        const string path_x = sim.path + sstm.str() + sim.ext;
        ofstream out_x(path_x, ios::out | ios::binary);
        std::ostream_iterator<T> output_iterator_x(out_x, "\n");
        std::copy(towrite.begin(), towrite.end(), output_iterator_x);
        //    fileExists(path_x.c_str());
        return 0;
    }


    int writemap(string name, double ** (map))
    {
        /*
         * Write 2D array to file
         */
        stringstream sstm;
        sstm = makeName(name);
        //    //test if path exists
        //    fileExists(sim.path.c_str());
        const string path_x = sim.path + sstm.str() + sim.ext;
        std::fstream os(path_x, ios::out | ios::binary);

        for (int i = 0; i < sim.NI; ++i)
        {
            for (int j = 0; j < sim.NI; ++j)
            {
                os << map[i][j]<<" ";
            }
            os<<"\n";
        }
        return 0;
    }

    stringstream makeName(string name) {
        int glob = sim.GLOB * 1;
        stringstream sstm;
        sstm << sim.simName << name << "_g-" << sim.gamma_c << "_TImean-" << (sim.TImean) << "_T-" << (sim.T * sim.dt) << "_Glob-" <<
             glob;
        sstm << "_dt-" << sim.dt << "_N-" << sim.N << "_r-" << sim.r << "_S-" << sim.stimulation << "_WII-" << sim.GammaII;
        if (sim.LTD) sstm << "_LTD-" << sim.LTD;
        if (sim.LTP) sstm << "_LTP-" << sim.LTP;
        sstm << "_model-"<< sim.model;
        sstm << "_sG-"<< sim.sharedG << "_sWII-" << sim.sharedWII << "_tauv-" << sim.tauv << "_both-" << sim.BOTH*1 ;
        sstm << "_plast_" << sim.COMPUTE_PLAST*1;

        return sstm;
    }

};

class MovingAverage {
public:
    double counter;
    double meanG;
    Simulation sim;

//    MovingAverage();
    MovingAverage(vector<double> outputVec) : outputVec(outputVec) {
        counter = 0;
        meanG = 0;
    }

    vector<double> outputVec;
    void compute(double instantMeanG, int t, int T);
};


class Plasticity
{
public:
    double FACT;
    double tau_lowsp;

    double A_gapD;
    double th_lowsp;
    double th_q;
    double tau_q;
    double A_gapP;
    Simulation sim;


    double Vgap;
    double VgapIN;

    double** VgapLocal;
    double** allowedConnections;
    double** WIILocal;

    // Synaptic weights
    //
    double WII;
    double WIE;
    double WEE;
    double WEI;


    void initData();
    void initConnections();
    void plasticity(double *burstTh, bool *nonbursting, int t);
    void plasticityLocal(double *burstTh, bool *nonbursting, int t);
} ;

class Fourier{
public:
    vector<double> freq;
    vector<double> amp;
    double fftPower;
    double fftFreq;
    Simulation sim;

    void fft( std::vector<double>& inp, std::vector<double>& out, bool forward);
    void computeFFT(vector<double> vm);
};

template <class T>
double getSum(T *tosum, int N, int N2=0) {
    double sum = 0;
    if (N2 == 0) {
        for (int i = 0; i < N; i++) {
            sum += (double)tosum[i];
        }
    }
    else {
        for (int i = N; i < N2; i++) {
            sum += (double)tosum[i];
        }
    }
    return (double)sum;
}

template <class T>
double getAvg(T *tosum, int N, int N2=0) {
    if (N2>0) return (float) getSum(tosum, N, N2) / (N2-N);
    else return (float) getSum(tosum, N) / N;
}


double getAvg2D( double **tosum, int N );
void fileExists(const char* pathname);

class Corr {
public:
    Simulation sim;
    double computeCorrelation(deque<int>* spikeTimesCor, double dt);
    double avgCorrelation(int N1, int N2, deque<int>* spikeTimesCor, double sig, double dt);
    double correlation(deque<int> list1, deque<int> list2, double dt, double sig);
    double correlationPairwise(deque<int> list1, deque<int> list2, double sig, double dt);
};

double gaussian(double x, double mu, double sig, double dt);
double multigaussian (deque<int> listSpike, int x, double sig, double dt);

void pl(bool CONSOLE, int line);


#endif /* defined(__cortex__utils__) */

