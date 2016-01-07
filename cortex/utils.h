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
    bool MIN_PLAST_THRESH;
    bool GLOB;
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
    std::string root;
    std::string computer;
    std::string directory;
    std::string ext;
    std::string path;
    std::string model;

    double Tsig;
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

    double gamma_c;
    double TImean;
    double TEmean;
    double TIMeanIN;
    double TEMeanIN;

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
        int glob = sim.GLOB * 1;
        stringstream sstm;
        sstm << name << "_g-" << sim.gamma_c << "_TImean-" << (sim.TImean) << "_T-" << (sim.T * sim.dt) << "_Glob-" <<
        glob;
        sstm << "_dt-" << sim.dt << "_N-" << sim.N << "_r-" << sim.r << "_S-" << sim.stimulation << "_WII-" << sim.GammaII;
        if (sim.LTD) sstm << "_LTD-" << sim.LTD;
        if (sim.LTP) sstm << "_LTP-" << sim.LTP;
        sstm << "_model-"<< sim.model;

        //    //test if path exists
        //    fileExists(sim.path.c_str());
        const string path_x = sim.path + sstm.str() + sim.ext;
        ofstream out_x(path_x, ios::out | ios::binary);
        std::ostream_iterator<T> output_iterator_x(out_x, "\n");
        std::copy(towrite.begin(), towrite.end(), output_iterator_x);
        //    fileExists(path_x.c_str());
        return 0;
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

    // Synaptic weights
    //
    double WII;
    double WIE;
    double WEE;
    double WEI;


    void initData();

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


//double getSum( double *tosum, int N );
//double getSumB( double *tosum, int N1, int N2 );
//double getSum( int *tosum, int N );
//double getAvg( double *tosum, int N );
//double getAvgB( double *tosum, int N1, int N2 );
//double getAvg( int *tosum, int N );
//template <class T> double getAvg(T *tosum, int N, int N2 = 0);
//template <class T> double getSum(T *tosum, int N, int N2 = 0);

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

