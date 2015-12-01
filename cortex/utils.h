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
    bool SPINDLE_LOWPASS;
    bool COMPUTE_PLAST;
    bool MIN_PLAST_THRESH;
    bool GLOB;
    bool CORRELATION;
    bool FOURIER;
    bool SPIKES;
    bool CONSOLE;
    bool SOFT;
    int d1, d2, d3;
    int T;
    int T1;
    int T2;
    int T3;
    int before;
    int after;
    double dt;
    int N;
    int NI;
    int NE;
    int stimulation;
    std::string root;
    std::string computer;
    std::string directory;
    std::string ext;
    std::string path;

    double Tsig;
    double tau_I;
    double tau_syn;
    double GammaII;

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
    double TIMeanIN;

    void initData();
    void initDuration();
    bool saveTime(int t);

};

class Util {
public:
    Simulation sim;
    int writedata(string name ,vector<double> towrite);
    int writedataint(string name ,vector<int> towrite);


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

    void plasticity(double *burstTh, double *nonbursting, int t);
    void plasticityLocal(double *burstTh, double *nonbursting, int t);
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


double getSum( double *tosum, int N );
double getSumB( double *tosum, int N1, int N2 );
double getSum( int *tosum, int N );
double getAvg( double *tosum, int N );
double getAvgB( double *tosum, int N1, int N2 );
double getAvg( int *tosum, int N );
double getAvg( double **tosum, int N );
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


#endif /* defined(__cortex__utils__) */

