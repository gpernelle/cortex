//
//  main.cpp
//  nn
//
//  Created by GP1514 Pernelle on 02/08/15.
//  Copyright (c) 2015 GP1514 Pernelle. All rights reserved.
//

#include "libs.h"
#include "utils.h"


using namespace std;
typedef std::vector<int>::iterator iter;

time_t starttime, endtime, previoustime;


// Initialization
//const int N = 750;



int main(int argc, const char * argv[])
{
    //init data
    //
    Util util;

    Plasticity plast;
    double tau_q;
    double th_q;
    double LTP;
    double LTD;
    double f = 0; //frequence for resonance computation
    dcomp res_val = 0; //resonance =  spike sum weighted with external oscillatory phase
    double cosVal = 0;
    int b = 2; // for TRN eq
    dcomp im =-1;
    string namecsv;
    Simulation *sim1;
    sim1 = &util.sim;
    pl(sim1->DEBUG, __LINE__);
    sim1->initData();
    for (int i = 1; i < argc; i++) {
        /* We iterate over argv[] to get the parameters stored inside.
        * Note that we're starting on 1 because we don't need to know the
        * path of the program, which is stored in argv[0]
        */
        if (i + 1 != argc) { // Check that we haven't finished parsing already
            if (!strcmp(argv[i], "-T")) {
                // We know the next argument *should* be the filename:
                sim1->T = atoi(argv[i + 1]);
            } else if (!strcmp(argv[i], "-G")) {
                sim1->gamma_c = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-N")) {
                sim1->N = atoi(argv[i + 1]);
            } else if (!strcmp(argv[i], "-r")) {
                sim1->r = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-ext")) {
                sim1->ext = argv[i + 1];
            } else if (!strcmp(argv[i], "-d1")) {
                sim1->d1 = atoi(argv[i + 1]) ;
            } else if (!strcmp(argv[i], "-d2")) {
                sim1->d2 = atoi(argv[i + 1]) ;
            } else if (!strcmp(argv[i], "-d3")) {
                sim1->d3 = atoi(argv[i + 1]) ;
            } else if (!strcmp(argv[i], "-before")) {
                sim1->before = atoi(argv[i + 1]);
            } else if (!strcmp(argv[i], "-after")) {
                sim1->after = atoi(argv[i + 1]);
            } else if (!strcmp(argv[i], "-S")) {
                sim1->stimulation = atoi(argv[i + 1]);
            } else if (!strcmp(argv[i], "-sG")) {
                sim1->sharedG = atoi(argv[i + 1]);
            } else if (!strcmp(argv[i], "-sWII")) {
                sim1->sharedWII = atoi(argv[i + 1]);
            } else if (!strcmp(argv[i], "-s")) {
                sim1->TsigI = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-tauv")) {
                sim1->tauv = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-WII")) {
                sim1->GammaII = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-tq")) {
                tau_q = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-thq")) {
                th_q = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-plast")) {
                sim1->COMPUTE_PLAST = (1==atoi(argv[i + 1])) ;
            } else if (!strcmp(argv[i], "-both")) {
                sim1->BOTH = (1==atoi(argv[i + 1])) ;
            } else if (!strcmp(argv[i], "-LTP")) {
                LTP = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-simName")) {
                sim1->simName = string(argv[i + 1]);
            } else if (!strcmp(argv[i], "-LTD")) {
                LTD = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-output")) {
                namecsv = string(argv[i + 1]);
                sim1->CONSOLE = true;
                sim1->COMPUTE_PLAST = false;
                sim1->SPIKES = false;
                sim1->FOURIER = true;

            } else if (!strcmp(argv[i], "-f")) {
                f = pow(10,atof(argv[i + 1]));
                sim1->RESONANCE = true;
            } else if (!strcmp(argv[i], "-model")) {
                sim1->model = string(argv[i + 1]);
            } else if (!strcmp(argv[i], "-directory")) {
                sim1->directory = string(argv[i + 1]);
            } else if (!strcmp(argv[i], "-global")) {
                sim1->GLOB = (atoi(argv[i + 1])==1);
            }

        }
    }
    sim1->NE = int(sim1->r*sim1->N);
    sim1->NI = sim1->N - sim1->NE;
    sim1->nbInClusters = sim1->NI/2;

    sim1->initDuration();
    vector<double> g {};
    MovingAverage mvgamma(g);
    MovingAverage mvgammaN1(g);
    MovingAverage mvgammaN2(g);
    MovingAverage mvgammaNshared(g);
    Fourier FFT;
    Fourier FFT1;
    Fourier FFT2;
    Corr correlation;
    pl(sim1->DEBUG, __LINE__);
    const int N = sim1->N;
    double dt = sim1->dt;
    int T = sim1->T;

    double* v = new double[N]{};

    double* u = new double[N]{0};
    double* t_rest = new double[N]{0};
    double* Iback = new double[N]{0};
    double* Igap = new double[N]{0};
    double* Ichem = new double[N]{0};
    double* Ieff = new double[N]{0};
    double* LowSp = new double[N]{0};
    double* I = new double[N]{0};
    double* p = new double[N]{0};
    double* q = new double[N]{0};
    double synSpikes = 0;
    bool* nonbursting = new bool[N]{false};
    bool* passive = new bool[N]{true};
    double meanG = 0;
    double meanRON_I = 0;
    int counter = 0;
    double I_TC = 0;
    pl(sim1->DEBUG, __LINE__);
    double Vsum;
    int NbSpikesI = 0;
    int NbSpikesE = 0;
    int NewNbSpikesE = 0; // temp var to get number of spikes of Exc neurons at t-1
    int NewNbSpikesI = 0; // temp var to get number of spikes of Inh neurons at t-1
    bool* vv = new bool[N]{false};
    vector<int> spikes_idx;
    vector<int> spikes_idy;
    vector<int> spikes_idx_tc;
    vector<int> spikes_idy_tc;
    vector<double> corrVect;
    vector<double> current1;
    vector<double> ssp;
    vector<double> ssp1;
    vector<double> current2;
    vector<double> ssp2;
    vector<double> current3;
    vector<double> ssp3;
    vector<double> vm;
    vector<double> vm1;
    vector<double> vm2;
    vector<double> voltage;  // for DEBUG
    vector<double> adaptation; // for DEBUG
    vector<double> pm;
    vector<double> qm;
    vector<double> lm;
    vector<double> stimulation;
    vector<double> RON_I;
    vector<double> RON_V;
    vector<double> vVect;
    vector<int>* spikeTimes = new vector<int>[N];
    vector<int>* spikeTimes_tc = new vector<int>[N];
    deque<int>* spikeTimesCor = new deque<int>[N];
    double* noise = new double[N]{0};
    double meanBurst= 0 ;
    double meanBurst1= 0 ;
    double meanBurst2= 0 ;
    double meanSpike = 0;
    double meanSpike1 = 0;
    double meanSpike2 = 0;
    double meanSpikeNonBurst = 0;
    double meanSpikeNonBurst1 = 0;
    double meanSpikeNonBurst2 = 0;

    double alpha = dt / (tau_q + dt);
    pl(sim1->DEBUG, __LINE__);
    // TC cortex COMM
    double tau_syn_tc = 20;
    double WEE = 1;
    double WEI = 0.0; // from TC to cortex
    double WIE = -8; // from cortex to TC
    double Rm=4;
    double Cm=1;

    // load simulation data
    //
    mvgamma.sim = *sim1;
    mvgammaN1.sim = *sim1;
    mvgammaN2.sim = *sim1;
    mvgammaNshared.sim = *sim1;
    FFT.sim = *sim1;
    FFT1.sim = *sim1;
    FFT2.sim = *sim1;
    plast.sim = *sim1;
    correlation.sim = *sim1;
    plast.initData();
    plast.tau_q = tau_q;
    plast.th_q = th_q;
    if (LTD) {
        plast.A_gapD = LTD;
        sim1->LTD = LTD;
    }
    if (LTP) {
        plast.A_gapP = LTP;
        sim1->LTP = LTP;
    }
    pl(sim1->DEBUG, __LINE__);
    // random number generation
    //
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<> dist(0.0, 1.0);


    if (!sim1->CONSOLE and argc==1) {
        std::cout << "Neural Network Initialization\n";
        cout << "*****************************" << endl;
        cout << "FACT: "<< plast.FACT << endl;
        cout << "LTD: "<< plast.A_gapD << endl;
        cout << "VgapIn: "<< plast.VgapIN << endl;
        cout << "Vgap: "<< plast.Vgap << endl;
        cout << "tau_lowsp: "<< plast.tau_lowsp << endl;
        cout << "WII: "<< plast.WII << endl;
        cout << "Simulation duration: " << T << endl;
        cout << "T1: "<< sim1->T1 << "\tT2: " << sim1->T2 << "\tT3: " << sim1->T3 << endl;
        cout << "d1: "<< sim1->d1 << "\td2: " << sim1->d2 << "\td3: " << sim1->d3 << endl;
        cout << "t1: "<< sim1->T1 - sim1->before/dt << "\tt2: " << sim1->T1 + sim1->after/dt << "\tt3: " << sim1->T2 - sim1->before/dt << "\tt4: " << sim1->T2+ sim1->after/dt << "\tt5: " << sim1->T3-sim1->before/dt << endl;

    }
    pl(sim1->DEBUG, __LINE__);
    std::fill(v,v+N,-60); // init v to -60mV

    // Init connection matrix
    //`
    if (!sim1->GLOB) {
        for (int i = 0; i < sim1->NI; ++i) {
            plast.VgapLocal[i] = new double[sim1->NI];
            plast.WIILocal[i] = new double[sim1->NI];
        }
        for (int m = 0; m<sim1->NI; m++) {
                v[m] = -50+(-50*(m<sim1->nbInClusters));
                vv[m] = m<sim1->nbInClusters;
            for (int n=0; n<m; n++) {
                plast.VgapLocal[m][n] = max(0.001,dist(e2)*(m!=n))*plast.Vgap;
                plast.VgapLocal[n][m] = plast.VgapLocal[m][n];
                plast.WIILocal[m][n] = plast.WII*(m!=n);
                plast.WIILocal[n][m] = plast.WII*(m!=n);
            }
        }
        plast.WEE = 0;
        plast.WIE = 0;
        plast.WEI = 0;

        //----------------
        // CLUSTERING
        if (sim1->CLUSTER){
            for (int m = 0; m<sim1->NI; m++) {
                for (int n=0; n<sim1->NI; n++) {
                    plast.WIILocal[m][n] *= plast.allowedConnections[m][n] ;
                    plast.VgapLocal[m][n] *= plast.allowedConnections[m][n] ;
                    }
                }
            }
        }
        util.writemap("GAP0", plast.VgapLocal);
        util.writemap("WII", plast.WIILocal);

        //------------------


    // start counter time
    //
    time (&starttime);
    time (&previoustime);
    pl(sim1->DEBUG, __LINE__);
    /***************************************************
     * INTEGRATION OVER TIME
     ***************************************************/
    for (int t=0; t< T ; t++) {
        /***************************************************
         * SET STIMULATION
         ***************************************************/
        if (!sim1->CONSOLE) {
            if(t<sim1->T1){ // first phase of the simulation
                sim1->TImean = sim1->TIMeanIN;
                sim1->TEmean = sim1->TEMeanIN;
            }
            else if (t>sim1->T1 and t<sim1->T2) { // second phase of the simulation
                sim1->TImean =   sim1->TIMeanIN + sim1->stimulation;
                sim1->TEmean =   sim1->TEMeanIN + sim1->stimulation;
            }
            else if (t>=sim1->T2) { // last phase of the simulation until T3
                sim1->TImean = sim1->TIMeanIN;
                sim1->TEmean = sim1->TEMeanIN;
            }
        }
        else {
            sim1->TImean = sim1->TIMeanIN + sim1->stimulation;
            sim1->TEmean = sim1->TEMeanIN + sim1->stimulation;
        }

        if (t==0) {
            sim1->sharedGTemp = sim1->sharedG;
            sim1->sharedG = 0;
            sim1->COMPUTE_PLAST = 1;
            plast.initConnections();
        }


        if (t==30000) {
            sim1->sharedG = sim1->sharedGTemp;
            sim1->COMPUTE_PLAST = 0;
            plast.initConnections();
        }


        /***************************************************
         * PRINT PROGRESS
         ***************************************************/
        if (!sim1->CONSOLE and argc == 1) {
            time (&endtime);
            double dif = difftime (endtime,previoustime);
            if(t % int(9*T/100) == 0 ) {
//                cout << t/3*100/T << " % \t spikes:" << getAvg<bool>(vv, sim1->NI) << "\t V:" << getAvg<double>(v, sim1->NI, sim1->NE) << endl;
            }
            if(dif > 2) {
                double dif2 = difftime (endtime,starttime);
                previoustime = endtime;
                if (sim1->GLOB) {
                    printf ("-- Elapsed time : %.2lf seconds.\t %.6lf \t %.6lf \n",
                            dif2, plast.Vgap*sim1->NI, meanRON_I );
                }
                else {
                    printf ("-- Elapsed time : %.2lf seconds.\t %.6lf \t %.6lf \n",
                            dif2, getAvg2D(plast.VgapLocal, sim1->NI), meanRON_I );
                }
            }
        }

        Vsum = getSum(v, sim1->NI);
        NewNbSpikesI = 0;
        NewNbSpikesE = 0;
        //
        /***************************************************
         * NEURON LOOP
         ***************************************************/

        /*
         * Equations for the neurons:
         * cortex:
         * RS cells: excitatory neurons
         * FS cells: inhibitory neurons
         *
         * FS (p299):
         *      20 v' = (v+55)(v+40) - u +I,
         *      u' = 0.2(U(v) - u )
         *      if v >= 25, then v <- -45mV
         *
         *      Slow non linear nullcline U(v):
         *      if v < vb, then U(v) = 0
         *      else U(v) = 0.025(v-vb)**3, with vb = -55
         *
         * RS pyramidal neuron (p283):
         *      100v' = 0.7(v+60)(v+40) -u + I,
         *      u' = 0.03(-2(v+60) -u)
         *      if v>=35, then v <- -50, u <- u+100
         *
         *
         * FS (nov 2003) publication:
         *      v' = 0.04v**2 + 5v + 140 -u + I
         *      u' = a (bv -u)
         *      if v>30, then v <- c and u <- u+d
         *
         *      FS: a = 0.1, b = 0.2, c = -65, d = 2
         */

        //        #pragma omp parallel for num_threads(NUM_THREADS)
        if (sim1-> RESONANCE) {
            cosVal = 100.0 * cos(2.0 * M_PI * f * (t/1000.0 * dt));
//            if (t < 100) cout << cosVal << "\t" << 2 * M_PI * f * (t/1000.0 * dt) << endl;
        }
        pl(sim1->DEBUG and t<=3 , __LINE__);
        for(int i=0; i < sim1->N ; i++){
            /*
             * 0 < i < NI : FS inhibitbory cells
             * NI < i < NE+NI : RS excitatory cells
             */
            if(i<sim1->NI) {
                noise[i] = dist(e2);
                Iback[i] = Iback[i] + dt/(sim1->tau_I*1.0) * (-Iback[i] + noise[i]);
                Ieff[i] = Iback[i] / sqrt(1/(2*(sim1->tau_I/dt))) * sim1->TsigI + sim1->TImean*((i>sim1->nbInClusters)*(t>300)*(sim1->BOTH) + (i<=sim1->nbInClusters) );


                // sum
                if (sim1->GLOB) {
                    Igap[i]= plast.Vgap*(Vsum -  sim1->NI * v[i]);
                    Ichem[i] = Ichem[i] + dt/(sim1->tau_syn*1.) * (-Ichem[i]
                                                                   + plast.WII * (NbSpikesI - ( v[i] > 25.0))
                                                                   + plast.WEI * NbSpikesE);
                }
                else {
                    Igap[i] = 0;
                    synSpikes = 0;
                    for (int k = 0; k<sim1->NI; k++) {
                        Igap[i] += (plast.VgapLocal[i][k]) * (v[k] - v[i]);

                        if (sim1->CLUSTER and sim1->localWII) {

                            synSpikes += plast.WIILocal[i][k] * ((vv[k])) * (i!=k) * (int(i/sim1->nbInClusters)==int(k/sim1->nbInClusters)) * plast.allowedConnections[i][k] ;
                        }

                    }
                    if (sim1->CLUSTER and sim1->localWII) {
                        // add a different time constants for neurons in the second cluster
                        Ichem[i] += dt/(sim1->tau_syn*(1.)) * (-Ichem[i]
                                                             + synSpikes
                                                             + plast.WEI * NbSpikesE);

                    } else {
                        Ichem[i] += + dt/(sim1->tau_syn*1.) * (-Ichem[i]
                                                                       + plast.WII * (NbSpikesI - ( v[i] > 25.0))
                                                                       + plast.WEI * NbSpikesE);
                    }

                }
                if (sim1->RESONANCE) {
                    Ieff[i] = Iback[i] / sqrt(1/(2*(sim1->tau_I/dt))) * sim1->TsigI + sim1->TImean + cosVal;
                    I[i] = Ieff[i];
                }
                else {
                    I[i] = Ieff[i] + Ichem[i] + Igap[i];
                }


                if (sim1->model == "gp-izh"){
                    v[i] += dt / 15 * ((v[i] + 60) * (v[i] + 50) - 20*u[i] + 8 * I[i]);
                    u[i] += dt * 0.044 * ((v[i] + 55) - u[i]);
                    vv[i] = v[i] > 25.0;
                    if (vv[i]) {
                        v[i] = -40;
                        u[i] += 50;
                    }
                } else if (sim1->model == "gp-izh-subnetworks"){
                    v[i] += dt / (15+(sim1->tauv-15)*(i>sim1->nbInClusters)) * ((v[i] + 60) * (v[i] + 50) - 20*u[i] + 8 * I[i]);
                    u[i] += dt * 0.044 * ((v[i] + 55) - u[i]);
                    vv[i] = v[i] > 25.0;
                    if (vv[i]) {
                        v[i] = -40;
                        u[i] += 50;
                    }
                }

                /***************************************************
                 * SAVE SPIKES
                 ***************************************************/
                if(vv[i]) {

                    NewNbSpikesI++;

                    if (sim1->CLUSTER) {
                        spikes_idx.push_back(t);
                        spikes_idy.push_back(i);
                    }

                    if ( !sim1->CONSOLE and sim1->SPIKES and  sim1->saveTime(t))
                    {
                        // save spikes only close to transitions
                        if (!sim1->CLUSTER) {
                            spikes_idx.push_back(t*dt);
                            spikes_idy.push_back(i);
                        }

                        if (t>200/dt) {
                            spikeTimes[i].push_back(t);
                        }
                    }

                    // FIFO Correlation
                    //
                    if(sim1->CORRELATION) {
                        if (t>200/dt) {
                            spikeTimesCor[i].push_back(t);
                            if (spikeTimesCor[i].size()>41) {
                                spikeTimesCor[i].pop_front();
                            }
                        }
                    }
                }


                /***************************************************
                 * BURST COUNT FOR PLASTICITY
                 ***************************************************/
                LowSp[i] = LowSp[i] + dt/plast.tau_lowsp * (vv[i] * plast.tau_lowsp / dt - LowSp[i]);

                if (sim1->SPINDLE_LOWPASS) {
                    q[i] = alpha * LowSp[i] + (1-alpha)*q[i];
                    p[i] = (q[i] > plast.th_q)*1.0;
                    nonbursting[i] = vv[i]*(q[i] <= plast.th_q);

                } else {
                    p[i] = (LowSp[i] > plast.th_lowsp)*1.0;
                    nonbursting[i] = vv[i]*(LowSp[i] <= plast.th_lowsp);
                }
            }
            else {
                /*
                * RS pyramidal neuron (p283):
                *      100v' = 0.7(v+60)(v+40) -u + I,
                *      u' = 0.03(-2(v+60) -u)
                *      if v>=35, then v <- -50, u <- u+100
                */
                noise[i] = dist(e2);
                Iback[i] = Iback[i] + dt/(sim1->tau_I*1.0) * (-Iback[i] + noise[i]);
                Ieff[i] = Iback[i] / sqrt(1/(2*(sim1->tau_I/dt))) * sim1->TsigE + sim1->TEmean;
                Ichem[i] = Ichem[i] + dt/(sim1->C_tau_syn*1.) * (-Ichem[i]
                                                                  + plast.WEE * (NbSpikesE - ( v[i] > 1.4))
                                                                  + plast.WIE * NbSpikesI);

                I[i] = Ieff[i] + Ichem[i];
                if (t>t_rest[i]) v[i] = v[i] + (-v[i] + I[i]*0.01) / 10 * dt;
                else v[i] = 0;
                if (v[i] >= 1) {
                    v[i] += 0.5;
                    t_rest[i] = t+4;
                }
                vv[i] = v[i] > 1.4;



                /***************************************************
                 * SAVE SPIKES
                 ***************************************************/
                if(vv[i]) {
                    NewNbSpikesE++;
                    v[i] = -50;
                    u[i] = u[i] + 100;
                    if ( !sim1->CONSOLE and sim1->SPIKES and  sim1->saveTime(t))
                    {
                        // save spikes only close to transitions
                        spikes_idx.push_back(t*dt);
                        spikes_idy.push_back(i);
                        if (t>200/dt) {
                            spikeTimes_tc[i].push_back(t);
                        }
                    }
                }

            }
        }

        // compute some stats
        //
        if (sim1->RESONANCE) {
            res_val += getSum(vv,sim1->NI)*exp(2.0*M_PI*im*f*(t/1000.0*dt));
        }


        NbSpikesI = NewNbSpikesI;
        NbSpikesE = NewNbSpikesE;

        meanBurst += getAvg<double>(p, sim1->NI);
        meanBurst1 += getAvg<double>(p, int((sim1->NI-sim1->sharedG)/2));
        meanBurst2 += getAvg<double>(p, int((sim1->NI+sim1->sharedG)/2), sim1->NI);

        meanSpikeNonBurst += getAvg<bool>(nonbursting, sim1->NI);
        meanSpikeNonBurst1 += getAvg<bool>(nonbursting,  \
		int((sim1->NI-sim1->sharedG)/2));
        meanSpikeNonBurst2 += getAvg<bool>(nonbursting, \
		int((sim1->NI+sim1->sharedG)/2), sim1->NI);

        meanSpike += getAvg<bool>(vv, sim1->NI);  // only for inhibitory neurons here
        meanSpike1 += getAvg<bool>(vv, int((sim1->NI-sim1->sharedG)/2));
        meanSpike2 += getAvg<bool>(vv, int((sim1->NI+sim1->sharedG)/2), sim1->NI);


        if (sim1->FOURIER){
            vm.push_back(getAvg<double>(v, sim1->NI+1, sim1->N));
            vm1.push_back(getAvg<double>(v, 0, int((sim1->NI-sim1->sharedG)/2)));
            vm2.push_back(getAvg<double>(v, int((sim1->NI+sim1->sharedG)/2), sim1->NI));
        }
        if (sim1->CLUSTER){
            current1.push_back(getAvg<double>(I, sim1->nbInClusters, 0));
            current2.push_back(getAvg<double>(I, sim1->nbInClusters, sim1->NI));
        }
        if (sim1->DEBUG) {
            voltage.push_back(getAvg<double>(v, sim1->NI));
            adaptation.push_back(getAvg<double>(u, sim1->NI));
        }
        pm.push_back(getAvg<double>(p, sim1->NI));
        qm.push_back(getAvg<double>(q, sim1->NI));
        lm.push_back(getAvg<double>(LowSp, sim1->NI));

        /***************************************************
         * PLASTICITY
         *
         ***************************************************/
        pl(sim1->DEBUG and t<3, __LINE__);
        if (!sim1->CONSOLE and sim1->COMPUTE_PLAST) {
            if(t % int(T/1000) == 0 ){
                stimulation.push_back(sim1->TImean);
            }
            if (sim1->GLOB) {
                // compute global plasticity
                //
                if (sim1->PLAST_RULE == "nonbursting") plast.plasticity(p, nonbursting, t);
                else if (sim1->PLAST_RULE == "spiking") plast.plasticity(p, vv, t);
                else if (sim1->PLAST_RULE == "passive") plast.plasticity(p, passive, t);

                mvgamma.compute(plast.Vgap,t, T);  //moving average
            }
            else {
                // compute local plasticity
                //
                if (sim1->PLAST_RULE == "nonbursting") plast.plasticityLocal(p, nonbursting, t);
                else if (sim1->PLAST_RULE == "spiking") plast.plasticityLocal(p, vv, t);
                else if (sim1->PLAST_RULE == "passive") plast.plasticityLocal(p, passive, t);
                double instantMeanG = 0;
                double instantMeanGN1 = 0;
                double instantMeanGN2 = 0;
                double instantMeanGshared = 0;
                for (int i = 0; i < sim1->NI; i++) {
                    for (int j = 0; j < i; j++) {
                        instantMeanG += plast.VgapLocal[i][j];
                    }
                }
                for (int i = 0; i < (sim1->NI/2 - sim1->sharedG); i++) {
                    for (int j = 0; j < (sim1->NI/2 - sim1->sharedG); j++) {
                        instantMeanGN1 += plast.VgapLocal[i][j]*1.0;
                    }
                }
                for (int i = (sim1->NI/2 + sim1->sharedG); i < sim1->NI; i++) {
                    for (int j = (sim1->NI/2 + sim1->sharedG); j < sim1->NI; j++) {
                        instantMeanGN2 += plast.VgapLocal[i][j];
                    }
                }
                for (int i = (sim1->NI/2 - sim1->sharedG); i < (sim1->NI/2 + sim1->sharedG); i++) {
                    for (int j = (sim1->NI/2 - sim1->sharedG); j < (sim1->NI/2 + sim1->sharedG); j++) {
                        instantMeanGshared += plast.VgapLocal[i][j];
                    }
                }
                instantMeanG /= (sim1->NI * (sim1->NI - 1) / 2);
                instantMeanGN1 /= pow(sim1->NI/2 - sim1->sharedG,2);
                instantMeanGN2 /= pow(sim1->NI/2 - sim1->sharedG,2);
                mvgamma.compute(instantMeanG*2, t, T); //moving average
                mvgammaN1.compute(instantMeanGN1, t, T); //moving average
                mvgammaN2.compute(instantMeanGN2, t, T); //moving average
                mvgammaNshared.compute(instantMeanGshared, t, T); //moving average
//                cout << "gamma\t" << instantMeanGN1 << endl;
            }
        }
        
        /***************************************************
         * CORRELATION
         ***************************************************/
        pl(sim1->DEBUG and t<3, __LINE__);
        if (sim1->CORRELATION and !sim1->CONSOLE){
            if(t % int(T/1000) == 0 and t>200/dt and N>100 ){
                corrVect.push_back(correlation.computeCorrelation(spikeTimesCor, dt));
            }
        }
        
        /***************************************************
         * SAVE SPIKE MEAN AND CURRENT AT TRANSITIONS
         ***************************************************/
        pl(sim1->DEBUG and t<3, __LINE__);
        if (!sim1->CONSOLE) {
            ssp.push_back(NbSpikesI+NbSpikesE);
            ssp1.push_back(NbSpikesE);

        }
    }
    pl(sim1->DEBUG, __LINE__);
    /***************************************************
     * END OF SIMULATION - SAVE DATA
     ***************************************************/
    
    // end time counter
    //
    if (!sim1->CONSOLE) {
        time (&endtime);
        double dif = difftime (endtime,starttime);
//        cout << "*****************************" << endl;
//        printf ("Elasped time is %.2lf seconds.", dif );
//        cout << " End of simulation: " << endl;
    }



    // SAVE DATA
    //
    if (!sim1->CONSOLE and !sim1->RESONANCE) {
//    if (1) {
        // SAVE DATA NORMAL MODE
        //
        util.writedata("gamma", mvgamma.outputVec);
        util.writedata("gammaN1", mvgammaN1.outputVec);
        util.writedata("gammaN2", mvgammaN2.outputVec);
//        cout <<"wirtestuff" << mvgammaN2.outputVec[300]<< endl;
        util.writedata("gammaNshared", mvgammaNshared.outputVec);

        util.writedata("stimulation", stimulation);
        util.writedata("correlation", corrVect);
        util.writedata("p", pm);
        util.writedata("q", qm);
        util.writedata("LowSp", lm);
        util.writedata("spike_x", spikes_idx);
        util.writedata("spike_y", spikes_idy);
//        util.writedata("spike_x_tc", spikes_idx_tc);
//        util.writedata("spike_y_tc", spikes_idy_tc);
        util.writedata("ssp", ssp);
        util.writedata("v", voltage);
        util.writedata("u", adaptation);

        if (sim1->CLUSTER) {
            util.writedata("current1", current1);
            util.writedata("current2", current2);
        }


//        util.writedata("resonance", abs(res_val));
        //cout << "resonance: " <<  abs(res_val) << "\t" << abs(res_val)/T/N/dt << endl;
        if (sim1->FOURIER) {
            util.writedata("vm", vm);
            util.writedata("vm1", vm1);
            util.writedata("vm2", vm2);
//            util.writedata("freq", FFT.freq);
//            util.writedata("amp", FFT.amp);
        }
        

//        util.writedata("current3", current3);
        
        util.writedata("sspE", ssp1);
//        util.writedata("ssp2", ssp2);
//        util.writedata("ssp3", ssp3);
        
//        util.writedata("RON_I", RON_I);
//        util.writedata("RON_V", RON_V);
//        util.writedata("V", vVect);

        //cout << " Data written in  " << util.sim.path << endl;

        if (sim1->GLOB) {
            cout << "gamma_c : " << plast.Vgap*sim1->NI <<"\tRON_I:\t" << meanRON_I << endl;
        }
        else {
            util.writemap("GAP", plast.VgapLocal);
//            util.writemap("WII", plast.WIILocal);
            //cout << "gamma_c avg : " << meanG/counter << endl;
        }
    }
    else if (sim1->CONSOLE) {
        // SAVE DATA CONSOLE MODE
        //
        if (sim1->FOURIER) {
            // compute FFT on Local Field Potential
            //
            FFT.computeFFT(vm);
            FFT1.computeFFT(vm1);
            FFT2.computeFFT(vm2);
        }
        
        double resultCorr = correlation.computeCorrelation(spikeTimesCor, dt);
        string path_csv =  sim1->csv_path + namecsv + ".csv";
	//cout << path_csv << endl;
        ofstream csvFile(path_csv, std::ofstream::out | std::ofstream::app);
        // ACTIVITY DIAG
        //csvFile << plast.Vgap*sim1->NI << ";" << sim1->TImean << ";" << resultCorr << ";" << meanSpike/T << ";" << meanSpikeNonBurst/T << ";" << meanBurst/T <<";"<< FFT.fftFreq<<";"<< FFT.fftPower<< endl;
        // ACTIVITY FOR SUBNETs
        csvFile << plast.Vgap*sim1->NI << ";" << sim1->TImean << ";" \
            	<< resultCorr \
            	<< ";" << meanBurst/T  << ";" << meanSpike/T << ";" << meanSpikeNonBurst/T \
            	<< ";" << meanBurst1/T  << ";" << meanSpike1/T << ";" << meanSpikeNonBurst1/T \
            	<< ";" << meanBurst2/T  << ";" << meanSpike2/T << ";" << meanSpikeNonBurst2/T \
		<< ";" << FFT.fftFreq <<";" << FFT.fftPower\
            	<< ";" << FFT1.fftFreq <<";" << FFT1.fftPower\
            	<< ";" << FFT2.fftFreq <<";" << FFT2.fftPower\
            	<< ";" << sim1->sharedGTemp << ";" << sim1->sharedWII << ";" << sim1->tauv << endl;
    }
    if (sim1->RESONANCE) {
        string path_csv =  sim1->root+sim1->computer+sim1->directory+"resonance.csv";
        ofstream csvFile(path_csv, std::ofstream::out | std::ofstream::app);
        csvFile << sim1->model<< ";" << f << ";" << sim1->stimulation+50.0 << ";" << abs(res_val) << ";" << endl;
        //cout << "Data written: " << f<< "\t" << abs(res_val) << endl ;

    }
    
    
    return 0;
}



