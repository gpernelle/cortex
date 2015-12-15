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
    string namecsv;
    Simulation *sim1;
    sim1 = &util.sim;

    sim1->initData();
//    cout << std::string(100, '*')<< endl;
//    cout << "BEGIN SIMULATION" << endl;
//    cout << std::string(100, '*')<< endl;
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
                sim1->NE = int(0*sim1->N);
                sim1->NI = sim1->N - sim1->NE;
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
            } else if (!strcmp(argv[i], "-s")) {
                sim1->Tsig = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-WII")) {
                sim1->GammaII = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-tq")) {
                tau_q = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-thq")) {
                th_q = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-LTP")) {
                LTP = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-LTD")) {
                LTD = atof(argv[i + 1]);
            } else if (!strcmp(argv[i], "-output")) {
                namecsv = string(argv[i + 1]);
                sim1->CONSOLE = true;
                sim1->COMPUTE_PLAST = false;
                sim1->SPIKES = false;
                sim1->FOURIER = true;

            }
//            cout << std::string(100, '*')<< endl;
//            cout <<  "Arguments read"<< endl;
//            cout << std::string(100, '*')<< endl;
        }
    }
    sim1->initDuration();
    vector<double> g {};
    MovingAverage mvgamma(g);
    Fourier FFT;
    Corr correlation;

    const int N = sim1->N;
    double dt = sim1->dt;
    int T = sim1->T;

    double* v = new double[N]{};

    double* u = new double[N]{0};
    double* Iback = new double[N]{0};
    double* Igap = new double[N]{0};
    double* Ichem = new double[N]{0};
    double* Ieff = new double[N]{0};
    double* LowSp = new double[N]{0};
    double* I = new double[N]{0};
    double* p = new double[N]{0};
    double* q = new double[N]{0};
    bool* nonbursting = new bool[N]{false};
    bool* passive = new bool[N]{true};
    double meanG = 0;
    double meanRON_I = 0;
    int counter = 0;
    double I_TC = 0;

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
    double meanSpike = 0;
    double meanSpikeNonBurst = 0;

    double alpha = dt / (tau_q + dt);

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
    FFT.sim = *sim1;
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

    // random number generation
    //
    random_device rd;
    mt19937 e2(rd());
    normal_distribution<> dist(0.0, 1.0);


    if (!sim1->CONSOLE) {
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

    std::fill(v,v+N,-60); // init v to -60mV

    // Init connection matrix
    //
    if (!sim1->GLOB) {
        for (int i = 0; i < sim1->NI; ++i) {
            plast.VgapLocal[i] = new double[sim1->NI];
        }
        for (int m = 0; m<sim1->NI; m++) {
            for (int n=0; n<sim1->NI; n++) {
                plast.VgapLocal[m][n] = plast.Vgap*1.0;
            }
        }
    }

    // start counter time
    //
    time (&starttime);
    time (&previoustime);

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
            }
            else if (t>sim1->T1 and t<sim1->T2) { // second phase of the simulation
                sim1->TImean =   sim1->TIMeanIN + sim1->stimulation;
            }
            else if (t>=sim1->T2) { // last phase of the simulation until T3
                sim1->TImean = sim1->TIMeanIN;
            }
        }
        else {
            sim1->TImean = sim1->TIMeanIN + sim1->stimulation;
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
         * CH neurons?
         *
         * FS (p299): 20 v' = (v+55)(v+40) - u +I,
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
         * FS (nov 2003) publications:
         *      v' = 0.04v**2 + 5v + 140 -u + I
         *      u' = a (bv -u)
         *      if v>30, then v <- c and u <- u+d
         *
         *      FS: a = 0.1, b = 0.2, c = -65, d = 2
         */

        //        #pragma omp parallel for num_threads(NUM_THREADS)

        for(int i=0; i < sim1->N ; i++){
            /*
             * 0 < i < NI : FS inhibitbory cells
             * NI < i < NE+NI : RS excitatory cells
             */
            if(i<=sim1->NI) {
                noise[i] = dist(e2);
                Iback[i] = Iback[i] + dt/(sim1->tau_I*1.0) * (-Iback[i] + noise[i]);
                Ieff[i] = Iback[i] / sqrt(1/(2*(sim1->tau_I/dt))) * sim1->Tsig + sim1->TImean;
                Ichem[i] = Ichem[i] + dt/(sim1->tau_syn*1.) * (-Ichem[i]
                                                               + plast.WII * (NbSpikesI - ( v[i] > 25.0))
                                                              + plast.WEI * NbSpikesE);

                // sum
                if (sim1->GLOB) {
                    Igap[i]= plast.Vgap*(Vsum -  sim1->NI * v[i]);
                }
                else {
                    Igap[i] = 0;

                    for (int k = 0; k<sim1->NI; k++) {
                        Igap[i] += (plast.VgapLocal[i][k]) * (v[k] - v[i]);
                    }
                }
                I[i] = Ieff[i] + Ichem[i] + Igap[i];
                // TRN EQUATIONS --------------------------------------
                //  v[i] = v[i] + dt * (1./40.) * (0.25 * (pow(v[i],2) + 110 * v[i] + 45*65) - u[i] + I[i]);  //TRN eq
                //  u[i] = u[i] + dt * 0.015 * ( b * (v[i] + 65) - u[i] );  // TRN eq
                // TRN EQUATIONS --------------------------------------

                // FS EQ --------------------------------------
//                v[i] += dt/20 * ((v[i]+55)*(v[i]+40) - u[i] + I[i]);
//                u[i] += dt * 0.2 * ( 0.025*pow(v[i]+55,3) * (v[i] > -55) - u[i] );
//                vv[i] = v[i] > 25.0;
                // --------------------------------------

                v[i] += dt/20 * ((v[i]+55)*(v[i]+40) - u[i] + 2.5*I[i]);
                u[i] += dt * 0.037 * ( 0.05*pow((v[i]+55),3) * (v[i] > -55) - u[i] );
                vv[i] = v[i] > 20.0;

                /***************************************************
                 * SAVE SPIKES
                 ***************************************************/
                if(vv[i]) {
                    NewNbSpikesI++;
//                    v[i] = -45; // FS
                    v[i] = -41;
                    u[i] += -20;
                    if ( !sim1->CONSOLE and sim1->SPIKES and  sim1->saveTime(t))
                    {
                        // save spikes only close to transitions
                        spikes_idx.push_back(t*dt);
                        spikes_idy.push_back(i);
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
                Ieff[i] = Iback[i] / sqrt(1/(2*(sim1->tau_I/dt))) * sim1->C_Tsig + sim1->TEmean;
                Ichem[i] = Ichem[i] + dt/(sim1->C_tau_syn*1.) * (-Ichem[i]
                                                                  + plast.WEE * (NbSpikesE - ( v[i] > 35.0))
                                                                  + plast.WIE * NbSpikesI);


                I[i] = Ieff[i] + Ichem[i];
                v[i] = v[i] + dt * (1./100.) * (0.7 * (v[i]+60)*(v[i]+40) - u[i] + I[i]);

                u[i] = u[i] + dt * 0.03 * ( -2 * (v[i] + 60) - u[i] );
                vv[i] = v[i] > 35.0;

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
        NbSpikesI = NewNbSpikesI;
        NbSpikesE = NewNbSpikesE;
        meanBurst += getAvg<double>(p, sim1->NI);
//        meanSpikeNonBurst += getAvg<bool>(nonbursting, sim1->NI);
        meanSpike += NbSpikesI/(sim1->NI*1.0);  // only for inhibitory neurons here

        if (sim1->FOURIER){
            vm.push_back(getAvg<double>(v, sim1->NI));
        }
        pm.push_back(getAvg<double>(p, sim1->NI));
        qm.push_back(getAvg<double>(q, sim1->NI));
        lm.push_back(getAvg<double>(LowSp, sim1->NI));

        /***************************************************
         * PLASTICITY
         ***************************************************/
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
                plast.plasticityLocal(p, nonbursting, t);
                double instantMeanG = 0;
                for (int i = 0; i < sim1->NI; i++) {
                    for (int j = 0; j < i; j++) {
                        instantMeanG += plast.VgapLocal[i][j];
                    }
                }
                instantMeanG /= (sim1->NI * (sim1->NI - 1) / 2);
                mvgamma.compute(instantMeanG, t, T); //moving average
            }
        }
        
        /***************************************************
         * CORRELATION
         ***************************************************/
        if (sim1->CORRELATION and !sim1->CONSOLE){
            if(t % int(T/1000) == 0 and t>200/dt ){
                double corr = correlation.computeCorrelation(spikeTimesCor, dt);
                corrVect.push_back(corr);
            }
        }
        
        /***************************************************
         * SAVE SPIKE MEAN AND CURRENT AT TRANSITIONS
         ***************************************************/
        if (!sim1->CONSOLE) {
            ssp.push_back(NbSpikesI);
//            if (t < sim1->T1 + sim1->after and t > sim1->T1 - sim1->before){
//                current1.push_back( getAvg(I, N) );
//                ssp1.push_back(NbSpikes);
//            }
//            else if (t < sim1->T2 + sim1->after and t > sim1->T2 - sim1->before){
//                current2.push_back(getAvg(I, N));
//                ssp2.push_back(NbSpikes);
//            }
//            else if (t < T and t > (T - sim1->after - sim1->before)){
//                current3.push_back(getAvg(I, N));
//                ssp3.push_back(NbSpikes);
//            }
        }
    }
    
    /***************************************************
     * END OF SIMULATION - SAVE DATA
     ***************************************************/
    
    // end time counter
    //
    if (!sim1->CONSOLE) {
        time (&endtime);
        double dif = difftime (endtime,starttime);
        cout << "*****************************" << endl;
        printf ("Elasped time is %.2lf seconds.", dif );
        cout << " End of simulation: " << endl;
    }



    // SAVE DATA
    //
    if (!sim1->CONSOLE) {
        // SAVE DATA NORMAL MODE
        //
        util.writedata("gamma", mvgamma.outputVec);
        util.writedata("stimulation", stimulation);
        util.writedata("correlation", corrVect);
        util.writedata("p", pm);
        util.writedata("q", qm);
        util.writedata("LowSp", lm);
        util.writedataint("spike_x", spikes_idx);
        util.writedataint("spike_y", spikes_idy);
        util.writedataint("spike_x_tc", spikes_idx_tc);
        util.writedataint("spike_y_tc", spikes_idy_tc);
        util.writedata("vm", vm);
        if (sim1->FOURIER) {
//            util.writedata("vm", vm);
//            util.writedata("freq", FFT.freq);
//            util.writedata("amp", FFT.amp);
        }
        
//        util.writedata("current1", current1);
//        util.writedata("current2", current2);
//        util.writedata("current3", current3);
        
        util.writedata("ssp", ssp);
//        util.writedata("ssp1", ssp1);
//        util.writedata("ssp2", ssp2);
//        util.writedata("ssp3", ssp3);
        
//        util.writedata("RON_I", RON_I);
//        util.writedata("RON_V", RON_V);
//        util.writedata("V", vVect);

        cout << " Data written in  " << util.sim.path << endl;

        if (sim1->GLOB) {
            cout << "gamma_c : " << plast.Vgap*sim1->NI <<"\tRON_I:\t" << meanRON_I << endl;
        }
        else {
            cout << "gamma_c avg : " << meanG/counter << endl;
        }
    }
    else {
        // SAVE DATA CONSOLE MODE
        //
        if (sim1->FOURIER) {
            // compute FFT on Local Field Potential
            //
            FFT.computeFFT(vm);
        }
        
        double resultCorr = correlation.computeCorrelation(spikeTimesCor, dt);
        string path_csv =  "/Users/"+sim1->computer+sim1->directory+namecsv+".csv";
        ofstream csvFile(path_csv, std::ofstream::out | std::ofstream::app);
        csvFile << plast.Vgap*sim1->NI << ";" << sim1->TImean << ";" << resultCorr << ";" << meanSpike/T << ";" << meanSpikeNonBurst/T << ";" << meanBurst/T <<";"<< FFT.fftFreq<<";"<< FFT.fftPower<< endl;
        cout << "Data written: " << plast.Vgap*N << " " << sim1->TImean << "\tCorr:" << resultCorr ;
        cout << "\tSp:\t" << meanSpike/T << "\tB:\t" << meanBurst/T;
        cout << "\tfreq: " << FFT.fftFreq << "\tpower: " << FFT.fftPower <<endl;
    }
    
    
    return 0;
}



