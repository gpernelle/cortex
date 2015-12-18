from IO import *

def getHist(data, binsize):
    binnb = 0
    val = []
    for ind, i in enumerate(data):
        if (ind%binsize)==0:
            if binnb > 0:
                val.append(accumulator)
            accumulator = 0
            binnb +=1
        else:
            accumulator+=i
    return val

def movingaverage(values,window):
    weigths = np.repeat(1.0, window)/window
    smas = np.convolve(values, weigths, 'valid')
    return smas # as a numpy array

def chart(list1):
    hour_list = list1
    print(hour_list)
    numbers=[x for x in range(0,24)]
    labels=[str(x) for x in numbers]
    plt.xticks(numbers, labels)
    plt.xlim(0,24)
    plt.hist(hour_list)
    plt.show()

def runPSTH(N,G,S, it,binsize, d1, d2, d3, before, after, s, WII):
    arr = []
    results = Parallel(n_jobs=num_cores)(delayed(runSimulation)(N,i,G,S, d1, d2, d3, before, after, s, WII) for i in range(it))

def readPSTH(N,G,S, it,binsize, d1, d2, d3, before, after):
    coeff = -300
    Rm = 0.1
    arr = []
    listSSP1 = Parallel(n_jobs=num_cores)(delayed(readSimulationSSP1)(N,i,G,S, d1, d2, d3, before, after) for i in range(it))
    listS = Parallel(n_jobs=num_cores)(delayed(readoutSpikes)(ssp1, Rm, coeff) for ssp1 in listSSP1)
#     for ind, v in enumerate(listS):
#         readSimulation(T,N,i,G)
#         s=readoutSpikes(ssp1, Rm, coeff)*1
#         arr.append(s)
    tot = np.sum(listS, axis=0)
    total = np.sum(listS)
    h = getHist(tot,binsize)
    spikes_x, spikes_y, spikes_x_tc, spikes_y_tc, gamma, correlation, ssp1, stimulation = readSimulation(N,0,G,S, d1, d2, d3, before, after)
    
    return h, stimulation, total


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
    self.u = u0 if u0 is not None else b*v0

	
class IzhSim:
    def __init__(self, n, T, dt=0.1):
        self.neuron = n
        self.dt     = dt
        self.t      = t = np.arange(0, T+dt, dt)
        self.stim   = np.zeros(len(t))
        self.x      = 5
        self.y      = 140
        self.du     = lambda a, b, v, u: a*(b*v - u)
	
    def integrate(self, n=None):
        if n is None: n = self.neuron
        trace = np.zeros((3,len(self.t)))
        b = 2
        p = 0
        tau_p = 10
        for i, j in enumerate(self.stim):
        #       n.v += self.dt * (0.04*n.v**2 + self.x*n.v + self.y - n.u + self.stim[i])
            n.v += self.dt * 1/40.0 * (0.25*(n.v**2 + 110*n.v + 45*65) - n.u + self.stim[i])  
        #       n.u += self.dt * self.du(n.a,n.b,n.v,n.u)
            n.u += self.dt * 0.015 * ( b * (n.v+65 ) - n.u )
            p += self.dt / tau_p * ((n.v > 0)*tau_p/self.dt - p )
            if n.v > 0:
                trace[0,i] = 30
                n.v        = -55
                n.u       += 50          
            else:
                trace[0,i] = n.v
                trace[1,i] = n.u
                
            if n.v < -65:
                b = 10
            else:
                b=2
            trace[2,i] = p
        return trace

def readoutneuroncortex(ax, ssp1, Rm, W=-200, x=None):
    dt = 0.25
    bTh = 1.3
    T   = len(ssp1)/4                  # total time to simulate (msec)
    
    time    = np.arange(0, T, dt)
    n = IzhNeuron("Burst Mode", a=0.02, b=0.25, c=-50, d=2, v0=-70)
    s1 = IzhSim(n, T=T, dt=dt)
    
    for i, t in enumerate(time):
        s1.stim[i] = W*ssp1[i]
    
    res = s1.integrate()
    
    ax.set_ylim([-150,30])
    ax.spines['left'].set_position('zero')
    ax.spines['right'].set_color('none')
    #ax.spines['bottom'].set_position('zero')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
    if x: 
        ax.set_xlabel('Time (msec)')
    else:
        ax.set_xticks([])

    ax.plot(s1.t, res[0], label='membrane voltage [mV]')
    return ax

def readoutSpikes(ssp1, Rm, W=-200):
    dt = 0.25
    bTh = 1.3
    T   = len(ssp1)/4                  # total time to simulate (msec)
    
    time    = np.arange(0, T, dt)
    n = IzhNeuron("Burst Mode", a=0.02, b=0.25, c=-50, d=2, v0=-70)
    s1 = IzhSim(n, T=T, dt=dt)
    
    for i, t in enumerate(time):
        s1.stim[i] = W*ssp1[i]
    
    res = s1.integrate()
    spikes = res[0]>0
    
    return spikes

def findindex(val, spikes_x, start=0):
    result = spikes_x[-1]
    for i in range(start,len(spikes_x)):
        if spikes_x[i] == val or spikes_x[i] > val:
            result = i
            break
    return result
                                            
def plotraster(ax, spikes_x, spikes_y, i1, i2, dur=1000, end=0, x=0):
    spx = spikes_x[i1:i2]
    spy = spikes_y[i1:i2]
    ax.spines['right'].set_color('none')
    ax.spines['top'].set_color('none')
    ax.xaxis.set_ticks_position('bottom')
    ax.yaxis.set_ticks_position('left')
#     ax.set_ylim([0,100])
    if x==0: 
        ax.set_xticks([])
    else:
        ax.set_xlabel('Time (msec)')  
    ax.plot(spx,spy,'.',markersize=1)
    
def plotTransition(spikes_x, spikes_y, dur = 1000):
    t1 = findindex(int(T/3)-dur, spikes_x)
    t2 = findindex(int(T/3)+dur, spikes_x, t1)
    t3 = findindex(int(2*T/3)-dur, spikes_x, t2)
    t4 = findindex(int(2*T/3)+dur, spikes_x, t3)
    t5 = findindex(int(3*T/3)-2*dur, spikes_x, t4)
    t6 = len(spikes_x)
    
    fig=plt.figure(figsize=(10,10))
    ax1=fig.add_subplot(311)
    ax1=plotraster(ax1, spikes_x, spikes_y, t1,t2,dur)
    ax2=fig.add_subplot(312)
    ax2=plotraster(ax2, spikes_x, spikes_y, t3,t4, dur)
    ax3=fig.add_subplot(313)
    ax3=plotraster(ax3, spikes_x, spikes_y, t5,t6, dur, 1, 1)
    
    plt.savefig(DIRECTORY + extension + "transition_spikes_save.pdf")
    
    
def plotPTSH(before, after,binsize, h, s, it, DIRECTORY, S, N):
    T = before+after
    simsize=T/0.25
    x2 = np.arange(0,(simsize/4-1)/1000, (simsize/len(s)/4)/1000)

    fig = plt.figure(figsize=(9,5))
    gs = gridspec.GridSpec(2, 1, height_ratios=[3, 1]) 
    ax0 = plt.subplot(gs[0])
    ax0.set_ylim([0,10])
    ax0.bar(np.arange(0,simsize/1000,simsize/len(h)/1000),np.array(h)/it, binsize/1000)
    ax0.set_title('PSTH - Input: %s' % S)
    ax0.set_xticks([])

    # plot stimulation s 
    ax = plt.subplot(gs[1])
    ax.set_ylim([30,250])
    ax.set_yticks([30,200])
    ax.set_xlabel('Time [s]')
    plt.plot(x2,s)
    plt.tight_layout()
    extension = "_S-%d_N-%d_T-%d" % (S, N, T)
    print(DIRECTORY + extension + '_PTSH.pdf')
    plt.savefig(DIRECTORY + extension + '_PTSH.pdf')

def plotDiagram(figure, ax, dataframe, title, column, filename, save=True, front=False, gridsize=60):
    ax.set_ylim(ymin=50, ymax=400)
    # Set color transparency (0: transparent; 1: solid)
    a = 1
    # Create a colormap
    customcmap = [(x/24.0,  x/48.0, 0.05) for x in range(len(dataframe))]

    dataframe.plot(kind="hexbin",
                   y='nuEI', x='gammaC', C=column,gridsize=gridsize, ax=ax, alpha=a, legend=False, colormap=cx4,
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
        dataframe['logburst']= (dataframe['burst']).apply(np.log10)
        df_sliced = dataframe[(dataframe['logburst']>-6) & (dataframe['logburst']<-5.8)]
        contour = df_sliced[['gammaC','nuEI']].get_values()
        yvals, xvals = bezier_curve(contour, nTimes=100)
        frontier = np.array([xvals,yvals])
        ax.plot(frontier[1,:], frontier[0,:], '-w', linewidth=3)
    if save:
        plt.tight_layout()
        plt.savefig(DIRECTORY + filename)
    return ax
    
def savePTSH(before, after, h, s, it, DIRECTORY, S, N):
    '''
    Save PTSH data
    '''
    T = before+after
    simsize=len(s)
    print(T/1000, simsize, T/1000/simsize)
    x2 = np.arange(0,T/1000, T/1000/simsize)
    extension = "_S-%d_N-%d_T-%d" % (S, N, T)
    np.save(DIRECTORY + extension + '_data.npy', np.array([np.arange(len(h)),np.array(h)/it]))
    
def func(x, a, b, c):
    return a * np.exp(-b * x) + c

def plotRaster(spikes_x,spikes_y):
    f = plt.figure(figsize=(4,3))
    ax = f.add_subplot(111)
    ax.set_xticks([])
    ax.set_yticks([0,300])
    ax.set_xlabel('Time [1.5s]')
    ax.set_ylabel('Neuron indices [0-300]')
    ax.set_title('Neuronal Activity')
    plt.plot(spikes_x,spikes_y, '.', markersize=1)
    # plt.savefig(DIRECTORY + extension + '_raster.pdf')
    # plt.savefig(DIRECTORY + extension + '_raster.png')
    
def plotRasterGPU(spikes_x, spikes_y, titlestr):
    '''
    Take advantage of WebGL to draw raster plot
    '''
    # output_file("spikesGPU.html", title="Neural activity")

    p = figure(plot_width=1000, plot_height=500, webgl=True, title = titlestr)
    if len(spikes_x) > 100000:
        p.scatter(spikes_x[0:100000],spikes_y[0:100000], alpha=0.5)
    else:
        p.scatter(spikes_x, spikes_y, alpha=0.5)
    # save(p, filename=titlestr)
    show(p)

def fourier(signal):
    f_val, p_val = maxPowerFreq(signal[int(signal.shape[0]/2):],0.25/1000)
    return [f_val, p_val]

def maxPowerFreq(y, dt):
    # return the max power of the signal and its associated frequency
    fs = 1. / dt
    y = y - np.mean(y)
    t = np.arange(0, y.shape[0], 1)
    p1,f1 = psd(y, NFFT=len(t), pad_to=len(t), Fs=fs)
    powerVal = 10*np.log10(max(p1))
    powerFreq = np.argmax(p1) * np.max(f1)/len(f1)

    return powerFreq, powerVal

def bernstein_poly(i, n, t):
    """
     The Bernstein polynomial of n, i as a function of t
    """

    return comb(n, i) * ( t**(n-i) ) * (1 - t)**i


def bezier_curve(points, nTimes=1000):
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

    polynomial_array = np.array([ bernstein_poly(i, nPoints-1, t) for i in range(0, nPoints)   ])

    xvals = np.dot(xPoints, polynomial_array)
    yvals = np.dot(yPoints, polynomial_array)

    return xvals, yvals

def frontgamma(nuEI, frontier, tol = 10):
    '''
    Return the value of GAMMA for a value of NUEI
    '''
    indices = np.where(np.logical_and(frontier[0]<=nuEI + tol, frontier[0]>=nuEI - tol))
    return np.mean(frontier[1][indices[0]])

def find_nearest(array,value):
    idx = (np.abs(array-value)).argmin()
    return array[idx], idx

def fixpoint(df, nuEI, ratio, rule, g0 = 7):
    '''
    Search and return the fix point for a plasticity rule, or 0 if none
    '''
    LTP = 10
    LTD = LTP * ratio
    df_sliced = df[df['nuEI']==nuEI]
    df_sliced = df_sliced.sort_values(by=['gammaC'])
    bg = df_sliced[['burst','gammaC']].get_values()
    sg = df_sliced[['spike','gammaC']].get_values()
    rg = df_sliced[['non-burst','gammaC']].get_values()
    df_sliced['passive'] = df_sliced['burst'].apply(lambda x :1-x)
    passive = df_sliced[['passive','gammaC']].get_values()
#     sg = 1 - np.array(bg)
    cg =  df_sliced[['corI','gammaC']].get_values()
    if rule == 1:
        dG = LTP * sg[:,0] - LTD * bg[:,0]
    elif rule == 4:
        dG = LTP * passive[:,0] - LTD * bg[:,0]
    elif rule == 5:
        dG = LTP * rg[:,0] - LTD * bg[:,0]
    elif rule ==0:
        dG = LTP * (g0 - sg[:,1] )/g0 * sg[:,0] - LTD * bg[:,0]
    elif rule ==2:
        dG = LTP * (g0 - passive[:,1] )/g0 * passive[:,0] - LTD * bg[:,0]
    elif rule ==3:
        dG = LTP * (g0 - rg[:,1] )/g0 * rg[:,0] - LTD * bg[:,0]
    res = 0
    dg2 = np.array(dG)
    if (dg2 > 0).all() or (dg2 < 0).all():
        res = 0
    else:
        minval , idx =  find_nearest(dg2,0)
        res = sg[idx,1]
    return res

def overunder(df, frontier, rule):
    '''
    return 0 if fixpoint over in the SYNC regime, 1 otherwise
    '''
    if rule in [0,1,5,3]:
        col = np.arange(3,5,0.05) #ratio for active rule
    else:
        col = np.arange(60,180,3) #ratio for passive rule
    row = np.arange(0,200,1) #nu
    Z = np.zeros(shape=(len(row),len(col)))
    for i,ratio in enumerate(col):
        for j, nuEI in enumerate(row):
            f= fixpoint(df, nuEI, ratio, rule)
            border = frontgamma(nuEI, frontier)
            Z[len(row)-1-j,i] = abs(f-border)*((f<border) and (f>0.15) and (nuEI>48))*1.0
    return Z

def plotStability(ax, Z, ylabel, title, extent):
    cx_blue = cubehelix.cmap(reverse=False, start=3., rot=0)
    ax.set_xlabel('ratio')
    ax.set_ylabel(r'Excitatory input $\nu$')
    ax.set_title('Hard. all-spiking', y=1.08)
    image = ax.imshow(Z, interpolation='nearest', extent=extent,cmap=cx_blue, aspect=2/200)#, cmap =cx4)# drawing the function
    plt.colorbar(image)
    return ax

