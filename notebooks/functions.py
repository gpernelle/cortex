from functionsTF import *

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


def readDataFile(path):
    '''
    Read data extracted form graph with GraphClick
    :param path: file path
    :return: x,y
    '''
    x=[]
    y=[]
    with open(path,'r') as f:
        reader = csv.reader(f,delimiter='\t')
        for row in reader:
            try:
                x.append(float(row[0]))
                y.append(float(row[1]))
            except:
                pass
    return x,y



@autojit
def resonanceFS(F, tauv=15):
    T = 2000
    dt = 1
    t = np.arange(0, T, dt)
    F = np.logspace(0.5, 2.3, 200)

    res_var = np.empty(len(F), dtype=np.float64)
    b = 2
    for k, f in enumerate(F):
        A = 0.01
        I = A * np.cos(2 * np.pi * f * t / 1000)
        res_v = []
        res_u = []
        u = 0
        t_rest = 0

        # izh neuron model for cortical fast spiking neurons (that burst)
        v = -60
        for i in range(len(t)):
            v += dt / tauv * ((v + 60) * (v + 50) - 20 * u + 8 * I[i])
            u += dt * 0.044 * ((v + 55) - u)
            if v > 25:
                v = -40
                u += 50
            if i * dt > 1500:
                res_v.append(v / A)

        var = np.var(res_v)
        #         var = np.max(res_v)-np.min(res_v)
        res_var[k] = var
    return res_var

@autojit
def fourier(signal):
    f_val, p_val = maxPowerFreq(signal[int(signal.shape[0] / 2):], 0.25 / 1000)
    return [f_val, p_val]

@autojit
def maxPowerFreq(y, dt):
    # return the max power of the signal and its associated frequency
    fs = 1. / dt
    y = y - np.mean(y)
    t = np.arange(0, y.shape[0], 1)
    p1, f1 = psd(y, NFFT=len(t), pad_to=len(t), Fs=fs)
    powerVal = 10 * np.log10(max(p1))
    powerFreq = np.argmax(p1) * np.max(f1) / len(f1)
    return powerFreq, powerVal

def svg2pdf(filename, path = '/Users/GP1514/Dropbox/0000_PhD/figures/20160704/'):
    subprocess.check_output(["inkscape", "--file", '%s%s.svg'%(path, filename),
                           '--export-area-drawing','--without-gui', '--export-pdf',
                            '%s%s.pdf'%(path,filename), '--export-latex'])

def svg2eps(filename, path = '/Users/GP1514/Dropbox/0000_PhD/figures/20160704/'):
    subprocess.check_output(["inkscape", '%s%s.svg'%(path, filename),
                            '-E', '%s%s.eps'%(path,filename), '--without-gui',
                             '--export-ignore-filters','--export-ps-level=3'])


def vmin_vmax(df, kind="burst", v=0):
    data = pd.melt(df, id_vars=['tauv', 'sG'], value_vars=[kind + '1', kind + '2'])
    #     print(data.head())
    if v >= 0:
        vmin, vmax = np.percentile(data['value'], v), np.percentile(data['value'], 100)
    else:
        vmin = None
        vmax = None
    # print(vmin, vmax)
    return vmin, vmax


def facet_heatmap(data, df=None, v=0, vmin=None, vmax=None, **kws):
    kind = data['variable'].get_values()[0][:-1]
    if vmin == None:
        vmin, _ = vmin_vmax(df, kind=kind, v=v)
    if vmax == None:
        _, vmax = vmin_vmax(df, kind=kind, v=v)
    data = data.pivot(index='tauv', columns='sG', values='value')
    im = sns.heatmap(data, yticklabels=10, xticklabels=10, vmin=vmin, vmax=vmax, **kws)  # <-- Pass kwargs to heatmap
    im.invert_yaxis()


def plotGridHeatmap(df, col_wrap=2, cols=['burst1', 'spike1', 'burst2', 'spike2'], v=-1, vmin=None, vmax=None, **kws):
    data = pd.melt(df, id_vars=['tauv', 'sG'], value_vars=cols)

    #     print(data.head())
    with sns.plotting_context(font_scale=5.5):
        g = sns.FacetGrid(data, col="variable", col_wrap=col_wrap, size=3, aspect=1)

    cbar_ax = g.fig.add_axes([.92, .3, .02, .4])  # <-- Create a colorbar axes
    g = g.map_dataframe(facet_heatmap, v=v, df=df, vmin=vmin, vmax=vmax,
                        cbar_ax=cbar_ax, **kws)  # <-- Specify the colorbar axes and limits
    g.set_titles(col_template="{col_name}", fontweight='bold', fontsize=18)
    g.fig.subplots_adjust(right=.9)  # <-- Add space so the colorbar doesn't overlap the plot
    return g

pd.options.mode.chained_assignment = None
def plotHeatmap(df, col, title='', cmap=None):
    plt.figure()
    '''
    plot heatmap using seaborn library
    '''
    burst = df[['tauv', 'sG', col]]
    burst.loc[:,('tauv')] = burst['tauv'].astype(int)
    burst.loc[:,('sG')] = burst['sG'].astype(int)
    c = burst.pivot('tauv','sG', col)
    im = sns.heatmap(c, yticklabels=5, xticklabels=2, cmap=cmap)
    im.invert_yaxis()
    sns.set_style("whitegrid")
    if not title:
        title=col
    plt.title(title)
    return 0