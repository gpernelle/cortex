import matplotlib

matplotlib.use("Agg")
import fns
from fns import *
from fns.functionsTFhardbound import *

today = datetime.date.today()
todayStr = '%04d%02d%02d' % (today.year, today.month, today.day)
# todayStr = '20151005'
DIRECTORY = os.path.expanduser("~/Dropbox/0000_PhD/figures/" + todayStr + "/")
CSV_DIR_TODAY = os.path.expanduser("~/Dropbox/0000_PhD/csv/" + todayStr + "/")
CSV_DIR = os.path.expanduser("~/Dropbox/0000_PhD/csv/")
FIG_DIR = os.path.expanduser("~/Dropbox/0000_PhD/figures/")
if not os.path.exists(DIRECTORY):
    os.makedirs(DIRECTORY)

PAPER = os.path.expanduser('~/Dropbox/ICL-2014/Presentations/2016-10-11-GJ-sync-paper/figures/')
sns.set_context("paper", font_scale=1.5, rc={"lines.linewidth": 2.5})

df = pd.DataFrame(columns=('tauv', 'sG', 'both', 'T', 'N', 'nu',
                           'corS_apple', 'corE_apple', 'corChange_apple',
                           'f1Begin', 'p1Begin', 'f2Begin', 'p2Begin',
                            'f1End', 'p1End', 'f2End', 'p2End',
                            'maxBegin', 'argmaxBegin', 'maxEnd', 'argmaxEnd'
                          )  )

i=0
N = 2000
dt = 0.1
T = 20000
s0 = 0
d2 = int(1000//dt)-s0
end = T
both = True
nuList = [0,50,100,150]

ratio = 0.5
for nu in nuList:
#         for tauv in np.arange(15,90, 5):
    for tauv in [15,30,45,60,90]:
#                 for sG in np.arange(0,30,2):
                for sG in  [0,10,50,100,200]:
                    i+=1
                    extension = extension = "-tauv-%d_g-%d_N-%d_T-%d_k-%d_r-%.2f_dt-%.1f" % (tauv, sG, N, T, nu,  ratio, dt)
                    filename = "../data/rasters/rastervarPlast2" + extension
                    if i==1:
                        print(filename)
                    a = np.load(filename)
                    i1 = a['i1N1']
                    i2 = a['i1N2']
                    v1 = a['vvmN1']
                    v2 = a['vvmN2']

                    cor1 = np.corrcoef(i1[s0:s0+d2], i2[s0:s0+d2])[0,1]
                    cor2 = np.corrcoef(i1[T-d2:T], i2[T-d2:T])[0,1]

                    corChange = cor2/cor1


                    f, Pxy = signal.csd(v1[s0:s0+d2], v2[s0:s0+d2], fs=1 / 0.0001, nperseg=1024)
                    f2, Pxy2 = signal.csd(v1[T-d2:T], v2[T-d2:T], fs=1 / 0.0001, nperseg=1024)

                    maxBegin = np.max(np.abs(Pxy))
                    argmaxBegin = np.argmax(np.abs(Pxy))
                    maxEnd = np.max(np.abs(Pxy2))
                    argmaxEnd = np.argmax(np.abs(Pxy2))

                    f1Begin = fourier(v1[s0:s0+d2])[0]
                    p1Begin = fourier(v1[s0:s0+d2])[1]

                    f2Begin = fourier(v2[s0:s0+d2])[0]
                    p2Begin = fourier(v2[s0:s0+d2])[1]

                    f1End = fourier(v1[T-d2:T])[0]
                    p1End = fourier(v1[T-d2:T])[1]

                    f2End = fourier(v2[T-d2:T])[0]
                    p2End = fourier(v2[T-d2:T])[1]


#                     df.loc[i] = [int(tauv), int(sG), bool(both), int(T), int(N), float(a['cor1']), float(a['cor2']) ]
                    df.loc[i] = [int(tauv), int(sG), bool(both), int(T), int(N), int(nu),
                                 cor1, cor2,
                                 corChange,
                                f1Begin, p1Begin, f2Begin, p2Begin,
                                f1End, p1End, f2End, p2End,
                                maxBegin, argmaxBegin, maxEnd, argmaxEnd]


vmin = 20
vmax = 60
plotHeatmap(df[df['nu']==100], col='f1Begin', x='sG',y='tauv', cmap='viridis',
           vmin=vmin, vmax=vmax, annot=True, fmt = '.0f')
plt.savefig(PAPER+'f1begin.pdf')
plotHeatmap(df[df['nu']==100], col='f2Begin', x='sG',y='tauv', cmap='viridis',
           vmin=vmin, vmax=vmax, annot=True)
plotHeatmap(df[df['nu']==100], col='f1End', x='sG',y='tauv', cmap='viridis',
           vmin=vmin, vmax=vmax, annot=True)
plotHeatmap(df[df['nu']==100], col='f2End', x='sG',y='tauv', cmap='viridis',
           vmin=vmin, vmax=vmax, annot=True)