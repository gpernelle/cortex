from utils import *

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