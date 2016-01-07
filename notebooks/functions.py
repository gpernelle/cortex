from IO import *

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