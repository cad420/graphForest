import tensorflow as tf
import numpy as np
import powerlaw

def JS(pl1, pl2): #JS散度计算
    X = [0.01*i for i in range(100000)]
    res = 0
    for x in X:
        p_x = pl1.c*(x**(-pl1.a))
        q_x = pl2.c*(x**(-pl2.a))
        res = res + p_x*np.log(2*p_x/(p_x+q_x))
    return res

def deg_distribution(seq):#计算power-law 分布
    data = np.array(seq)
    results = powerlaw.distribution_fit(data)
    dd = {}
    dd.a = results.power_law.alpha
    dd.c = results.power_law.xmin
    return dd

def Bivalue(logist, label):
    logist = 1 / (1 + np.exp(-logist))
    # print(logist)
    # print(label)
    logist[logist > 0.5] = 1.0
    logist[logist <= 0.5] = 0

    exist_total = 0
    no_exist_total = 0
    exist_label = 0
    no_exist_label = 0
    for n in range(logist.shape[0]):
        for i in range(logist[n].shape[0]):
            for j in range(logist[n][i].shape[0]):
                # total += 1
                # if abs(logist[n][i][j] - label[n][i][j]) < 0.00001:
                #     true_label += 1
                if label[n][i][j] == 1:
                    exist_total += 1
                    if logist[n][i][j] == label[n][i][j]:
                        exist_label += 1
                else:
                    no_exist_total += 1
                    if logist[n][i][j] == label[n][i][j]:
                        no_exist_label += 1
    return exist_label/exist_total, no_exist_label/no_exist_total, (exist_label+no_exist_label)/(exist_total+no_exist_total)

def Softmax(arr):
    arr = np.exp(arr)
    total = np.sum(arr)
    return arr / total

def get_walk_size(args, G):
    pass