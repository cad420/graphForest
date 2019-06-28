import tensorflow as tf
import numpy as np
import powerlaw
import random

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

def Bivalue(logist, label): #计算正确率
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

def get_walk_size(args, G): #得到动态步长，需要修改公式
    node_num = len(G.nodes())
    size_list = [0 for i in range(args.node_num)]
    degree = [0 for i in range(args.node_num)]
    for edge in G.edges():
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    deg_map = {}
    for node in G.nodes():
        deg_map[node] = degree[node]
    ds = [0 for i in range(args.node_num)]
    k = np.log10(node_num)
    for v in G.nodes():
        ds[v] += degree[v]
        for node in G.neighbors(v):
            ds[v] += degree[node]
        size_list[v] = int(args.max_each * (degree[v] / ds[v] + 1 / (1 + np.exp(-degree[v] / k))))
    return size_list, deg_map

def walker(args, sub_size_list, degree, G, node_list): # 得到子图的集合
    G = G[-1]
    subgraph_set = []
    for i in node_list:
        for k in range(sub_size_list[i]):
            sub_node_num = random.randint(3, args.max_graph_size)
            seta = 5 * sub_node_num
            tem_vis = [0 for j in range(args.node_num)]
            tem_node_set = set()
            sub_node_set = []
            tem_node_set.add(i)
            tem_vis[i] = 1
            while len(sub_node_set) < sub_node_num:
                choose_node = random.sample(tem_node_set, 1)
                tem_node_set.remove(choose_node[0])
                sub_node_set.append(choose_node[0])
                if len(tem_node_set) < seta:
                    for j in G.neighbors(choose_node[0]):
                        if not tem_vis[j] and degree[j] >= 2:
                            tem_vis[j] = 1
                            tem_node_set.add(j)
                if (len(tem_node_set) <= 0):
                    break
            subgraph_set.append(sub_node_set)
    return subgraph_set

def spliter(subgraph_set, G_list, get_dis): # 对子图加工得到输入模型的数据
    if len(G_list) == 1:
        Go = G_list
        Gn = G_list
    else:
        Go = G_list[0]
        Gn = G_list[1]
    loc_dis, glo_dis = get_dis
    sub = []
    loc = []
    glo = []
    out = []
    for sub_set in subgraph_set:
        sub_size = len(sub_set)
        tem_sub = np.zeros((sub_size, sub_size))
        tem_loc = np.zeros((sub_size, sub_size))
        tem_glo = np.zeros((sub_size, sub_size))
        tem_out = np.zeros((sub_size, sub_size))
        all_size = sub_size**2
        cou = 0
        for i, node in enumerate(sub_set):
            for j in range(i+1, sub_size):
                if node in Go.neighbor(sub_set[j]):
                    cou += 1
                    tem_sub[i][j] = 1
                    tem_sub[j][i] = 1
                tem_loc[i][j] = JS(loc_dis[node], loc_dis[sub_set[j]])
                tem_loc[j][i] = tem_loc[i][j]

                tem_glo[i][j] = JS(glo_dis[node], glo_dis[sub_set[j]])
                tem_glo[j][i] = tem_glo[i][j]
                if node in Gn.neighbor(sub_set[j]):
                    tem_out[i][j] = 1
                    tem_out[j][i] = 1
        if cou / all_size < 0.3:
            continue
        sub.append(tem_sub)
        loc.append(tem_loc)
        glo.append(tem_glo)
        out.append(tem_out)
    return [sub, loc, glo, out]

def convert_2_str(x): #转化为字符串
    str_x = []
    for x_i in x:
        tem = ''
        row = len(x_i)
        for it in x_i:
            for y in it:
                tem = tem + str(y) + '_'
        tem = tem + str(row) + '_' + str(len(x_i[0]))
        str_x.append(tem)
    return str_x

def convert_2_arr(x): #转化为矩阵
    arr_x = []
    # print(x)
    tem = x.split("_")
    row = int(tem[-2])
    col = int(tem[-1])
    pos = 0
    for i in range(row):
        tem_col = []
        for j in range(col):
            # print(pos)
            tem_col.append(float(tem[pos]))
            pos += 1
        arr_x.append(tem_col)

    return np.array(arr_x), row

def batch_fn(sub, loc, glo, out): # 批量数据产生
    for sub_, loc_, glo_, out_ in zip(sub, loc, glo, out):
        sub_, x_seqlen = convert_2_arr(sub_.decode())
        loc, _ = convert_2_arr(loc_.decode())
        glo_, _ = convert_2_arr(glo_.decode())
        out_ , y_seqlen = convert_2_arr(out_.decode())
        yield (sub_, loc_, glo_, x_seqlen), (out_, y_seqlen)