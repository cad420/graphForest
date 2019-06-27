import networkx as nx
import glob
from multiprocessing import Pool

def read_one_graph(threadID, G_file_name):
    G = nx.Graph()
    cnt = 0
    count = 0
    f = open(G_file_name, "r")
    for line in f.readlines():
        count = count + 1
    f.close()
    G_file = open(G_file_name, 'r')
    for line in G_file:
        tem = line[:-1].split(' ')
        if len(tem) < 2:
            break
        x = int(tem[0])
        y = int(tem[1])
        G.add_edge(x, y)
        cnt += 1
        print("\r读取第%d个图  %.4f" % (threadID, cnt/count), end=" ")
    G_file.close()
    return G

def read_graph(args, file_add):
    files = glob.glob(file_add+"/*.dat")
    file_num = len(files)
    results = []
    pool = Pool(processes=file_num)
    for i in range(file_num):
        results.append(
            pool.apply_async(read_one_graph, (i+1, files[i])))
    pool.close()
    pool.join()
    G_set = [res.get() for res in results]
    return G_set

def train_data(args, id, G):
    pass

def eval_data(args, id, G):
    pass

def test_data(args, id):
    pass

