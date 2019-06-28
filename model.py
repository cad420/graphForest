import tensorflow as tf
import numpy as np

from multiprocessing import Pool
from module import Bivalue, JS, Softmax, get_walk_size
from walks import generate_dis

class graphForest:
    def __init__(self, hp):
        self.G = []
        self.core_num = 0
        self.core = []
        self.weight = None
        self.hp = hp

    def add_core(self, core):
        self.core.append(core)
        self.core_num += 1
        self.G.append(core.G[-1])


    def update(self, ID_G):
        pass

    def adjust_weight(self, G_new):
        js = np.zeros((self.core_num))
        for i, G in enumerate(self.G):
            js[i] = JS(G, G_new)
        self.weight = Softmax(js)

    def voted_predict(self, xs):
        y_w_set = []
        for x in xs:
            y_w = 0
            for i, m in enumerate(self.core):
                y_w += m.predict(x) * self.weight[i]
            y_w_set.append(y_w)
        return y_w_set

    def voted_eval(self, xs, ys):
        y_w_set = []
        for x in xs:
            y_w = 0
            for i, m in enumerate(self.core):
                y_w += m.predict(x) * self.weight[i]
            y_w_set.append(y_w)
        auc = tf.py_func(Bivalue, [y_w_set, ys], [tf.double], stateful=True)
        return auc

class graphCore:
    def __init__(self, hp, G):
        self.hp = hp
        self.G = G #  a list
        self.dis_local = [{} for i in range(len(self.G))]
        self.dis_global = [{} for i in range(len(self.G))]

    def get_dis(self):
        for k, G in enumerate(self.G):
            sub_size_list, degree = get_walk_size(self.hp, G)
            node_list = list(G.nodes())
            node_list = node_list[degree[node_list]>3]
            per_threads_node = len(node_list) // self.hp.walkers
            results = []
            pool = Pool(processes=self.hp.walkers)
            for i in range(self.hp.walkers):
                if i == self.hp.walkers - 1:
                    results.append(
                        pool.apply_async(generate_dis, (self.hp, node_list[per_threads_node * i:], sub_size_list, G)))
                else:
                    results.append(pool.apply_async(generate_dis, (
                    self.hp, node_list[per_threads_node * i:per_threads_node * (i + 1)], sub_size_list, G)))
            pool.close()
            pool.join()
            results = [res.get() for res in results]
            for loc, glo in results:
                for jk in loc.keys():
                    self.dis_local[k][jk] = loc[jk]
                for jk in glo.keys():
                    self.dis_global[k][jk] = glo[jk]

    def train(self, xs, ys):
        pass

    def eval(self, xs, ys):
        pass

    def predict(self, xs):
        pass