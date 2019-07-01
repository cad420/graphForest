import tensorflow as tf
import numpy as np

from module import ff, positional_encoding, multihead_attention, noam_scheme, Bivalue, Biclass
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

    def Attention(self, xs, training=True):
        with tf.variable_scope("Attention", reuse=tf.AUTO_REUSE):
            sub, loc, glo, seqlens = xs
            att = multihead_attention(queries=sub,
                                              keys=loc,
                                              values=glo,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
            for i in range(self.hp.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i), reuse=tf.AUTO_REUSE):
                    # self-attention
                    att = multihead_attention(queries=att,
                                              keys=att,
                                              values=att,
                                              num_heads=self.hp.num_heads,
                                              dropout_rate=self.hp.dropout_rate,
                                              training=training,
                                              causality=False)
                    # feed forward
                    att = ff(att, num_units=[self.hp.d_ff, self.hp.d_model])
            return att
    def Dense(self, att):
        att = tf.layers.dense(inputs=att, units=1024, activation=tf.nn.relu)
        att = tf.layers.dense(inputs=att, units=128, activation=tf.nn.relu)
        att = tf.einsum('ntd,nkd->ntk', att, att)  # (N, T2, T2)
        logits = (att + tf.transpose(att, [0, 2, 1])) / 2  # 强制最终结果为一个对称矩阵，符合
        return logits

    def train(self, xs, ys):
        y, seqlens = ys
        att = self.Attention(xs)
        logits = self.Dense(att)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_sum(ce)
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step

    def eval(self, xs, ys):
        y, _ = ys
        att = self.Attention(xs)
        logits = self.Dense(att)
        exist_pre, no_exist_pre, all_pre = tf.py_func(Bivalue, [logits, y], [tf.double, tf.double, tf.double],
                                                      stateful=True)
        return exist_pre, no_exist_pre, all_pre

    def predict(self, xs):
        att = self.Attention(xs)
        logits = self.Dense(att)
        logits = tf.py_func(Biclass, [logits], [tf.double], stateful=True)
        return logits