import tensorflow as tf
import numpy as np

from module import ff, positional_encoding, multihead_attention, noam_scheme, Bivalue, Biclass, get_token_embeddings
from multiprocessing import Pool
from module import Bivalue, JS_graph, Softmax, get_walk_size
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
        self.G.append(self.core[self.core_num-1].G[-1])


    def update(self, ID_G):
        pass

    def adjust_weight(self, G_new):
        js = [0.0 for i in range(self.core_num)]
        for i in range(self.core_num):
            js[i] = JS_graph(self.G[i], G_new)
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
        y_w = self.core[0].predict(xs) * self.weight[0]
        for i in range(1, self.core_num):
            y_w = y_w + self.core[i].predict(xs) * self.weight[i]
        exist_pre, no_exist_pre, all_pre = tf.py_func(Bivalue, [y_w, ys], [tf.double, tf.double, tf.double], stateful=True)
        return exist_pre, no_exist_pre, all_pre

class graphCore:
    def __init__(self, hp, ID, G, max_node):
        self.core_ID = ID
        self.hp = hp
        self.G = G #  a list
        self.max_node = max_node + 10
        self.dis_local = [{} for i in range(len(self.G))]
        self.dis_global = [{} for i in range(len(self.G))]
        self.embeddings = get_token_embeddings(self.hp.max_graph_size, self.hp.d_model, self.core_ID, zero_pad=True)

    def get_dis(self):
        for k in range(len(self.G)):
            G = self.G[k]
            sub_size_list, degree = get_walk_size(self.hp, G, self.max_node)
            node_list = list(G.nodes())
            # node_list = [x for x in node_list if degree[x] > 3]
            per_threads_node = len(node_list) // self.hp.walkers
            results = []
            pool = Pool(processes=self.hp.walkers)
            for i in range(self.hp.walkers):
                if i == self.hp.walkers - 1:
                    results.append(
                        pool.apply_async(generate_dis, (self.hp, node_list[per_threads_node * i:], sub_size_list, G, self.max_node, degree)))
                else:
                    results.append(pool.apply_async(generate_dis, (
                    self.hp, node_list[per_threads_node * i:per_threads_node * (i + 1)], sub_size_list, G, self.max_node, degree)))
            pool.close()
            pool.join()
            results = [res.get() for res in results]
            for loc, glo in results:
                for jk in loc.keys():
                    self.dis_local[k][jk] = loc[jk]
                for jk in glo.keys():
                    self.dis_global[k][jk] = glo[jk]
            # print(self.dis_local[k])

    def Dimension_Fit(self, sub, loc, glo):
        sub = tf.einsum('ntd,dk->ntk', sub, self.embeddings)
        loc = tf.einsum('ntd,dk->ntk', loc, self.embeddings)
        glo = tf.einsum('ntd,dk->ntk', glo, self.embeddings)
        return sub, loc, glo

    def Concate_Attention(self, sub, loc, glo, training=True):
            with tf.variable_scope("sub_attention_%d"%self.core_ID, reuse=tf.AUTO_REUSE):
                sub = multihead_attention(queries=sub,
                                                  keys=sub,
                                                  values=sub,
                                                  num_heads=self.hp.num_heads,
                                                  dropout_rate=self.hp.dropout_rate,
                                                  training=training,
                                                  causality=False)
                sub = ff(sub, num_units=[self.hp.d_ff, self.hp.d_model])
            with tf.variable_scope("loc_attention_%d"%self.core_ID, reuse=tf.AUTO_REUSE):
                loc = multihead_attention(queries=loc,
                                          keys=loc,
                                          values=loc,
                                          num_heads=self.hp.num_heads,
                                          dropout_rate=self.hp.dropout_rate,
                                          training=training,
                                          causality=False)
                loc = ff(loc, num_units=[self.hp.d_ff, self.hp.d_model])
            with tf.variable_scope("glo_attention_%d"%self.core_ID, reuse=tf.AUTO_REUSE):
                glo = multihead_attention(queries=glo,
                                          keys=glo,
                                          values=glo,
                                          num_heads=self.hp.num_heads,
                                          dropout_rate=self.hp.dropout_rate,
                                          training=training,
                                          causality=False)

                glo = ff(glo, num_units=[self.hp.d_ff, self.hp.d_model])
            with tf.variable_scope("con_attention_%d"%self.core_ID, reuse=tf.AUTO_REUSE):
                att = multihead_attention(queries=sub,
                                          keys=loc,
                                          values=glo,
                                          num_heads=self.hp.num_heads,
                                          dropout_rate=self.hp.dropout_rate,
                                          training=training,
                                          causality=False)

                att = ff(att, num_units=[self.hp.d_ff, self.hp.d_model])
            return att
    def Attention(self, att, training = True):
        for i in range(self.hp.num_blocks):
            with tf.variable_scope("num_blocks_%d_%d"%(i, self.core_ID), reuse=tf.AUTO_REUSE):
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
        with tf.variable_scope("Dense_%d" % (self.core_ID), reuse=tf.AUTO_REUSE):
            att = tf.layers.dense(inputs=att, units=1024, activation=tf.nn.relu)
            att = tf.layers.dense(inputs=att, units=128, activation=tf.nn.relu)
            att = tf.einsum('ntd,nkd->ntk', att, att)  # (N, T2, T2)
            logits = (att + tf.transpose(att, [0, 2, 1])) / 2  # 强制最终结果为一个对称矩阵，符合
        return logits

    def train(self, xs, ys):
        sub, loc, glo = xs
        y = ys
        sub, loc, glo = self.Dimension_Fit(sub, loc, glo)
        att = self.Concate_Attention(sub, loc, glo)
        att = self.Attention(att)
        logits = self.Dense(att)
        ce = tf.nn.sigmoid_cross_entropy_with_logits(logits=logits, labels=y)
        loss = tf.reduce_sum(ce)
        global_step = tf.train.get_or_create_global_step()
        lr = noam_scheme(self.hp.lr, global_step, self.hp.warmup_steps)
        optimizer = tf.train.AdamOptimizer(lr)
        train_op = optimizer.minimize(loss, global_step=global_step)

        return loss, train_op, global_step

    def eval(self, xs, ys):
        sub, loc, glo = xs
        y = ys
        sub, loc, glo = self.Dimension_Fit(sub, loc, glo)
        att = self.Concate_Attention(sub, loc, glo)
        att = self.Attention(att)
        logits = self.Dense(att)
        exist_pre, no_exist_pre, all_pre = tf.py_func(Bivalue, [logits, y], [tf.double, tf.double, tf.double],
                                                      stateful=True)
        return exist_pre, no_exist_pre, all_pre

    def predict(self, xs):
        sub, loc, glo = xs
        sub, loc, glo = self.Dimension_Fit(sub, loc, glo)
        att = self.Concate_Attention(sub, loc, glo)
        att = self.Attention(att)
        logits = self.Dense(att)
        # logits = tf.py_func(Biclass, [logits], [tf.float32], stateful=True)
        return logits