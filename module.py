import tensorflow as tf
import numpy as np
import powerlaw
import random
import networkx as nx

def JS(pl1, pl2): #JS散度计算
    # return 0.1
    X = [0.01*i for i in range(1, 100000)]
    res = 0
    for x in X:
        p_x = (pl1['c']*(x**(-pl1['a'][0])))
        q_x = (pl2['c']*(x**(-pl2['a'][0])))
        # print(p_x,"  ",q_x)
        # print(np.log(2*p_x/(p_x+q_x)))
        res = res + p_x*(np.log(2*p_x/(p_x+q_x)))
    return res

def JS_graph(G1, G2): #JS散度计算
    # return 0.1
    seq_1 = []
    seq_2 = []
    for node, de in nx.degree(G1):
        seq_1.append(de)
    for node, de in nx.degree(G2):
        seq_2.append(de)
    pl1 = deg_distribution(seq_1)
    pl2 = deg_distribution(seq_2)
    X = [0.01*i for i in range(1, 100000)]
    res = 0
    for x in X:
        p_x = (pl1['c']*(x**(-pl1['a'][0])))
        q_x = (pl2['c']*(x**(-pl2['a'][0])))

        res = res + p_x*(np.log(2*p_x/(p_x+q_x)))
    return res

def deg_distribution(seq):#计算power-law 分布
    data = np.array(seq)
    if len(data)<3:
        dd = {}
        dd['a'] = [random.uniform(2, 3)]
        dd['c'] = random.uniform(4, 5)
        return dd

    results = powerlaw.distribution_fit(data)
    # print(results)
    dd = {}
    dd['a'] = results['fits']['power_law'][0]
    dd['c'] = results['fits']['power_law'][1]
    return dd

def Bivalue(logist, label): #计算正确率
    logist = 1 / (1 + np.exp(-logist))
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

def Biclass(logist): #得到预测矩阵
    logist = 1 / (1 + np.exp(-logist))
    logist[logist > 0.5] = 1.0
    logist[logist <= 0.5] = 0.0
    return logist

def Softmax(arr):
    arr = np.exp(arr)
    total = np.sum(arr)
    return (arr / total).tolist()

def get_walk_size(args, G, max_node): #得到动态步长，需要修改公式
    node_num = len(G.nodes())
    size_list = [0 for i in range(max_node)]
    degree = [0 for i in range(max_node)]
    for edge in G.edges():
        degree[edge[0]] += 1
        degree[edge[1]] += 1
    deg_map = {}
    for node in G.nodes():
        deg_map[node] = degree[node]
    ds = [0 for i in range(max_node)]
    k = np.log10(node_num)
    for v in G.nodes():
        ds[v] += degree[v]
        for node in G.neighbors(v):
            ds[v] += degree[node]
        size_list[v] = int(args.max_each * (degree[v] / ds[v] + 1 / (1 + np.exp(-degree[v] / k))))
    return size_list, deg_map

def walker(args, sub_size_list, all_node, G_p, node_list, max_node): # 得到子图的集合
    type, G = G_p
    if type == 'test':
        G = G[len(G)-2]
    subgraph_set = []
    for i in node_list:
        for k in range(sub_size_list[i]):
            sub_node_num = random.randint(3, args.max_graph_size)
            seta = 5 * sub_node_num
            tem_vis = [0 for j in range(max_node)]
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
                        if not tem_vis[j] and j in all_node:
                            tem_vis[j] = 1
                            tem_node_set.add(j)
                if (len(tem_node_set) <= 0):
                    break
            subgraph_set.append(sub_node_set)
    return subgraph_set

def spliter(args, subgraph_set, G_p, get_dis): # 对子图加工得到输入模型的数据
    type, G_list = G_p
    if type == 'train':
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
        tem_sub = np.zeros((args.max_graph_size, args.max_graph_size))
        tem_loc = np.zeros((args.max_graph_size, args.max_graph_size))
        tem_glo = np.zeros((args.max_graph_size, args.max_graph_size))
        tem_out = np.zeros((args.max_graph_size, args.max_graph_size))
        all_size = sub_size**2
        cou = 0
        for i, node in enumerate(sub_set):
            for j in range(i+1, sub_size):
                if node in Go.neighbors(sub_set[j]):
                    cou += 1
                    tem_sub[i][j] = 1
                    tem_sub[j][i] = 1
                # else:
                #     continue
                tem_loc[i][j] = JS(loc_dis[node], loc_dis[sub_set[j]])
                tem_loc[j][i] = tem_loc[i][j]

                tem_glo[i][j] = JS(glo_dis[node], glo_dis[sub_set[j]])
                tem_glo[j][i] = tem_glo[i][j]
                if node in Gn.neighbors(sub_set[j]):
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

    return np.array(arr_x)

def batch_fn(sub, loc, glo, out): # 批量数据产生
    for sub_, loc_, glo_, out_ in zip(sub, loc, glo, out):
        sub_ = convert_2_arr(sub_.decode())
        loc_ = convert_2_arr(loc_.decode())
        glo_ = convert_2_arr(glo_.decode())
        out_ = convert_2_arr(out_.decode())
        yield (sub_, loc_, glo_), (out_)

def ln(inputs, epsilon=1e-8, scope="ln"): # 网络层正则化
    '''Applies layer normalization. See https://arxiv.org/abs/1607.06450.
    inputs: A tensor with 2 or more dimensions, where the first dimension has `batch_size`.
    epsilon: A floating number. A very small number for preventing ZeroDivision Error.
    scope: Optional scope for `variable_scope`.

    Returns:
      A tensor with the same shape and data dtype as `inputs`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.get_variable("beta", params_shape, initializer=tf.zeros_initializer())
        gamma = tf.get_variable("gamma", params_shape, initializer=tf.ones_initializer())
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs

def scaled_dot_product_attention(Q, K, V, #另一种attention用法
                                 causality=False, dropout_rate=0.,
                                 training=True,
                                 scope="scaled_dot_product_attention"):
    '''See 3.2.1.
    Q: Packed queries. 3d tensor. [N, T_q, d_k].
    K: Packed keys. 3d tensor. [N, T_k, d_k].
    V: Packed values. 3d tensor. [N, T_k, d_v].
    causality: If True, applies masking for future blinding
    dropout_rate: A floating point number of [0, 1].
    training: boolean for controlling droput
    scope: Optional scope for `variable_scope`.
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        d_k = Q.get_shape().as_list()[-1]

        # dot product
        outputs = tf.matmul(Q, tf.transpose(K, [0, 2, 1]))  # (N, T_q, T_k)

        # scale
        outputs /= d_k ** 0.5

        # softmax
        outputs = tf.nn.softmax(outputs)
        attention = tf.transpose(outputs, [0, 2, 1])
        tf.summary.image("attention", tf.expand_dims(attention[:1], -1))

        # query masking
        # outputs = mask(outputs, Q, K, type="query")

        # dropout
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=training)

        # weighted sum (context vectors)
        outputs = tf.matmul(outputs, V)  # (N, T_q, d_v)

    return outputs

def multihead_attention(queries, keys, values, #多头注意力
                        num_heads=8,
                        dropout_rate=0,
                        training=True,
                        causality=False,
                        scope="multihead_attention"):
    '''Applies multihead attention. See 3.2.2
    queries: A 3d tensor with shape of [N, T_q, d_model].
    keys: A 3d tensor with shape of [N, T_k, d_model].
    values: A 3d tensor with shape of [N, T_k, d_model].
    num_heads: An int. Number of heads.
    dropout_rate: A floating point number.
    training: Boolean. Controller of mechanism for dropout.
    causality: Boolean. If true, units that reference the future are masked.
    scope: Optional scope for `variable_scope`.

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    # print(queries)
    d_model = queries.get_shape().as_list()[-1]
    # d_model = 128
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Linear projections
        # print(queries)
        Q = tf.layers.dense(queries, d_model, use_bias=False)  # (N, T_q, d_model)
        K = tf.layers.dense(keys, d_model, use_bias=False)  # (N, T_k, d_model)
        V = tf.layers.dense(values, d_model, use_bias=False)  # (N, T_k, d_model)

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, d_model/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, d_model/h)

        # Attention
        outputs = scaled_dot_product_attention(Q_, K_, V_, causality, dropout_rate, training)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, d_model)

        # Residual connection
        outputs += queries

        # Normalize
        outputs = ln(outputs)

    return outputs


def ff(inputs, num_units, scope="positionwise_feedforward"): #前向反馈层
    '''position-wise feed forward net. See 3.3

    inputs: A 3d tensor with shape of [N, T, C].
    num_units: A list of two integers.
    scope: Optional scope for `variable_scope`.

    Returns:
      A 3d tensor with the same shape and dtype as inputs
    '''
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        outputs = tf.layers.dense(inputs, num_units[0], activation=tf.nn.relu)

        # Outer layer
        outputs = tf.layers.dense(outputs, num_units[1])

        # Residual connection
        outputs += inputs

        # Normalize
        outputs = ln(outputs)

    return outputs

def noam_scheme(init_lr, global_step, warmup_steps=4000.): #学习梯度调整
    '''Noam scheme learning rate decay
    init_lr: initial learning rate. scalar.
    global_step: scalar.
    warmup_steps: scalar. During warmup_steps, learning rate increases
        until it reaches init_lr.
    '''
    step = tf.cast(global_step + 1, dtype=tf.float32)
    return init_lr * warmup_steps ** 0.5 * tf.minimum(step * warmup_steps ** -1.5, step ** -0.5)

def positional_encoding(inputs,
                        maxlen,
                        hp,
                        masking=False,
                        scope="positional_encoding"):
    '''Sinusoidal Positional_Encoding. See 3.5
    inputs: 3d tensor. (N, T, E)
    maxlen: scalar. Must be >= T
    masking: Boolean. If True, padding positions are set to zeros.
    scope: Optional scope for `variable_scope`.

    returns
    3d tensor that has the same shape as inputs.
    '''
    # print(type(maxlen))
    E = hp.d_model  # static
    N, T = tf.shape(inputs)[0], tf.shape(inputs)[1]  # dynamic
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # position indices
        position_ind = tf.tile(tf.expand_dims(tf.range(T), 0), [N, 1])  # (N, T)

        # First part of the PE function: sin and cos argument
        position_enc = np.array([
            [pos / np.power(10000, (i - i % 2) / E) for i in range(E)]
            for pos in range(maxlen)])

        # Second part, apply the cosine to even columns and sin to odds.
        position_enc[:, 0::2] = np.sin(position_enc[:, 0::2])  # dim 2i
        position_enc[:, 1::2] = np.cos(position_enc[:, 1::2])  # dim 2i+1
        position_enc = tf.convert_to_tensor(position_enc, tf.float32)  # (maxlen, E)

        # lookup
        outputs = tf.nn.embedding_lookup(position_enc, position_ind)

        # masks
        if masking:
            outputs = tf.where(tf.equal(inputs, 0), inputs, outputs)

        return tf.to_float(outputs)

def get_token_embeddings(vocab_size, num_units, core_ID, zero_pad=True):
    '''Constructs token embedding matrix.
    Note that the column of index 0's are set to zeros.
    vocab_size: scalar. V.
    num_units: embedding dimensionalty. E.
    zero_pad: Boolean. If True, all the values of the first row (id = 0) should be constant zero
    To apply query/key masks easily, zero pad is turned on.

    Returns
    weight variable: (V, E)
    '''
    with tf.variable_scope("shared_weight_matrix"):
        embeddings = tf.get_variable("weight_mat_%d"%core_ID,
                                     dtype=tf.float32,
                                     shape=(vocab_size, num_units),
                                     initializer=tf.contrib.layers.xavier_initializer())
        if zero_pad:
            embeddings = tf.concat((tf.zeros(shape=[1, num_units]),
                                    embeddings[1:, :]), 0)
    return embeddings