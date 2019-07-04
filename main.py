import sys
sys.path.append("/")
import tensorflow as tf
import glob
from model import graphCore, graphForest
from tqdm import tqdm
from data_loader import read_graph, train_data, eval_data, test_data
from args import args
from walks import generate_dis
from module import get_walk_size
from multiprocessing import Pool
import numpy as np
import math

def main():
    hparams = args()
    parser = hparams.parser
    hp = parser.parse_args()
    gF = graphForest(hp)
    graph_file_set = glob.glob("data/"+hp.dataset+"/train/*")
    m = {}
    G_list = {}
    input_set = {}
    num_train_batches = {}
    eval_input_set = {}
    train_init_op = {}
    eval_init_op = {}

    loss = {}
    train_op = {}
    global_step = {}
    exist_pre = {}
    no_exist_pre = {}
    all_pre = {}
    for model_id in range(hp.graph_num):
        print("开始读取数据")
        G_list[model_id], max_node = read_graph(hp, graph_file_set[model_id])
        print("构建模型 graphCore %d" % (model_id + 1))
        m[model_id] = graphCore(hp, model_id, G_list[model_id], max_node)
        m[model_id].get_dis()

        input_set[model_id],  train_results = train_data(hp, model_id, G_list[model_id][:-1], [m[model_id].dis_local, m[model_id].dis_global], m[model_id].max_node)
        num_train_batches[model_id], num_train_samples = train_results
        eval_input_set[model_id], eval_results = eval_data(hp, model_id, G_list[model_id][len(G_list[model_id])-2:], [m[model_id].dis_local, m[model_id].dis_global], m[model_id].max_node)
        print("读取数据完成")


    test_file = "data/" + hp.dataset + "/test/"
    test_G_list, max_node = read_graph(hp, test_file)
    sub_size_list, degree = get_walk_size(hp, test_G_list[0], max_node)
    node_list = list(test_G_list[0].nodes())
    # node_list = [x for x in node_list if degree[x] > 3]
    per_threads_node = len(node_list) // hp.walkers
    results = []
    pool = Pool(processes=hp.walkers)
    for i in range(hp.walkers):
        if i == hp.walkers - 1:
            results.append(
                pool.apply_async(generate_dis,
                                 (hp, node_list[per_threads_node * i:], sub_size_list, test_G_list[0], max_node, degree)))
        else:
            results.append(pool.apply_async(generate_dis, (
                hp, node_list[per_threads_node * i:per_threads_node * (i + 1)], sub_size_list, test_G_list[0], max_node, degree)))
    pool.close()
    pool.join()
    results = [res.get() for res in results]
    loc_dis = {}
    glo_dis = {}
    for loc, glo in results:
        for jk in loc.keys():
            loc_dis[jk] = loc[jk]
        for jk in glo.keys():
            glo_dis[jk] = glo[jk]
    test_input_set = test_data(hp, test_G_list, [loc_dis, glo_dis], max_node)

    iter = tf.data.Iterator.from_structure(input_set[0].output_types, input_set[0].output_shapes)
    xs, ys = iter.get_next()
    for model_id in range(hp.graph_num):
        train_init_op[model_id] = iter.make_initializer(input_set[model_id])
        eval_init_op[model_id] = iter.make_initializer(eval_input_set[model_id])
        loss[model_id], train_op[model_id], global_step[model_id] = m[model_id].train(xs, ys)
        exist_pre[model_id], no_exist_pre[model_id], all_pre[model_id] = m[model_id].eval(xs, ys)
    test_init_op = iter.make_initializer(test_input_set)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        for model_id in range(hp.graph_num):
            sess.run(tf.global_variables_initializer())
            print("开始训练 graphCore %d" % (model_id + 1))
            sess.run(train_init_op[model_id])
            total_steps = hp.num_epochs * num_train_batches[model_id]
            _gs = sess.run(global_step[model_id])
            for i in tqdm(range(_gs, total_steps+1)):
                _, _gs = sess.run([train_op[model_id], global_step[model_id]])
                epoch = math.ceil(_gs / num_train_batches[model_id])
                if _gs and _gs % num_train_batches[model_id] == 0:
                    _loss = sess.run(loss[model_id]) # train loss
                    _ = sess.run([eval_init_op[model_id]])
                    pre, no_pre, al_pre = sess.run([exist_pre[model_id], no_exist_pre[model_id], all_pre[model_id]])
                    print("\n有边的预测准确率为：  ", pre)
                    print("无边的预测准确率为：  ", no_pre)
                    print("综合预测准确率为：  ", al_pre)
                    print("Epoch : %02d   loss : %.2f" % (epoch, _loss))
                    sess.run(train_init_op[model_id])
            gF.add_core(m[model_id])
        sess.run(tf.global_variables_initializer())
        sess.run(tf.local_variables_initializer())
        sess.run(test_init_op)
        gF.adjust_weight(test_G_list[0])
        test_exist_pre, test_no_exist_pre, test_all_pre = gF.voted_eval(xs, ys)
        pre, no_pre, al_pre = sess.run([test_exist_pre, test_no_exist_pre, test_all_pre])
        print("\n有边的预测准确率为：  ", pre)
        print("无边的预测准确率为：  ", no_pre)
        print("综合预测准确率为：  ", al_pre)

if __name__ == '__main__':
    main()