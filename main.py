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
    for model_id in range(hp.graph_num):
        print("开始读取数据")
        G_list, max_node = read_graph(hp, graph_file_set[model_id])
        print("构建模型 graphCore %d" % (model_id + 1))
        m = graphCore(hp, model_id, G_list, max_node)
        m.get_dis()

        input_set,  train_results = train_data(hp, model_id, G_list[:-1], [m.dis_local, m.dis_global], m.max_node)
        num_train_batches, num_train_samples = train_results
        eval_input_set, eval_results = eval_data(hp, model_id, G_list[len(G_list)-2:], [m.dis_local, m.dis_global], m.max_node)
        print("读取数据完成")

        iter = tf.data.Iterator.from_structure(input_set.output_types, input_set.output_shapes)
        xs, ys = iter.get_next()

        train_init_op = iter.make_initializer(input_set)
        eval_init_op = iter.make_initializer(eval_input_set)

        loss, train_op, global_step = m.train(xs, ys)
        exist_pre, no_exist_pre, all_pre = m.eval(xs, ys)
        print("开始训练 graphCore %d"%(model_id+1))
        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            sess.run(train_init_op)
            total_steps = hp.num_epochs * num_train_batches
            _gs = sess.run(global_step)
            for i in tqdm(range(_gs, total_steps+1)):
                _, _gs = sess.run([train_op, global_step])
                epoch = math.ceil(_gs / num_train_batches)
                if _gs and _gs % num_train_batches == 0:
                    _loss = sess.run(loss) # train loss
                    _ = sess.run([eval_init_op])
                    pre, no_pre, al_pre = sess.run([exist_pre, no_exist_pre, all_pre])
                    print("\n有边的预测准确率为：  ", pre)
                    print("无边的预测准确率为：  ", no_pre)
                    print("综合预测准确率为：  ", al_pre)
                    print("Epoch : %02d   loss : %.2f" % (epoch, _loss))
                    sess.run(train_init_op)
        gF.add_core(m)

    test_file = "data/"+hp.dataset+"/test/"
    G_list, max_node = read_graph(hp, test_file)
    gF.adjust_weight(G_list[0])
    sub_size_list, degree = get_walk_size(hp, G_list[0], max_node)
    node_list = list(G_list[0].nodes())
    # node_list = [x for x in node_list if degree[x] > 3]
    per_threads_node = len(node_list) // hp.walkers
    results = []
    pool = Pool(processes=hp.walkers)
    for i in range(hp.walkers):
        if i == hp.walkers - 1:
            results.append(
                pool.apply_async(generate_dis,
                                 (hp, node_list[per_threads_node * i:], sub_size_list, G_list[0], max_node)))
        else:
            results.append(pool.apply_async(generate_dis, (
                hp, node_list[per_threads_node * i:per_threads_node * (i + 1)], sub_size_list, G_list[0], max_node)))
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
    test_input_set = test_data(hp, G_list, [loc_dis, glo_dis], max_node)
    iter = tf.data.Iterator.from_structure(test_input_set.output_types, test_input_set.output_shapes)
    xs, ys = iter.get_next()
    test_init_op = iter.make_initializer(test_input_set)

    exist_pre, no_exist_pre, all_pre = gF.voted_eval(xs, ys)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(test_init_op)
        pre, no_pre, al_pre = sess.run([exist_pre, no_exist_pre, all_pre])
        print("\n有边的预测准确率为：  ", pre)
        print("无边的预测准确率为：  ", no_pre)
        print("综合预测准确率为：  ", al_pre)
if __name__ == '__main__':
    main()