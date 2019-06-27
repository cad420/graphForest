import sys
sys.path.append("/")
import tensorflow as tf
import glob
from model import graphCore, graphForest
from tqdm import tqdm
from data_loader import read_graph, train_data, eval_data, test_data
from args import args
import math

def main():
    hparams = args()
    parser = hparams.parser
    hp = parser.parse_args()
    gF = graphForest(hp)
    graph_file_set = glob.glob("../data/"+hp.dataset+"_train/*")
    for model_id in range(hp.graph_num):
        print("开始读取数据")
        G_list = read_graph(hp, graph_file_set[model_id])
        input_set,  results = train_data(hp, model_id, G_list[:-1])
        num_train_batches, num_train_samples = results
        eval_input_set= eval_data(hp, model_id, G_list[-2:])
        print("读取数据完成")

        iter = tf.data.Iterator.from_structure(input_set.output_types, input_set.output_shapes)
        xs, ys = iter.get_next()

        train_init_op = iter.make_initializer(input_set)
        eval_init_op = iter.make_initializer(eval_input_set)
        print("构建模型 graphCore %d"%(model_id+1))
        m = graphCore(hp, G_list[model_id])
        m.get_dis()
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
        gF.core.append(m)
    test_file = glob.glob("../data/"+hp.dataset+"_train/*.dat")
    for i in range(len(test_file)-1):
        G, xs, ys = test_data(hp, test_file[i:i+2])
        gF.adjust_weight(G)
        pre = gF.voted_eval(xs, ys)
        print("graphForest 预测第%d个图的准确率为： %lf"%(i+1, pre))
if __name__ == '__main__':
    main()