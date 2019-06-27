import networkx as nx
import glob
import pickle
import os
import generator
import tensorflow as tf
from module import spliter, convert_2_str, batch_fn
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

def train_data(args, id, G_list):
    graph_filename = '../train/'+args.dataset+'_train/'+str(id)+".sub"
    if not os.path.exists(graph_filename):
        subgraph_set = generator.generate_train_data(args, id, G_list)
        pickle.dump(subgraph_set, open(graph_filename, 'wb'))
    else:
        print("已经存在数据，开始读取数据")
        subgraph_set = pickle.load(open(graph_filename, 'rb'))
    print("开始生成训练数据")
    # print(subgraph_set)
    sub = []
    loc = []
    glo = []
    output = []
    for s, sub_set in enumerate(subgraph_set):
        subset_size = len(sub_set)
        per_spliter_each = subset_size // args.spliter
        results = []
        pool = Pool(processes=args.spliter)
        for i in range(args.spliter):
            if i == args.spliter - 1:
                results.append(pool.apply_async(spliter, (sub_set[per_spliter_each * i:], G_list[s])))
            else:
                results.append(
                    pool.apply_async(spliter,
                                     (sub_set[per_spliter_each * i:per_spliter_each * (i + 1)], G_list[s])))
        pool.close()
        pool.join()

        results = [res.get() for res in results]
        s_sub = []
        s_loc = []
        s_glo = []
        s_out = []
        for i in range(args.spliter):
            s_sub += results[i][0]
            s_loc += results[i][1]
            s_glo += results[i][2]
            s_out += results[i][3]
        sub.extend(s_sub)
        loc.extend(s_loc)
        glo.extend(s_glo)
        output.extend(s_out)
        print("第 %d 批训练数据完成" % s)
    print("生成训练数据完成")

    print("取出   训练    线程数据")
    #None是变长，其他维数固定
    shapes = (([None, None], [None, None], [None, None], ()),
              ([None, None]))
    padded_shapes = (([None, None], [None, None], [None, None], ()),
              ([None, None]))
    types = ((tf.float32, tf.float32, tf.float32),
             (tf.float32))

    sub_s = convert_2_str(sub)
    loc_s = convert_2_str(loc)
    glo_s = convert_2_str(glo)
    out_s = convert_2_str(output)

    number_samples = len(sub_s)
    print("共有训练数据集:  ", number_samples)

    input_dataset = tf.data.Dataset.from_generator(
        batch_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sub_s, loc_s, glo_s, out_s))  # <- arguments 必须是定长的，不可以是不定长的list

    input_dataset = input_dataset.repeat()  # iterate forever
    input_dataset = input_dataset.padded_batch(args.batch_size, padded_shapes, padding_values=None, drop_remainder=True).prefetch(1)

    results = [number_samples // args.batch_size + (number_samples % args.batch_size != 0), number_samples]
    return input_dataset, results

def eval_data(args, id, G_list):
    graph_filename = '../train/'+args.dataset+'_eval/'+str(id)+".sub"
    if not os.path.exists(graph_filename):
        subgraph_set = generator.generate_eval_data(args, id, G_list)
        pickle.dump(subgraph_set, open(graph_filename, 'wb'))
    else:
        print("已经存在数据，开始读取数据")
        subgraph_set = pickle.load(open(graph_filename, 'rb'))
    print("开始生成验证数据")
    # print(subgraph_set)
    sub = []
    loc = []
    glo = []
    output = []

    subset_size = len(subgraph_set)
    per_spliter_each = subset_size // args.spliter
    results = []
    pool = Pool(processes=args.spliter)
    for i in range(args.spliter):
        if i == args.spliter - 1:
            results.append(pool.apply_async(spliter, (subgraph_set[per_spliter_each * i:], G_list)))
        else:
            results.append(
                pool.apply_async(spliter,
                                 (subgraph_set[per_spliter_each * i:per_spliter_each * (i + 1)], G_list)))
    pool.close()
    pool.join()

    results = [res.get() for res in results]
    s_sub = []
    s_loc = []
    s_glo = []
    s_out = []
    for i in range(args.spliter):
        s_sub += results[i][0]
        s_loc += results[i][1]
        s_glo += results[i][2]
        s_out += results[i][3]
    sub.extend(s_sub)
    loc.extend(s_loc)
    glo.extend(s_glo)
    output.extend(s_out)

    print("生成验证数据完成")
    print("取出   验证    线程数据")

    shapes = (([None, None], [None, None], [None, None], ()),
              ([None, None]))
    padded_shapes = (([None, None], [None, None], [None, None], ()),
              ([None, None]))
    types = ((tf.float32, tf.float32, tf.float32),
             (tf.float32))

    sub_s = convert_2_str(sub)
    loc_s = convert_2_str(loc)
    glo_s = convert_2_str(glo)
    out_s = convert_2_str(output)

    number_samples = len(sub_s)
    print("共有训练数据集:  ", number_samples)

    input_dataset = tf.data.Dataset.from_generator(
        batch_fn,
        output_shapes=shapes,
        output_types=types,
        args=(sub_s, loc_s, glo_s, out_s))  # <- arguments 必须是定长的，不可以是不定长的list

    input_dataset = input_dataset.repeat()  # iterate forever
    input_dataset = input_dataset.padded_batch(args.batch_size, padded_shapes, padding_values=None, drop_remainder=True).prefetch(1)

    results = [number_samples // args.batch_size + (number_samples % args.batch_size != 0), number_samples]
    return input_dataset, results


