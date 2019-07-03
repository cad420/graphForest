from module import get_walk_size, walker
from multiprocessing import Pool

def generate_train_data(args, max_node, G_list):
    res = []
    print(max_node)
    print("开始随机采样子图")
    for k in range(len(G_list)):
        # print(G_list[k])
        subgraph_set = []
        sub_size_list, degree = get_walk_size(args, G_list[k], max_node)
        all_node = list(G_list[k].nodes())
        node_list = [x for x in all_node if degree[x] > 3]
        per_threads_node = len(node_list) // args.walkers
        # 创建新线程
        results = []
        pool = Pool(processes=args.walkers)
        for i in range(args.walkers):
            if i == args.walkers - 1:
                results.append(
                    pool.apply_async(walker, (args, sub_size_list, all_node, ['train', G_list[k]], node_list[per_threads_node * i:], max_node)))
            else:
                results.append(pool.apply_async(walker, (
                args, sub_size_list, all_node, ['train', G_list[k]], node_list[per_threads_node * i:per_threads_node * (i + 1)], max_node)))
        pool.close()
        pool.join()
        print("所有游走完成")
        results = [res.get() for res in results]
        for it in results:
            for jk in it:
                subgraph_set.append(jk)
        print("取出   游走   线程数据")
        res.append(subgraph_set)
    return res

def generate_eval_data(args, max_node, G_list):
    subgraph_set = []
    sub_size_list, degree = get_walk_size(args, G_list[0], max_node)
    all_node = list(set(G_list[0].nodes()) & set(G_list[1].nodes()))
    node_list = [x for x in all_node if degree[x] > 3]
    per_threads_node = len(node_list) // args.walkers
    print("开始生成测试数据")
    # 创建新线程
    results = []
    pool = Pool(processes=args.walkers)
    for i in range(args.walkers):
        if i == args.walkers - 1:
            results.append(
                pool.apply_async(walker, (args, sub_size_list, all_node, ['test', G_list], node_list[per_threads_node * i:], max_node)))
        else:
            results.append(pool.apply_async(walker, (
                args, sub_size_list, all_node, ['test', G_list], node_list[per_threads_node * i:per_threads_node * (i + 1)], max_node)))
    pool.close()
    pool.join()
    print("所有游走完成")
    results = [res.get() for res in results]
    for it in results:
        for jk in it:
            subgraph_set.append(jk)
    print("取出   游走   线程数据")
    return subgraph_set
