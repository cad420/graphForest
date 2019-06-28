from module import get_walk_size, walker
from multiprocessing import Pool

def generate_train_data(args, id, G_list):
    res = []
    for k, G in enumerate(G_list):
        subgraph_set = []
        sub_size_list, degree = get_walk_size(args, G)
        node_list = list(G.nodes())
        node_list = node_list[degree[node_list]>3]
        per_threads_node = len(node_list) // args.walkers
        # 创建新线程
        results = []
        pool = Pool(processes=args.walkers)
        for i in range(args.walkers):
            if i == args.walkers - 1:
                results.append(
                    pool.apply_async(walker, (args, sub_size_list, degree, G, node_list[per_threads_node * i:])))
            else:
                results.append(pool.apply_async(walker, (
                args, sub_size_list, degree, G, node_list[per_threads_node * i:per_threads_node * (i + 1)])))
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

def generate_eval_data(args, id, G_list):
    subgraph_set = []
    sub_size_list, degree = get_walk_size(args, G_list[0])
    node_list = list(set(G_list[0].nodes()) & set(G_list[1].nodes()))
    node_list = node_list[degree[node_list]>3]
    per_threads_node = len(node_list) // args.walkers
    # 创建新线程
    results = []
    pool = Pool(processes=args.walkers)
    for i in range(args.walkers):
        if i == args.walkers - 1:
            results.append(
                pool.apply_async(walker, (args, sub_size_list, degree, G_list, node_list[per_threads_node * i:])))
        else:
            results.append(pool.apply_async(walker, (
                args, sub_size_list, degree, G_list, node_list[per_threads_node * i:per_threads_node * (i + 1)])))
    pool.close()
    pool.join()
    print("所有游走完成")
    results = [res.get() for res in results]
    for it in results:
        for jk in it:
            subgraph_set.append(jk)
    print("取出   游走   线程数据")
    return subgraph_set
