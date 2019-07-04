import networkx as nx
import random
import numpy as np
from module import deg_distribution
from queue import Queue

def neighbor_sampling(args, node_list, G, max_node, degree):
    dis_set = {}
    for node in node_list:
        vis = [0 for j in range(max_node)]
        walk = []
        vis[node] = 1
        q = Queue()
        q.put(node)
        while len(walk) < args.walk_l and q.qsize()>0:
            u = q.get()
            walk.append(u)
            for v in G.neighbors(u):
                if q.qsize()>= args.walk_length:
                    break
                if not vis[v]:
                    q.put(v)
                    vis[v] = 1
        dis_set[node] = deg_distribution([degree[i] for i in walk])
    return dis_set

def onepath_walking(args, node_list, walk_size_list, G, max_node, degree):
    dis_set = {}
    for node in node_list:
        walks = []
        for k in range(walk_size_list[node]):
            vis = [0 for j in range(max_node)]
            walk = []
            vis[node] = 1
            walk.append(node)
            u = node
            while len(walk) < args.walk_length:
                updata = 0
                for v in G.neighbors(u):
                    if not vis[v]:
                        vis[v] = 1
                        walk.append(v)
                        updata = 1
                        break
                if not updata:
                    break
            walks.extend(walk)
        dis_set[node] = deg_distribution([degree[i] for i in walks])
    return dis_set

def generate_dis(args, node_list, walk_size_list, G, max_node, degree):
    dis_loc = neighbor_sampling(args, node_list, G, max_node, degree)
    dis_glo = onepath_walking(args, node_list, walk_size_list, G, max_node, degree)
    return [dis_loc, dis_glo]