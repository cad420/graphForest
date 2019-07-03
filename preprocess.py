# 运行主程序前用来预处理数据
import glob
import os
from args import args
hparams = args()
parser = hparams.parser
hp = parser.parse_args()

files = glob.glob("raw-data/"+hp.dataset+"/*.txt")
print("共有 %d 个图"%len(files))
for k, file in enumerate(files):
    print("处理第 %d 个图" % (k+1))
    f = open(file, 'r')
    edge_cnt = 0
    node_set = set()
    biao = {}
    for line in f:
        temp = line[:-1].split(' ')
        # print(tem)
        if (len(temp)) < 3:
            break
        x = int(temp[0])
        y = int(temp[1])
        edge_cnt += 1
        node_set.add(x)
        node_set.add(y)
    f.close()
    biao = {}
    cn = 1
    for i in node_set:
        biao[str(i)] = str(cn)
        cn += 1
    node_num = cn - 1
    print('时间步数量 : ' + str(hp.timestep))
    print("节点数量: " + str(node_num))
    print("边数量 : " + str(edge_cnt))
    rep_num = edge_cnt // hp.timestep
    cou = 0
    edge_sort = [[] for i in range(hp.timestep)]
    G = [set() for i in range(hp.timestep)]
    f = open(file, 'r')
    for line in f:
        temp = line[:-1].split(' ')
        # print(fn)
        if (len(temp)) < 3:
            break
        time_stamp = cou // rep_num
        # print(time_stamp)
        if time_stamp >= hp.timestep:
            break
        edge_sort[time_stamp].append([biao[temp[0]], biao[temp[1]]])
        cou += 1
    if not os.path.exists("data/"+hp.dataset+"/"+str(k)):
        os.makedirs("data/"+hp.dataset+"/"+str(k))
    cnt = 0
    for start_t in range(hp.timestep):
        filename = str(cnt)
        cnt += 1
        f = open("data/"+hp.dataset+"/"+str(k) + '/graph_' + filename + '.inp', 'w')
        for it in edge_sort[start_t]:
            f.write(it[0] + ' ' + it[1] + '\n')
        f.close()