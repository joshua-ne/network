import networkx as nx
import random, time, sys

# build graph from given file containing graph info
def build_graph(inputfile):
    
    print("building graph...", flush = True)
    
    G = nx.DiGraph()
    with open(inputfile, "r") as f:
        for line in f:
            if line[0] == '#':
                continue
            edge = [int(i) for i in line.split()]
            G.add_edge(edge[1], edge[0], weight = random.random())

    hosts = [n for n, d in G.in_degree() if d == 0]
    fogs = [n for n, d in G.out_degree() if d == 0]
    
    print("graph built with " + str(len(G.node())) + " nodes, and " + str(len(G.edges())) + " edges.", flush = True)
    print("Number of hosts: ", len(hosts), flush = True)
    print("Number of fog devices: ", len(fogs), flush = True)

    return [G, hosts, fogs]



class Path:
    def __init__(self, existing_path = None):
        if existing_path == None:
            self.path = []
            self.weight = 0.0
        else:
            self.path = existing_path.path
            self.weight = existing_path.weight
        
    def __str__(self):
        return str(self.weight) + " " + str(self.path)


def generate_path_set(src, des, G, maxLength = 20, silent = False):
    start_time = time.time()
    # init an empty set
    path_set = []
    
    # init local variables
    vis = [False] * (len(G.node()) + 100)
    cur_path = Path()
    
    find_path(src, des, cur_path, vis, path_set, G, maxLength)
    path_set.sort(key = lambda x: x.weight)
    
    if not silent:
        print(str(src) + " " + str(des) + ": " + str(len(path_set)) + " paths found in " + str(time.time() - start_time) + "")

    return path_set


def find_path(cur, des, cur_path, vis, path_set, G, maxLength):
    vis[cur] = True
    if cur == des:
        # print(cur_path)
        path_set.append(Path(cur_path))
        vis[cur] = False
        return
    
    # exit if the length of cur_path exceeds maxLength limit
    if (len(cur_path.path) > maxLength):
        vis[cur] = False
        return

    for u in G[cur]:
        if not vis[u]:
            cur_path.path.append(u)
            cur_path.weight = cur_path.weight + G[cur][u]['weight']
            find_path(u, des, cur_path, vis, path_set, G, maxLength)
            
            # backtrack
            del cur_path.path[-1]
            cur_path.weight = cur_path.weight - G[cur][u]['weight']
            
    vis[cur] = False
    return
