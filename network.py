import networkx as nx
import numpy as np
import pandas as pd
import random
import matplotlib.pyplot as plt
import math
import time
import statistics
import os
import pickle


'''
Network class that uses networkx graph to represent the network
local variables:
n: # of nodes
m: # of links
graph: graph representation of the network
BANDWIDTH: bandwidth of each link if all links share the same bandwidth
DELAY_LOW: lower bound of delay time for links
DELAY_HIGH: higher bound of delay time for links
'''

class Network:
	def __init__(self, n = 0, m = 0, input_file = None, BANDWIDTH = 10, DELAY_LOW = 5, DELAY_HIGH = 25):
		self.BANDWIDTH = BANDWIDTH
		self.DELAY_LOW = DELAY_LOW
		self.DELAY_HIGH = DELAY_HIGH

		if (input_file):
			self.graph = self.build_network_from_input_file(input_file)
		else:
			# build_random_network(self, n, m)
			pass

	def build_network_from_input_file(self, input_file):
		with open(input_file) as f:
			self.n = int(next(f))
			G = nx.Graph()

			# add nodes to the graph
			G.add_nodes_from([0, self.n - 1])

			# add edges to the graph

			# get each row from input file
			for _ in range(self.n):
				cur_row = [int(x) for x in next(f).split()]
				u = cur_row[0]
				for v in cur_row[1:]:
					G.add_edge(u, v, bandwidth = self.BANDWIDTH, \
							   delay = random.randint(self.DELAY_LOW, self.DELAY_HIGH), \
							   residual_bandwidth = self.BANDWIDTH)

			return G

class Flow:
    def __init__(self, src, des, bandwidth_request, delay_request):
        self.bandwidth_request = bandwidth_request
        self.delay_request = delay_request
        self.src = src
        self.des = des

    def __str__(self):
        return str((self.src, self.des, self.bandwidth_request, self.delay_request))

    def __repr__(self):
        return str((self.src, self.des, self.bandwidth_request, self.delay_request))


    @classmethod
    def generate_random_flow(self, network, BANDWIDTH_LOW = 1, BANDWIDTH_HIGH = 5, DELAY_LOW = 0, DELAY_HIGH = 200):
        number_of_nodes = network.graph.number_of_nodes()
        src = random.randint(0, number_of_nodes - 1)
        des = src
        while (des == src):
            des = random.randint(0, number_of_nodes - 1)
        return Flow(src, des, \
                    random.randint(BANDWIDTH_LOW, BANDWIDTH_HIGH), \
                    random.randint(DELAY_LOW, DELAY_HIGH))

    @classmethod
    def generate_random_flows(self, n, network, BANDWIDTH_LOW = 1, BANDWIDTH_HIGH = 5, DELAY_LOW = 0, DELAY_HIGH = 200):
        flows = []
        for _ in range(n):
            flows.append(Flow.generate_random_flow(network, BANDWIDTH_LOW = BANDWIDTH_LOW, BANDWIDTH_HIGH = BANDWIDTH_HIGH, DELAY_LOW = DELAY_LOW, DELAY_HIGH = DELAY_HIGH))
        return flows



def pd_mcr(network, flows):

    # Initialize x(e) := 0, ∀e∈E
    for e in network.graph.edges():
        u, v = e
        network.graph[u][v]['x'] = 0

    num_of_accepted = 0
    for flow in flows:
        if (route(network, flow)):
            num_of_accepted += 1

    success_rate = num_of_accepted / len(flows)
    # print("success rate: " + str(success_rate))
    return success_rate




def route(network, flow):
    # print("trying to route: " + str(flow))

    G = network.graph
    n = network.graph.number_of_nodes()
    path = mcr_bellman(network.graph, flow, 1, n, 'x')

    path_x = 0
    path_delay = 0

    if path:
        for e in path:
            u, v = e

            G[u][v]['residual_bandwidth'] = G[u][v]['residual_bandwidth'] - flow.bandwidth_request
            path_x = path_x + G[u][v]['x']
            path_delay = path_delay + G[u][v]['delay']

            G[u][v]['x'] = G[u][v]['x'] * math.exp(math.log(1 + n) \
                                                   * flow.bandwidth_request / G[u][v]['bandwidth']) \
                            + 1 / n * (math.exp(math.log(1 + n) \
                                                   * flow.bandwidth_request / G[u][v]['bandwidth']) - 1)

            # print(u, v, G[u][v]['x'])
        # print("--success, by path: ", path, ', x: ', path_x, ', delay: ', path_delay)
        return True

    else:
        # print("--fail to route!")
        return False


'''
flow: src, des, bandwidth_request, delay_request(c2 \\in Z), c1 = 1
network: (G)  G[u][v]['delay'] -> w2; G[u][v]['x'] -> w1
'''
def mcr_bellman(G, flow, c1, n, w1):
    # !!! note this is undirected graph, so need to relax twice for both directions
    def relax(u, v, k):
        kk = k + G[u][v]['delay']
        if kk <= c2 and dist[v][kk] > dist[u][k] + G[u][v][w1]:
            dist[v][kk] = dist[u][k] + G[u][v][w1]
            par[v][kk] = u

    def construct_path():

        # path is feasible, reconstruct the path from end to start
        path = []
        prevNode = None
        curNode = des
        delay = 0
        # pathNodes = {curNode}

        # check if the result is feasible, if not, return null
        path_exist = False
        for k in range(c2 + 1):
            if dist[des][k] <= c1:
                # print(des, k, dist[des][k])
                path.append((par[curNode][k], curNode))
                prevNode, curNode = curNode, par[curNode][k]
                delay = k
                path_exist = True
                break

        if not path_exist:
            return None

        while curNode != src:
            # print(curNode, prevNode, delay)
            delay = delay - G[curNode][prevNode]['delay']
            nextNode = par[curNode][delay]
            prevNode, curNode = curNode, nextNode
            path.append((curNode, prevNode))


        # while curNode != src:
        #     for k in range(c2 + 1):
        #         # if src == 2 and des == 11:
        #         #     print(k, curNode, par[curNode][k])
        #         if par[curNode][k] != None and par[curNode][k] not in pathNodes:
        #             path.append((par[curNode][k], curNode))
        #             curNode = par[curNode][k]
        #             pathNodes.add(curNode)
        #             break

        # reverse list
        path.reverse()
        return path


    src = flow.src
    des = flow.des
    # c1 = 1
    c2 = flow.delay_request
    # n = G.number_of_nodes()

    # Initialization
    dist = [[math.inf] * (c2 + 1) for _ in range(n)]
    par = [[None] * (c2 + 1) for _ in range(n)]
    for i in range(c2 + 1):
        dist[src][i] = 0

    # Bellman-Ford
    for j in range(n):
        for k in range(c2 + 1):
            for e in G.edges:
                u, v = e
                relax(u, v, k)
                relax(v, u, k)


    # print(dist)
    # print(par)
    # CONSTRUCT PATH
    path = construct_path()

    return path

def e2e(network, flows):
    num_of_accepted = 0
    for flow in flows:
        if (e2e_route(network, flow)):
            num_of_accepted += 1

    success_rate = num_of_accepted / len(flows)
    # print("success rate: " + str(success_rate))
    return success_rate



def e2e_route(network, flow):
    # print("trying to route: " + str(flow))
    G = network.graph
    n = network.graph.number_of_nodes()

    # create subgraph for current flow
    valid_edges = []
    for e in G.edges():
        u, v = e
        if G[u][v]['residual_bandwidth'] >= flow.bandwidth_request:
            valid_edges.append(e)

    curG = G.edge_subgraph(valid_edges)

    max_bandwidth_utilization = 0

    for e in curG.edges():
        u, v = e;
        G[u][v]['bandwidth_utilization'] = flow.bandwidth_request / G[u][v]['residual_bandwidth']
        max_bandwidth_utilization = max(max_bandwidth_utilization, G[u][v]['bandwidth_utilization'])


    # path = e2e_mcr_bellman(curG, flow, max_bandwidth_utilization, n)
    path = mcr_bellman(curG, flow, max_bandwidth_utilization * n, n, 'bandwidth_utilization')


    if path:
        for e in path:
            u, v = e
            G[u][v]['residual_bandwidth'] = G[u][v]['residual_bandwidth'] - flow.bandwidth_request
        # print("--success, by path: ", path)
        return True

    else:
        # print("--fail to route!")
        return False





def run_experiment(config):
    x = []
    acc_rate_pd_mcr_mean = []
    acc_rate_e2e_mean = []
    acc_rate_pd_mcr_sd = []
    acc_rate_e2e_sd = []
    all_flows = []


    time_pd_mcr_mean = []
    time_e2e_mean = []
    time_pd_mcr_sd = []
    time_e2e_sd = []

    input_file = os.path.join(config.get('network_folder', ''), config['network'])
    print(input_file)


    for num_of_flow in config['number_of_flows_range']:
        acc_rate_pd_mcr = []
        acc_rate_e2e = []
        time_pd_mcr = []
        time_e2e = []

        print("Testing: ", num_of_flow, "flows")
        for _ in range(config['duplicates']):
            network = Network(input_file = input_file, BANDWIDTH = config['network_bandwidth'], DELAY_LOW = config['network_delay_low'], DELAY_HIGH = config['network_delay_high'])
            flows = Flow.generate_random_flows(num_of_flow, network, BANDWIDTH_LOW = config['flow_bandwidth_low'], BANDWIDTH_HIGH = config['flow_bandwidth_high'], DELAY_LOW = config['flow_delay_low'], DELAY_HIGH = config['flow_delay_high'])

            if config['same_flow_test']:
                flows = [flows[0]] * num_of_flow

            all_flows.append(flows)

            network = Network(input_file = input_file, BANDWIDTH = config['network_bandwidth'], DELAY_LOW = config['network_delay_low'], DELAY_HIGH = config['network_delay_high'])

            start_1 = time.time()
            acc_rate_pd_mcr.append(pd_mcr(network, flows))
            time_pd_mcr.append(time.time() - start_1)



            network = Network(input_file = input_file, BANDWIDTH = config['network_bandwidth'], DELAY_LOW = config['network_delay_low'], DELAY_HIGH = config['network_delay_high'])

            start_2 = time.time()
            acc_rate_e2e.append(e2e(network, flows))
            time_e2e.append(time.time() - start_2)


        print("pd_mcr: ", sum(acc_rate_pd_mcr) / len(acc_rate_pd_mcr))
        print("e2e: ", sum(acc_rate_e2e) / len(acc_rate_e2e))

        x.append(num_of_flow)

        acc_rate_pd_mcr_mean.append(statistics.mean(acc_rate_pd_mcr))
        acc_rate_e2e_mean.append(statistics.mean(acc_rate_e2e))
        acc_rate_pd_mcr_sd.append(statistics.stdev(acc_rate_pd_mcr))
        acc_rate_e2e_sd.append(statistics.stdev(acc_rate_e2e))

        time_pd_mcr_mean.append(statistics.mean(time_pd_mcr))
        time_e2e_mean.append(statistics.mean(time_e2e))
        time_pd_mcr_sd.append(statistics.stdev(time_pd_mcr))
        time_e2e_sd.append(statistics.stdev(time_e2e))




    filename_head = os.path.join(config.get('output_folder', ''), config['file_name_prefix']) + config['network'] + str(time.time())

    # with error bar

    plt.xlabel("Number of Flows")
    plt.ylabel("Mean Acceptance Rate of" + str(config['duplicates']) + "experiments")
    plt.title(config['network'] + "Acceptance Rate")

    plt.plot(x,acc_rate_pd_mcr_mean,label='pd_mcr', color='red',marker='o')
    plt.errorbar(x, acc_rate_pd_mcr_mean, acc_rate_pd_mcr_sd, linestyle='None', color='red', marker='o', capsize=config['duplicates'])

    plt.plot(x,acc_rate_e2e_mean,label='e2e', color='blue',marker='o')
    plt.errorbar(x, acc_rate_e2e_mean, acc_rate_e2e_sd, linestyle='None', color='blue', marker='o', capsize=config['duplicates'])

    plt.legend()
    if config['file_output']:
        plt.savefig(filename_head + '_experiment_acc_rate_with_error_bar.pdf')
    plt.show()




    plt.xlabel("Number of Flows")
    plt.ylabel("Mean Execution Time of" + str(config['duplicates']) + "experiments")
    plt.title(config['network'] + "Execution Time")

    plt.plot(x,time_pd_mcr_mean,label='pd_mcr', color='red',marker='o')
    plt.errorbar(x, time_pd_mcr_mean, time_pd_mcr_sd, linestyle='None', color='red', marker='o', capsize=config['duplicates'])

    plt.plot(x, time_e2e_mean,label='e2e', color='blue',marker='o')
    plt.errorbar(x, time_e2e_mean, time_e2e_sd, linestyle='None', color='blue', marker='o', capsize=config['duplicates'])

    plt.legend()
    if config['file_output']:
        plt.savefig(filename_head + '_experiment_ext_time_with_error_bar.pdf')
    plt.show()



    # without error bar

    plt.xlabel("Number of Flows")
    plt.ylabel("Mean Acceptance Rate of" + str(config['duplicates']) + "experiments")
    plt.title(config['network'] + "Acceptance Rate")
    plt.plot(x,acc_rate_pd_mcr_mean,label='pd_mcr', color='red',marker='o')

    plt.plot(x,acc_rate_e2e_mean,label='e2e', color='blue',marker='o')

    plt.legend()
    if config['file_output']:
        plt.savefig(filename_head  + '_experiment_acc_rate.pdf')
    plt.show()




    plt.xlabel("Number of Flows")
    plt.ylabel("Mean Execution Time of" + str(config['duplicates']) + "experiments")
    plt.title(config['network'] + "Execution Time")

    plt.plot(x,time_pd_mcr_mean,label='pd_mcr', color='red',marker='o')

    plt.plot(x, time_e2e_mean,label='e2e', color='blue',marker='o')

    plt.legend()
    if config['file_output']:
        plt.savefig(filename_head + '_experiment_ext_time.pdf')
    plt.show()


    result = Result(config, all_flows, x, acc_rate_pd_mcr_mean, acc_rate_pd_mcr_sd, acc_rate_e2e_mean, acc_rate_e2e_sd, time_pd_mcr_mean, time_pd_mcr_sd, time_e2e_mean, time_e2e_sd)

    with open(filename_head + 'all_flows_and_outputs.pickle', 'wb') as handle:
        pickle.dump(result, handle, protocol=pickle.HIGHEST_PROTOCOL)


class Result:
    def __init__(self, config, all_flows, x, acc_rate_pd_mcr_mean, acc_rate_pd_mcr_sd, acc_rate_e2e_mean, acc_rate_e2e_sd, time_pd_mcr_mean, time_pd_mcr_sd, time_e2e_mean, time_e2e_sd):
        self.config = config
        self.all_flows = all_flows
        self.x = x
        self.acc_rate_pd_mcr_mean = acc_rate_pd_mcr_mean
        self.acc_rate_pd_mcr_sd = acc_rate_pd_mcr_sd
        self.acc_rate_e2e_mean = acc_rate_e2e_mean
        self.acc_rate_e2e_sd = acc_rate_e2e_sd
        self.time_pd_mcr_mean = time_pd_mcr_mean
        self.time_pd_mcr_sd = time_pd_mcr_sd
        self.time_e2e_mean = time_e2e_mean
        self.time_e2e_sd = time_e2e_sd



def main():
    pass



if __name__ == '__main__':
    main()













