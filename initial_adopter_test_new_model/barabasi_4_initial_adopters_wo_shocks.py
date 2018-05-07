#By Sally
#1. Apply shocks to only a subset of nodes
#2. Can control iterations by pressing enter
#3. Recording more info

#!/usr/bin/env python

# ==============================================================================
# LIBRARIES
# ==============================================================================
import bisect                       # For CDF functionality
# import matplotlib.pyplot as plt     # Drawing
import networkx as nx               # Constructing and visualizing graph
import numpy as np                  # Numerical methods
import os                           # File reading and checking
import re                           # Output formatting
import sys                          # Command line argument parsing
import timeit                       # Timing
import yaml                         # YAML parsing
import random
import time
from datetime import datetime       # Capture current time


# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================
SHOCK_MEAN = 0
SHOCK_SD = 0.01
# BARABASI_EDGE_FACTOR = 5
SHOCK_PROB = 0.2
# GRAPH_TOPOLOGY_NAME = ["random", "barabasi_albert", "watts_strogatz", "star"]
INITIAL_ADOPTER_GENERATOR = ["greedy", "degree", "influence", "discounter_degree"]
# WATTS_STROGATZ_REWIRE_FACTOR = 0.2


# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

# These are parameters provided by the user.
num_nodes = 0
prob_of_initial = 0
graphs = []

initial_thresholds = []
initial_states = []
edge_info = []
agent_state = []
agent_thresholds = []
num_initial_adopter = 0
adopter_generation_time = [0, 0, 0, 0]


def initial_adopter_selection_by_degree(graph_index):
    global edge_info, num_initial_adopter

    if (num_initial_adopter == 0): return [0]*num_nodes

    node_degree = [0] * num_nodes

    for node in range(num_nodes):
        for neighbor in range(num_nodes):
            if (edge_info[graph_index][node][neighbor] != 0):
                node_degree[node] = node_degree[node] + 1

    node_degree_copy = node_degree * 1

    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if (node_degree[i] < node_degree[j]):
                temp = node_degree[i]
                node_degree[i] = node_degree[j]
                node_degree[j] = temp
    lowest_degree = node_degree[num_initial_adopter-1]

    initial_adopter_by_degree = [0] * num_nodes
    for node in range(num_nodes):
        if (node_degree_copy[node] >= lowest_degree):
            initial_adopter_by_degree[node] = 1
        if (sum(initial_adopter_by_degree) == num_initial_adopter): break

    return initial_adopter_by_degree





def initial_adopter_selection_by_influence(graph_index):
    global edge_info, num_initial_adopter

    if (num_initial_adopter == 0): return [0]*num_nodes

    node_influence = []
    for node in range(num_nodes):
        node_influence.append(sum(edge_info[graph_index][node]))
    node_influence_copy = node_influence*1
    for i in range(num_nodes):
        for j in range(i, num_nodes):
            if (node_influence[i] < node_influence[j]):
                temp = node_influence[i]
                node_influence[i] =node_influence[j]
                node_influence[j] = temp
    lowest_influence = node_influence[num_initial_adopter-1]

    initial_adopter_by_influence = [0] * num_nodes
    for node in range(num_nodes):
        if (node_influence_copy[node] >= lowest_influence):
            initial_adopter_by_influence[node] = 1
        if (sum(initial_adopter_by_influence) == num_initial_adopter): break

    return initial_adopter_by_influence




def run_til_eq(graph_index, state, node_to_try, dynamic_threshold):

    global edge_info, num_nodes, edge_info

    state_copy = state * 1
    state_copy[node_to_try] = 1
    dynamic_threshold_copy = dynamic_threshold * 1
    dynamic_threshold_copy[node_to_try] = 0

    new_state = state_copy * 1

    while 1:
        for node in range(num_nodes):
            if (state_copy[node] == 1): continue
            influence_from_neighbor = 0
            for neighbor in range(num_nodes):
                influence_from_neighbor = influence_from_neighbor + state_copy[neighbor] * edge_info[graph_index][neighbor][node]
            if(influence_from_neighbor >= dynamic_threshold_copy[node]): new_state[node] = 1

        if(sum(new_state) == sum(state_copy)): break
        state_copy = new_state * 1

    return sum(new_state)-sum(state)





def initial_adopter_selection_greedy(graph_index):
    global initial_thresholds, num_nodes, edge_info

    if (num_initial_adopter == 0): return [0]*num_nodes

    dynamic_threshold = initial_thresholds * 1
    num_converted = [-1] * num_nodes
    greedy_optimal = [0] * num_nodes

    state = [0] * num_nodes


    while ((sum(greedy_optimal) != num_initial_adopter) and (sum(state) != num_nodes)):
        for node in range(num_nodes):
            if (state[node] != 1):
                num_converted[node] = run_til_eq(graph_index, state, node, dynamic_threshold)
        index = num_converted.index(max(num_converted))
        dynamic_threshold[index] = 0
        num_converted = [-1] * num_nodes
        greedy_optimal[index] = 1
        state[index] = 1

        new_state = state * 1

        while 1:
            for node in range(num_nodes):
                if (state[node] == 1): continue
                influence_from_neighbor = 0
                for neighbor in range(num_nodes):
                    influence_from_neighbor = influence_from_neighbor + state[neighbor] * edge_info[graph_index][neighbor][node]
                if(influence_from_neighbor >= dynamic_threshold[node]): new_state[node] = 1

            if(sum(new_state) == sum(state)):
                state = new_state * 1
                break

            state = new_state * 1

    return greedy_optimal



def initial_adopter_selection_by_discounter_degree(graph_index):
    global num_nodes, num_initial_adopter, edge_info

    if (num_initial_adopter == 0): return [0]*num_nodes

    discounted_degree_optimal = [0] * num_nodes

    initial_node_degree = [0] * num_nodes

    for node in range(num_nodes):
        for neighbor in range(num_nodes):
            if (edge_info[graph_index][node][neighbor] != 0):
                initial_node_degree[node] = initial_node_degree[node] + 1

    adopter = 0

    while (adopter < num_initial_adopter):
        max_index = initial_node_degree.index(max(initial_node_degree))
        discounted_degree_optimal[max_index] = 1
        initial_node_degree[max_index] = 0
        for neighbor in range(num_initial_adopter):
            if (edge_info[graph_index][neighbor][node] != 0 and discounted_degree_optimal[neighbor] != 1):
                initial_node_degree[neighbor] = initial_node_degree[neighbor] - 1
        adopter = adopter + 1
        if (sum(initial_node_degree) <= 0):
            break

    node = 0
    while (adopter < num_initial_adopter and node < num_nodes):
        if discounted_degree_optimal[node] == 0:
            discounted_degree_optimal[node] = 1
            adopter = adopter + 1
        node = node + 1

    return discounted_degree_optimal





def find_initial_adopter(graph_index):
    global initial_states, adopter_generation_time
    initial_states = []
    timer = time.time()
    initial_states.append(initial_adopter_selection_greedy(graph_index))
    timer = time.time() - timer
    adopter_generation_time[0] = timer
    timer = time.time()
    initial_states.append(initial_adopter_selection_by_degree(graph_index))
    timer = time.time() - timer
    adopter_generation_time[1] = timer
    timer = time.time()
    initial_states.append(initial_adopter_selection_by_influence(graph_index))
    timer = time.time() - timer
    adopter_generation_time[2] = timer
    timer = time.time()
    initial_states.append(initial_adopter_selection_by_discounter_degree(graph_index))
    timer = time.time() - timer
    adopter_generation_time[3] = timer


def find_equilibrium(graph_index, round_num):

    global num_nodes, graphs, edge_info, agent_state

    new_state = agent_state * 1
    iteration = 0
    max_iter = 2**num_nodes

    while iteration < max_iter:
        iteration = iteration + 1
        for node in range(num_nodes):
            influence_from_neighbor = 0
            for neighbor in range(num_nodes):
                influence_from_neighbor = influence_from_neighbor + edge_info[graph_index][neighbor][node] * agent_state[neighbor]
            if (influence_from_neighbor >= agent_thresholds[node]): new_state[node] = 1
            else: new_state[node] = 0
        if np.array_equal(new_state, agent_state): break
        else:
            agent_state = new_state



def simulate_next_shock(graph_index, round_num):
    global num_nodes, edge_info, graphs, agent_state, agent_thresholds
    shock_value = np.random.uniform(-1, 1, 1)
    shocked_agent = np.random.binomial(1, SHOCK_PROB, num_nodes)


    agent_thresholds = agent_thresholds + shock_value * (agent_thresholds - agent_thresholds * agent_thresholds) * shocked_agent

    find_equilibrium(graph_index, round_num)




def main():

    global num_nodes, num_initial_adopter, edge_info, initial_states, initial_thresholds, graphs, agent_state, agent_thresholds

    argv = sys.argv
    argc = len(argv)

    # processing command line input
    if (argc == 2):
        num_nodes = int(argv[1])

    else:
        print("Error: Invalild command line inputs.")
        print("Correct command line format: new_model_20171115.py <num_nodes>")
        sys.exit()

    print("There are {} agents in the game.".format(num_nodes))

    #   generate random graphs
    random.seed(None)
    BARABASI_EDGE_FACTOR = 5

    # record essential information
    # name files using current time
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')


    adopter_record_greedy = open("barabasi_{}_{}_greedy_adopter_hist".format(BARABASI_EDGE_FACTOR,num_nodes), "w")
    adopter_record_influence = open("barabasi_{}_{}_influence_adopter_hist".format(BARABASI_EDGE_FACTOR,num_nodes), "w")
    adopter_record_degree = open("barabasi_{}_{}_degree_adopter_hist".format(BARABASI_EDGE_FACTOR,num_nodes), "w")
    adopter_record_DD = open("barabasi_{}_{}_DD_adopter_hist".format(BARABASI_EDGE_FACTOR,num_nodes), "w")


    graph_num_trial = 40

    for graph_num in range(graph_num_trial):

        barabasi_graph = nx.barabasi_albert_graph(num_nodes, BARABASI_EDGE_FACTOR).to_directed()

        graphs = [barabasi_graph]

        edge_in_graph = [[0 for x in range(num_nodes)] for y in range(num_nodes)]

        for node in range(num_nodes):

            in_degree = barabasi_graph.in_degree(node)

            if not in_degree:
                continue

            # calculate the total weight of influence received from neighbors
            total_weight = np.random.uniform(0, 1, 1)
            edge_weights = np.random.uniform(0, 1, in_degree)
            edge_weights_sum = sum(edge_weights)
            # normalizing weights
            edge_weights = edge_weights/edge_weights_sum*total_weight

            for neighbor_index, neighbor in enumerate(barabasi_graph.predecessors(node)):
                edge_in_graph[neighbor][node] = edge_weights[neighbor_index]

        edge_info = [edge_in_graph]

        for total_initial_adopters in range(int(num_nodes * 0.1)):

            num_initial_adopter = total_initial_adopters+1

            # generate initial agent thresholds
            # format: array of real numbers between 0 and 1
            # indicating the magnitude of threshold
            initial_thresholds = np.random.uniform(0, 1, num_nodes)


            # generating an initial state for all agents
            # format: array of 0 and 1 indicating the intial adoption decision
            # 1 for adopters and 0 otherwise
            # initial_state = np.random.binomial(1, prob_of_initial, num_nodes)
            find_initial_adopter(0)


            for initial_adopter_approach_index, initial_adopter_approach_name in enumerate(INITIAL_ADOPTER_GENERATOR):
#                print(initial_adopter_approach_name)


                initial_state = initial_states[initial_adopter_approach_index] * 1
                agent_thresholds = initial_thresholds * 1

                for i in range(num_nodes):
                    if (initial_state[i] == 1):
                        agent_thresholds[i] = 0


                # agent_state = initial_states[initial_adopter_approach_index] * 1
                agent_state = initial_state * 1
                agent_thresholds = initial_thresholds * 1

#                print(num_initial_adopter)
#                print(sum(initial_state))

                for i in range(num_nodes):
                    if (initial_state[i] == 1):
                        agent_thresholds[i] = 0

                timer = time.time()
                find_equilibrium(0, 0)
                timer = time.time() - timer

                if (initial_adopter_approach_index == 0):
                    adopter_record_greedy.write("{:.5f} {} {} {:.5f}".format(adopter_generation_time[0], num_initial_adopter, sum(agent_state), timer))
                    adopter_record_greedy.write("\n")
                elif (initial_adopter_approach_index == 1):
                    adopter_record_degree.write("{:.5f} {} {} {:.5f}".format(adopter_generation_time[1], num_initial_adopter, sum(agent_state), timer))
                    adopter_record_degree.write("\n")
                elif (initial_adopter_approach_index == 2):
                    adopter_record_influence.write("{:.5f} {} {} {:.5f}".format(adopter_generation_time[2], num_initial_adopter, sum(agent_state), timer))
                    adopter_record_influence.write("\n")
                elif (initial_adopter_approach_index == 3):
                    adopter_record_DD.write("{:.5f} {} {} {:.5f}".format(adopter_generation_time[3], num_initial_adopter, sum(agent_state), timer))
                    adopter_record_DD.write("\n")

        #print("\n\n\n")

    adopter_record_greedy.close()
    adopter_record_influence.close()
    adopter_record_degree.close()
    adopter_record_DD.close()


main()
