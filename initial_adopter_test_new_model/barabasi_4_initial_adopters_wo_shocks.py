#By Sally
#1. Apply shocks to only a subset of nodes
#2. Can control iterations by pressing enter
#3. Recording more info

#!/usr/bin/env python

# ==============================================================================
# LIBRARIES
# ==============================================================================
import networkx as nx               # Constructing and visualizing graph
import numpy as np                  # Numerical methods
import os                           # File reading and checking
import sys                          # Command line argument parsing
import random
import time
from datetime import datetime       # Capture current time


# ==============================================================================
# GLOBAL CONSTANTS
# ==============================================================================

# WHEN ADDING NEW GRAPH TOPOLOGIES: Add new entry to class below, add function call in find_initial_adopter()
class Topologies:
    random = "random"
    barabasi = "barabasi_albert"
    watts = "watts_strogatz"
    star = "star"

# WHEN ADDING NEW HEURISTICS: Add new entry to class below, add function call in find_initial_adopter()
class Heuristics:
    greedy = "greedy"
    deg = "degree"
    inf = "influence"
    dd = "discounted_degree"
    gdd = "greedy_disc_degree"

HEUR = Heuristics()
TOP = Topologies()
GRAPH_NUM_TRIAL = 40
BARABASI_EDGE_FACTOR = 5
GRAPH_TOPOLOGY_NAME = [TOP.barabasi, TOP.watts, TOP.random, TOP.star]
INITIAL_ADOPTER_GENERATOR = [HEUR.dd, HEUR.greedy, HEUR.gdd]
WATTS_STROGATZ_REWIRE_FACTOR = 0.2
WATTS_STROGATZ_EDGE_FACTOR = 4
RANDOM_EDGE_FACTOR = 4


# ==============================================================================
# GLOBAL VARIABLES
# ==============================================================================

# These are parameters provided by the user.
num_nodes = 0
prob_of_initial = 0
graphs = {}

initial_thresholds = []
initial_states = {}
edge_info = {}
agent_state = []
agent_thresholds = []
num_initial_adopter = 0
adopter_generation_time = {}


# Compartmentalizes heuristic selection so each type runs iff it's in the list
def find_initial_adopter(graph_index, initial_adopter_approach):
    if initial_adopter_approach == HEUR.greedy:
        return initial_adopter_selection_greedy(graph_index)
    elif initial_adopter_approach == HEUR.deg:
        return initial_adopter_selection_by_degree(graph_index)
    elif initial_adopter_approach == HEUR.inf:
        return initial_adopter_selection_by_influence(graph_index)
    elif initial_adopter_approach == HEUR.dd:
        return initial_adopter_selection_by_discounted_degree(graph_index)
    elif initial_adopter_approach == HEUR.gdd:
        return initial_adopter_selection_greedy_discounted_degree(graph_index)


# Compartmentalizes graph creation so each type runs iff it's in the list
def make_graph(graph_type):
    global num_nodes, graphs
    if graph_type == TOP.barabasi:
        graphs[graph_type] = nx.barabasi_albert_graph(num_nodes, BARABASI_EDGE_FACTOR).to_directed()
    elif graph_type == TOP.random:
        graphs[graph_type] = nx.random_regular_graph(RANDOM_EDGE_FACTOR, num_nodes).to_directed()
    elif graph_type == TOP.watts:
        graphs[graph_type] = nx.watts_strogatz_graph(num_nodes, WATTS_STROGATZ_EDGE_FACTOR, WATTS_STROGATZ_REWIRE_FACTOR).to_directed()
    elif graph_type == TOP.star:
        graphs[graph_type] = nx.star_graph(num_nodes-1).to_directed()


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
                node_influence[i] = node_influence[j]
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
    global initial_thresholds, num_nodes, edge_info, num_initial_adopter

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


def initial_adopter_selection_by_discounted_degree(graph_index):
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
        for neighbor in range(num_nodes):
            if (edge_info[graph_index][neighbor][node] != 0 and discounted_degree_optimal[neighbor] != 1):
                initial_node_degree[neighbor] = initial_node_degree[neighbor] - 1
        adopter = adopter + 1
        if (sum(initial_node_degree) <= 0):
            break

    # Error catcher: select first n nodes iff above loop breaks (0 or negative degree)
    node = 0
    while (adopter < num_initial_adopter and node < num_nodes):
        if discounted_degree_optimal[node] == 0:
            discounted_degree_optimal[node] = 1
            adopter = adopter + 1
        node = node + 1

    return discounted_degree_optimal


# New hybrid algorithm: mix of greedy (with cascades) and discounted degree
def initial_adopter_selection_greedy_discounted_degree (graph_index):
    global initial_thresholds, num_nodes, edge_info, num_initial_adopter

    if (num_initial_adopter == 0): return [0]*num_nodes

    dynamic_threshold = initial_thresholds * 1
    num_converted = [-1] * num_nodes
    greedy_discounted_optimal = [0] * num_nodes

    state = [0] * num_nodes

    initial_node_degree = [0] * num_nodes

    # Calculate initial degrees
    for node in range(num_nodes):
        for neighbor in range(num_nodes):
            if (edge_info[graph_index][node][neighbor] != 0):
                initial_node_degree[node] = initial_node_degree[node] + 1

    while ((sum(greedy_discounted_optimal) != num_initial_adopter) and (sum(state) != num_nodes)):
        # Selects maximum degree node in terms of unselected nodes
        max_index = initial_node_degree.index(max(initial_node_degree))
        greedy_discounted_optimal[max_index] = 1
        initial_node_degree[max_index] = 0

        # Calculate equilibrium
        while 1:
            new_state = state * 1

            for node in range(num_nodes):
                if (state[node] == 1): continue

                # Calculate influence from neighbour
                influence_from_neighbor = 0
                for neighbor in range(num_nodes):
                    influence_from_neighbor = influence_from_neighbor + state[neighbor] * edge_info[graph_index][neighbor][node]

                # If influence is high enough, switch action and propogate degree changes
                if(influence_from_neighbor >= dynamic_threshold[node]):
                    new_state[node] = 1
                    for neighbor in range(num_nodes):
                        if (edge_info[graph_index][neighbor][node] != 0 and greedy_discounted_optimal[neighbor] != 1):
                            initial_node_degree[neighbor] = initial_node_degree[neighbor] - 1

            # While condition of simulated do-while loop - if nothing changed
            if(sum(new_state) == sum(state)):
                break

    return greedy_discounted_optimal


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


def main():

    global num_nodes, num_initial_adopter, edge_info, initial_states, initial_thresholds, graphs, agent_state, agent_thresholds

    argv = sys.argv
    argc = len(argv)

    # processing command line input
    if (argc == 2):
        num_nodes = int(argv[1])

    else:
        temp_nodes = int(input("Enter number of players: "))
        if (type(temp_nodes) == int and temp_nodes >= 10):
            num_nodes = temp_nodes
        else:
            print("Error: not a valid input!")
            print("Input must be an integer >= 10")
            sys.exit()

    print("There are {} agents in the game.".format(num_nodes))

    #   generate random graphs
    random.seed(None)

    # record essential information
    # name files using current time
    current_time = datetime.now().strftime('%Y_%m_%d_%H_%M_%S')

    # initialize the dictionary of solutions (replacement for adopter_record_xxx variables)
    soln_dict = {}

    for graph_num in range(GRAPH_NUM_TRIAL):

        for graph_type in GRAPH_TOPOLOGY_NAME:
            make_graph(graph_type)
         
            # initializes graph's solution type dictionary
            if graph_type not in soln_dict:
                soln_dict[graph_type] = {}

        edge_in_graph = [[0 for x in range(num_nodes)] for y in range(num_nodes)]

        for graph_type, graph in graphs.items():

            for node in range(num_nodes):

                in_degree = graph.in_degree(node)

                if not in_degree:
                    continue

                # calculate the total weight of influence received from neighbors
                total_weight = np.random.uniform(0, 1, 1)
                edge_weights = np.random.uniform(0, 1, in_degree)
                edge_weights_sum = sum(edge_weights)
                # normalizing weights
                edge_weights = edge_weights/edge_weights_sum*total_weight

                for neighbor_index, neighbor in enumerate(graph.predecessors(node)):
                    edge_in_graph[neighbor][node] = edge_weights[neighbor_index]

            edge_info[graph_type] = edge_in_graph

            for total_initial_adopters in range(int(num_nodes * 0.1)):

                num_initial_adopter = total_initial_adopters+1

                # generate initial agent thresholds
                # format: array of real numbers between 0 and 1
                # indicating the magnitude of threshold
                initial_thresholds = np.random.uniform(0, 1, num_nodes)

                for initial_adopter_approach in INITIAL_ADOPTER_GENERATOR:
                    # select inital adopters and time selection process
                    timer = time.time()
                    initial_states[initial_adopter_approach] = find_initial_adopter(graph_type, initial_adopter_approach)
                    timer = time.time() - timer
                    adopter_generation_time[initial_adopter_approach] = timer

                    initial_state = initial_states[initial_adopter_approach] * 1
                    agent_thresholds = initial_thresholds * 1

                    for i in range(num_nodes):
                        if (initial_state[i] == 1):
                            agent_thresholds[i] = 0

                    agent_state = initial_state * 1
                    agent_thresholds = initial_thresholds * 1

                    for i in range(num_nodes):
                        if (initial_state[i] == 1):
                            agent_thresholds[i] = 0

                    # perform propogation and time propogation process
                    timer = time.time()
                    find_equilibrium(graph_type, 0)
                    timer = time.time() - timer

                    # initializes solution type's solution information array
                    if initial_adopter_approach not in soln_dict[graph_type]:
                        soln_dict[graph_type][initial_adopter_approach] = []

                    soln_dict[graph_type][initial_adopter_approach].append([])

                    info = (sum(agent_state), adopter_generation_time[initial_adopter_approach], timer, initial_state)

                    soln_dict[graph_type][initial_adopter_approach][num_initial_adopter - 1].append(info)


    # Handles recorded information and writes to csv files
    adopter_record = open("run-{}-{}.csv".format(num_nodes,current_time), "w")
    for graph_type, graph_info in soln_dict.items():
        for initial_adopter_approach, content in graph_info.items():
            for num_adopters, num_adopter_arr in enumerate(content):
                for graph_num, info in enumerate(num_adopter_arr):
                    adopter_record.write("{},{},{},{},{},{:.5f},{:.5f},{}".format(
                      graph_type,
                      graph_num + 1,
                      initial_adopter_approach,
                      num_adopters + 1,
                      info[0],
                      info[1],
                      info[2],
                      info[3]))
                    adopter_record.write("\n")

    adopter_record.close()


main()
