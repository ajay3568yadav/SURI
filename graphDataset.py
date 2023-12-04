import re
import os
import sys
import pickle
sys.path.append("classes")
from pin import Pin
from delayobject import DelayObject
from node import Node
import numpy as np

# Load node dictionary from a pickle file
with open('../newPickle/node_dictionary.pkl', 'rb') as file:
    node_dictionary = pickle.load(file)

# Initialize an empty graph dictionary
graph = {}

# Build a directed graph based on the node inputs
for node in node_dictionary:
    node_object = node_dictionary[node][0]  # Object node with above properties
    inputs_list = node_object.inputs  # Inputs going into the node
    graph[node_object] = []
    
    for inputs in inputs_list:
        if inputs in node_dictionary:
            o_object = node_dictionary[inputs][0]  # Input of node_object object
            if o_object not in graph:
                graph[o_object] = []
            graph[o_object].append(node_object)  # Node dictionary[key] maps to a list, hence [0]

# Update the fanout property of each node in the graph
for node in graph:
    node.fanout = len(graph[node])

# Initialize an empty wire graph dictionary
wire_graph = {}

# Build a directed wire graph based on the node inputs
for node in node_dictionary:
    node_object = node_dictionary[node][0]
    wire_graph[node_object] = []
    inputs_list = node_object.inputs
    
    for inputs in inputs_list:
        if inputs == "1'b0" or inputs == "1'b1" or inputs[0] == "{":
            continue
        input_object = node_dictionary[inputs][0]
        if input_object not in wire_graph:
            wire_graph[input_object] = []
        wire_graph[input_object].append(node_object)

# Calculate load for nodes with a cell value
for node in wire_graph:
    if node.cell != "":
        node.load = sum([wire.maxcap for wire in wire_graph[node]])

# Additional points for power calculation
additional_points = np.linspace(0.9, 1.5, 1000).tolist()

# Function to calculate power given parameters
def get_power(value, peak, t1, t2, t):
    c = value + t1
    r2 = c + t2
    r1 = value
    m1 = peak / (c - r1)
    m2 = peak / (c - r2)
    
    if t <= c:
        return m1 * (t - r1)
    return m2 * (t - r2)

# Function to generate a time to power dictionary for a given node
def get_power_dict(node, additional_points):
    time_powerdict = {}
    newdict = node.toggle
    newdict_keys = list(newdict.keys())
    power_list = [] 
    
    for time in newdict_keys:
        power_list.append(time)  # Starting time
        power_list.append(time + node.t1 + node.t2)  # Ending time
    
    power_list = list(set(power_list + additional_points))  # Convert additional_points to a list before adding
    power_list.sort()
    
    for value in newdict_keys:
        if newdict[value] == 1:
            for time in power_list:
                power = get_power(value, node.peak, node.t1, node.t2, time)
                if power >= 0:
                    if time not in time_powerdict:
                        time_powerdict[time] = []
                    time_powerdict[time].append(power)
    
    for toggle_time in list(time_powerdict.keys()):
        time_powerdict[toggle_time] = sum(time_powerdict[toggle_time])
    
    return time_powerdict

# Create time to power value dictionary for each node
for node in wire_graph:
    node.powertimedict = get_power_dict(node, additional_points)

# Function to get input objects for a given node
def get_input_objects(node):
    li = []
    for input_node in node.inputs:
        if input_node[0] == "{" or input_node == "1'b1" or input_node == "1'b0":
            continue
        li.append(node_dictionary[input_node][0])
    return li

# Initialize a dictionary mapping nodes to their input objects
node_inputs_dictionary = {}

for wires in wire_graph:
    node_inputs_dictionary[wires] = get_input_objects(wires)

# Function to check if a node is a D flip-flop
def is_DFF(node):
    return node.cell.find("DFF") >= 0

# Function to get the state of a node
def get_state(node):
    return node_inputs_dictionary[node]

# Function to generate a subset of nodes starting from a given node
def generate_subset(start_node):
    def dfs(node):
        visited.add(node)
        if is_DFF(node):
            return
        for neighbor in get_state(node):
            if neighbor not in visited:
                dfs(neighbor)

    visited = set()
    dfs(start_node)

    return visited

# Create a list of subsets of nodes
datas = []
for x in node_inputs_dictionary:
    datas.append(generate_subset(x))

# Function to create a graph from a list of nodes
def graphify(nodes_list):
    graph = {}
    for node in nodes_list:
        if is_DFF(node):
            if node not in graph:
                graph[node] = []
            continue
        graph[node] = node_inputs_dictionary[node]
    return graph

# Create a list of graphs from the subsets of nodes
graph_dataset = []
for x in datas:
    graph_dataset.append(graphify(x))

# Save the graph dataset to a pickle file
with open('graph_dataset.pkl', 'wb') as file:
    pickle.dump(graph_dataset, file)
    
# Assuming additional_points is defined in your code
additional_points = np.linspace(0.9, 1.5, 1000).tolist()

# Pickle the additional_points variable
with open('additional_points.pkl', 'wb') as file:
    pickle.dump(additional_points, file)