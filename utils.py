
import numpy as np
import networkx as nx
import os
from collections import defaultdict
import nxmetis


def agile_HDA(G, B):

    
    degrees = dict(G.degree())
    degree_buckets = defaultdict(set)
    max_degree = 0

    for vertex, degree in degrees.items():
        degree_buckets[degree].add(vertex)
        max_degree = max(max_degree, degree)

    # Initialize the ordering list
    ordering = []
    for _ in range(B):
        # Find the largest non-empty degree bucket
        found_bucket = False
        for degree in range(max_degree, 0, -1):
            if degree_buckets[degree]:
                found_bucket = True
                break
        if(not found_bucket):
            break
        
        max_degree = degree
        
        # Remove a vertex from the bucket and update degrees
        max_degree_vertex = degree_buckets[degree].pop()
        ordering.append(max_degree_vertex)

        degrees[max_degree_vertex] = -1
        # Update the degree of neighboring vertices
        for neighbor in list(G.neighbors(max_degree_vertex)):
            #if neighbour still exists in the network
            if degrees[neighbor] > 0:
                old_degree = degrees[neighbor]
                new_degree = old_degree - 1
                degrees[neighbor] = new_degree

                degree_buckets[old_degree].remove(neighbor)
                degree_buckets[new_degree].add(neighbor)

        # Remove the vertex from the graph
        G.remove_node(max_degree_vertex)


    return ordering


def calc_R0(G, t=1, only_gcc=False, increment_degree0=None):
    if(only_gcc):
        G = G.subgraph(max(nx.connected_components(G), key=len))

    degrees = np.array(G.degree)[:, 1]
    unique_degrees, degree_freq = np.unique(degrees, return_counts=True)

    unique_degrees = unique_degrees.astype(float)
    if(increment_degree0):
        if(0 in unique_degrees):
            zero_degree_index = np.where(unique_degrees == 0)
            unique_degrees[zero_degree_index] += 0.1

    dd = degree_freq/sum(degree_freq)
    m = sum(unique_degrees*dd)
    if(m-1 <= 0):
        return 0
    s = sum(unique_degrees**2*dd)
    r = t*(s/m-1)

    return r


def get_2hop_node(G, node):
    one_hop = set(G.neighbors(node))
    two_hop = set()
    
    for n1 in one_hop:
        neighbors = set(G.neighbors(n1))
        two_hop.update(neighbors)
        #add the node itself

    two_hop.update(one_hop)
    two_hop.add(node)

    return two_hop

def get_total_2hop_connectivity(G):
    nodes = set(G.nodes)
    total = 0
    
    for node in nodes:
        two_hop = get_2hop_node(G, node)
        total += len(two_hop) - 1

    total = total / 2
    
    return total

def get_k_hop_connectivity(G, k=2):
    k_hop_connectivity_sum = 0
    for node in G.nodes:
        curr_connectivity = nx.single_source_dijkstra_path_length(
            G, node, cutoff=k)
        k_hop_connectivity_sum += len(curr_connectivity) - 1

    return k_hop_connectivity_sum/2


def sol_to_txt(sol_arr, export_path):
    sol_np = np.array(sol_arr, dtype=int).reshape(-1)
    np.savetxt(export_path, sol_np, fmt="%i", delimiter=',')

def make_dir(path):
    try:
        os.makedirs(path)
    except:
        return -1
    

def calc_graph_connectivity(G, experiment_type, T=1):
    if(G.number_of_nodes() in [0, 1]):
        return 0
    N = G.number_of_nodes()
    _CN_denom = N * (N - 1)/2
    if(experiment_type == "CN"):
        pairwise_connectivity = 0
        for i in list(nx.connected_components(G)):
            pairwise_connectivity += (len(i) * (len(i) - 1)) / 2
        pc = pairwise_connectivity / _CN_denom
        return pc
    elif(experiment_type == "GCC"):
        maxCC = len(max(nx.connected_components(G), key=len))
        #print(maxCC , _g_num_nodes)
        # return maxCC / _g_num_nodes
        return maxCC
    print("experiment type not impelemented")
    return -1

def fix_simplicials(G):
    simplicials = []
            
    for vertex in G.nodes():
        neighbor_set = list(G.neighbors(vertex))
        induced_subgraph = G.subgraph(neighbor_set)
        n = induced_subgraph.number_of_nodes()
        m = induced_subgraph.number_of_edges()
        if m == n*(n-1)/2: simplicials.append(vertex)
    
    subgraph_of_simplicials = G.subgraph(simplicials)
    fixed_simplicial_nodes = []
    #iterate over connected components
    for component in nx.connected_components(subgraph_of_simplicials):
        #get first node of component
        first_node = list(component)[0]
        fixed_simplicial_nodes.append(first_node)

    #print("Number of independent simplicials: ", nx.number_connected_components(subgraph_of_simplicials))
    print("Number of independent simplicials: ", len(fixed_simplicial_nodes))

    return fixed_simplicial_nodes

def partition_G(G, num_partition):
    G.node = G.nodes
    partitions = nxmetis.partition(G, num_partition)
    print("Partitions calculated")
    # number of nodes to remove

    for partition in range(num_partition):
        #H = G.subgraph(partitions[1][partition])
        # partition vertices
        for vertex in partitions[1][partition]:
            G.nodes[vertex]["partition"] = partition

    return G, partitions

