import networkx as nx
import matplotlib.pyplot as plt
import itertools
import torch
import math

import copy

from GraphDecomposition.DirectedFunctionalGraph import *

def plot(G):
    pos = nx.spectral_layout(G)
    nx.draw(G, pos=pos, with_labels = True)
    plt.show()

# class DirectedFunctionalGraph(nx.DiGraph):
#     def __init__(self, incoming_graph_data=None, **attr):
#         super().__init__(incoming_graph_data, **attr)
#
#     def add_edge(self, u_of_edge, v_of_edge, **attr):
#         """Add an edges from each u to v.
#         Parameters
#         ----------
#         u_of_edge : Ordered list of input nodes
#
#         v_of_edge : node
#         """
#         if not (isinstance(u_of_edge, list) or isinstance(u_of_edge, tuple)):
#             u_of_edge = [u_of_edge]
#         required_inputs = self.nodes[v_of_edge]["component"].inputs
#         if required_inputs > 1:
#             assert len(u_of_edge) == required_inputs, f"node {v_of_edge} require {required_inputs} parents, only {len(u_of_edge)} supplied"
#             for u in u_of_edge:
#                 if not u is None:
#                     super().add_edge(u, v_of_edge, **attr)
#         else:
#             super().add_edge(u_of_edge[0], v_of_edge, **attr)
#         self.nodes[v_of_edge]["parents"] = u_of_edge
#
#     def forward(self, sources:dict, sink):
#         def backward_(node):
#             component = self.nodes[node]["component"]
#             if node in sources:
#                 x = sources[node]
#                 if not torch.is_tensor(x):
#                     x = torch.tensor(x)
#                 return component(x)
#             assert "parents" in self.nodes[node]
#             input = [backward_(parent) for parent in self.nodes[node]["parents"]]
#             input = torch.tensor(input)
#             return component(input)
#         return backward_(sink)

# get all decomposition based on a bfs expansion from a black box
def find_all_decomposition_bfs(all_black_box, G):
    def find_decomposition_from_bfs(bfs_edge_list):
        decompositions = []
        d = set()
        d.add(bfs_edge_list[0][0])  # the black box
        decompositions.append(d.copy())
        for edge in bfs_edge_list:
            d.add(edge[1])
            decompositions.append(d.copy())
        return decompositions

    decomposition = {}
    for black_box in all_black_box.copy():
        other_black_box = all_black_box.copy()
        other_black_box.remove(black_box)

        # remove other black box
        G_with_only_one_black_box = G.copy()
        for b in other_black_box:
            G_with_only_one_black_box.remove_node(b)

        bfs_edge = list(nx.bfs_edges(G_with_only_one_black_box.to_undirected(), source=black_box))
        d = find_decomposition_from_bfs(bfs_edge)
        decomposition[black_box] = d
    return decomposition

def find_all_decomposition_full(all_black_box, G):
    def get_all_connected_subgraphs(G):
        """Get all connected subgraphs by a recursive procedure"""

        con_comp = [c for c in sorted(nx.connected_components(G), key=len, reverse=True)]

        def recursive_local_expand(node_set, possible, excluded, results, max_size):
            """
            Recursive function to add an extra node to the subgraph being formed
            """
            results.append(node_set)
            if len(node_set) == max_size:
                return
            for j in possible - excluded:
                new_node_set = node_set | {j}
                excluded = excluded | {j}
                new_possible = (possible | set(G.neighbors(j))) - excluded
                recursive_local_expand(new_node_set, new_possible, excluded, results, max_size)

        results = []
        for cc in con_comp:
            max_size = len(cc)

            excluded = set()
            for i in G:
                excluded.add(i)
                recursive_local_expand({i}, set(G.neighbors(i)) - excluded, excluded, results, max_size)

        results.sort(key=len)

        return results

    decomposition = {}
    for black_box in all_black_box.copy():
        other_black_box = all_black_box.copy()
        other_black_box.remove(black_box)

        # remove other black box
        G_with_only_one_black_box = G.copy()
        for b in other_black_box:
            G_with_only_one_black_box.remove_node(b)

        d = (get_all_connected_subgraphs(G_with_only_one_black_box.to_undirected()))
        d = [x for x in d if black_box in x] # the function returns all subgraph. We are only interested in those containing the black box
        decomposition[black_box] = d
    return decomposition

def get_all_valid_decomposition(black_box_decomp):
    #input: dictionary of {blackbox_name : list of all possible sets of decomposition}

    # get all possible combination
    all_decomposition = black_box_decomp.values()
    all_decomposition = itertools.product(*list(all_decomposition))
    result = list(all_decomposition)

    def is_valid(s):
        for x in s:
            if len(x) == 1:
                return False
        # ({1, 2, 4, 'Blackbox3'}, {2, 4, 'Blackbox5'}, {10, 7, 8, 9, 'Blackbox6'})
        for idx_set_one in range(len(s)):
            for idx_set_two in range(idx_set_one + 1, len(s)):
                if len(s[idx_set_one].intersection(s[idx_set_two]))!=0: # remove overlapping
                    return False
        return True
    # remove combination with overlapping components (might want to allow them in the future for future research challenges)
    all_decomposition = [x for x in result if len(set.intersection(*x))==0 and is_valid(x)]

    return all_decomposition

def goodness_measure(DG, decomposition, l=0.1, debug=False):
    total_error_support = 0
    total_error_leakage = 0
    max_param = 0
    print("decomposition: ", decomposition)
    for separated_part in decomposition:  # separated_part is a cluster of nodes
        print("cluster in decomposition: ", separated_part)
        # we reject one-sized cluster (since it is the black box comp)
        if len(separated_part) == 0:
            continue
        total_num_params_for_cluster = 0
        error_support_for_cluster = 0
        error_leakage_for_cluster = 0
        for node in separated_part:
            if "Blackbox" in str(node):
                continue
            total_num_params_for_cluster += len(DG.nodes[node]["component"].get_params())
            error_support_for_cluster += DG.nodes[node][
                                             "component"].lipschitz * 1  # assuming ball is 1, need to modify later possibly
            all_neighbours = DG.successors(node)
            all_neighbours = [x for x in all_neighbours if
                              x not in separated_part]  # avoid neighbours already in cluster
            for n in all_neighbours:
                error_leakage_for_cluster += DG.nodes[node][
                                                 "component"].lipschitz * 1  # assuming ball is 1, need to modify later possibly
        print("number of param in cluster: ", total_num_params_for_cluster)
        print("error support of cluster: ", error_support_for_cluster)
        print("error leakage for cluster: ", error_leakage_for_cluster)
        total_error_support += error_support_for_cluster
        total_error_leakage += error_leakage_for_cluster
        max_param = max(max_param, total_num_params_for_cluster)

    # S / (1 + L + \lambda * M), the magic formula
    score = (total_error_support) / (1 + (1 * total_error_leakage + l * math.sqrt(max_param)))
    print("decomposition score: ", score)
    return score

def goodness_measure_mutual_information(DG, decomposition):
    total_mutual_info = 0.0
    for separated_part in decomposition:  # separated_part is a cluster of 
        print("cluster in decomposition: ", separated_part)
        # we reject one-sized cluster (since it is the black box comp)
        total_mutual_info -= len(separated_part)/100
        if len(separated_part) == 0:
            continue

        for node in separated_part:
            if "Blackbox" in str(node):
                continue
            
            all_neighbours = DG.successors(node)
            all_neighbours = [x for x in all_neighbours if
                              x not in separated_part]  # avoid neighbours already in cluster
            for n in all_neighbours:
                total_mutual_info += DG.nodes[node]["mi"]  # assuming ball is 1, need to modify later possibly

    score = -total_mutual_info
    print("decomposition score: ", score)
    return score

def get_best_decomposition(decomposition, DG, l=0.1, measure = "mi"):
    # input: tuple of set

    best_decomposition = None
    best_score = -10000000
    for d in decomposition:
        if measure == "mi":
            score = goodness_measure_mutual_information(DG, d)
        elif measure == "lip":
            score = goodness_measure(DG,d,l)
        if score >= best_score:
            best_score = score
            best_decomposition = d
    return best_decomposition, best_score

def luke_partition_based_on_mi(DG : DirectedFunctionalGraph):
    # assume mi is attached to each DG already
    
    # step 1, for each node with multiple childen, remove edges until one left (chosen randomly, because each of the edge has same weight)
    graph =  copy.deepcopy(DG)
    
    # step 2, get dict of edge : mi from the DG (simple)
    
    # step 3, lukes_partitioning(G, max_size, node_weight=None, edge_weight=None)
    
    return
    
    