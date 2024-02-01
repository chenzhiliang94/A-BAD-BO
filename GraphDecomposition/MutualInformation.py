import copy
import random

from GraphDecomposition.DirectedFunctionalGraph import DirectedFunctionalGraph
from Models.Model import Model

import numpy as np
from sklearn.feature_selection import mutual_info_regression

def mutual_information_nodes_samples(DG : DirectedFunctionalGraph) -> dict:
    
    local_loss_samples = {}
    
    # for each node in DG, perform grad descent, until convergence (need a function to check for convergence)
    all_components = DG.get_components()
    for node_name in all_components.keys():
        component:Model = all_components[node_name]["component"] 
        mapping_between_loss_to_param = {}
        
        for step in range(100): # until convergence, how to check
            #print(DG.get_all_params()[1])
            component.do_one_descent_on_local(0.001)
            mapping_between_loss_to_param[component.get_local_loss()] = component.get_params()
        
        local_loss_samples[node_name] = mapping_between_loss_to_param
    
    return local_loss_samples
    
    # # sample from losses and params to get system loss
    # # for node 1
    # for x in range(100):
    #     system_param_sample = []

def get_mi(dg : DirectedFunctionalGraph):
    param_original = copy.deepcopy(dg.get_all_params()[1])
    samples = mutual_information_nodes_samples(dg)
    dg.assign_params(param_original)

    all_local_losses = []
    all_system_losses = []
    for x in range(100):
        l = np.array(dg.get_local_losses()).flatten()
        all_local_losses.append(l)
        L = dg.get_system_loss()
        all_system_losses.append(L)
        for node_name in samples:
            local_node_loss = samples[node_name].keys()
            sampled_local_loss = random.sample(list(local_node_loss), 1)[0]
            sampled_param = samples[node_name][sampled_local_loss]
            dg.assign_param_to_node(node_name, sampled_param)
        

    all_local_losses = np.stack(all_local_losses)
    mi = mutual_info_regression(all_local_losses, all_system_losses, n_neighbors=100)
    return mi