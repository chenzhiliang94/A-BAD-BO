import itertools
import copy
import time

import networkx as nx
import torch
import warnings
import torch.nn as nn
import numpy as np
from random import sample
from collections import OrderedDict
import matplotlib.pyplot as plt
from Models.ModelConstant import ModelConstant

class DirectedFunctionalGraph(nx.DiGraph):
    noise = 0.1
    noise_std = 0.05
    system_x = None
    system_y = None
    to_perturb = True
    def __init__(self, noise=0.3, incoming_graph_data=None, **attr):
        self.noise = noise
        super().__init__(incoming_graph_data, **attr)

    def add_edge(self, u_of_edge, v_of_edge, **attr):
        """Add an edges from each u to v.
        Parameters
        ----------
        u_of_edge : Ordered list of input nodes

        v_of_edge : node
        """
        if not (isinstance(u_of_edge, list) or isinstance(u_of_edge, tuple)):
            u_of_edge = [u_of_edge]
        if "parents" in self.nodes[v_of_edge]:
            warnings.warn(f'Parents of {v_of_edge} previously defined as {self.nodes[v_of_edge]["parents"]}, attempting to overwrite with {u_of_edge}')
            for edge in self.nodes[v_of_edge]["parents"]:
                self.remove_edge(edge, v_of_edge)
            self.nodes[v_of_edge].pop("parents")
        required_inputs = self.nodes[v_of_edge]["component"].inputs
        assert len(u_of_edge) == required_inputs, f"node {v_of_edge} require {required_inputs} parents, but {len(u_of_edge)} supplied"
        if required_inputs > 1:
            for u in u_of_edge:
                if u is not None:
                    print(f"adding edge from {u} to {v_of_edge}")
                    super().add_edge(u, v_of_edge, **attr)
        else:
            super().add_edge(u_of_edge[0], v_of_edge, **attr)
        self.nodes[v_of_edge]["parents"] = u_of_edge

    def forward(self, sources:dict, sink, perturbed_black_box=False):
        def backward_(node):
            component = self.nodes[node]["component"]
            if node in sources:
                
                x = sources[node]
                if component.inputs > 1:
                    for i in range(component.inputs):
                        if x[i] is None:    # Input not provided, query upwards
                            assert "parents" in self.nodes[node], f"Parents for node {node} required but not defined"
                            assert self.nodes[node]["parents"][i] is not None, f"Parent {i} for node {node} required but not defined"
                            x[i] = backward_(self.nodes[node]["parents"][i])
                if not torch.is_tensor(x):
                    x = torch.tensor(x)
                if isinstance(component, ModelConstant): 
                    return component(x,noisy=False)
                if "Blackbox" in str(node):
                    if perturbed_black_box:
                        return component(x,noisy=True, noise_mean=self.noise)
                    return component(x,noisy=False)
                return component(x,noisy=False)
            
            assert "parents" in self.nodes[node]
            input = []
            for i, parent in enumerate(self.nodes[node]["parents"]):
                assert parent is not None, f"Input {i} for node {node} is required but not provided"
                input.append(backward_(parent))
            input = torch.tensor(input)
            component = self.nodes[node]["component"]
            if isinstance(node, ModelConstant): 
                    return component(input,noisy=False)
            if "Blackbox" in str(node):
                if perturbed_black_box:
                    return component(input,noisy=True, noise_mean=self.noise, noise_std=self.noise_std)
                return component(input,noisy=True, noise_std=self.noise_std)
            return component(input, noisy=True, noise_std=self.noise_std)
        return backward_(sink)
    
    def generate_sub_system(self, node_set : set):
        extended_node_set = set()
        for n in node_set:
            extended_node_set.add(n)
            extended_node_set = extended_node_set | set(list(self.predecessors(n)))
        return extended_node_set
        
    def retain_nodes(self, node_set : set):
        all_nodes = copy.deepcopy(self.nodes)
        for n in all_nodes:
            if n not in node_set:
                self.remove_node(n)
                for node in self.nodes:
                    if "parents" not in self.nodes[node]:
                        continue
                    parents : list = self.nodes[node]["parents"]
                    if n in parents:
                        parents.remove(n)
                    self.nodes[node]["parents"] = parents

    def get_exit(self):
        exit = [n for n,d in self.out_degree() if d==0][0]
        return exit

    def get_entry(self):
        entry = [n for n,d in self.in_degree() if d==0]
        return entry
        
    def get_system_loss_with_inputs(self, X, y):
        mse = nn.MSELoss()
        y_pred = []
        exit = self.get_exit()
        entry = self.get_entry()
        for x in X:
            assert len(x) == len(entry)
            input_dict = {}
            for node_idx, input in zip(entry, x):
                input_dict[node_idx] = input
            y_pred.append(self.forward(input_dict, exit, perturbed_black_box=self.to_perturb))
        loss = mse(torch.tensor(y_pred), y)
        #print("MAE loss: ", nn.L1Loss()(torch.tensor(y_pred), y))
        return loss

    def get_system_loss(self):
        return self.get_system_loss_with_inputs(self.system_x, self.system_y)

    def get_system_loss_with_params(self, param):
        print(param)
        self.assign_params(param)
        return self.get_system_loss()
    
    def get_local_losses(self):
        losses = []
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n):
                continue
            losses.append(self.nodes[n]["component"].get_local_loss())
        return torch.DoubleTensor([losses])

    def get_components(self):
        components = OrderedDict((x, self.nodes[x]) for x in self.nodes if "Dummy" not in str(x) and "Blackbox" not in str(x))
        return components
    
    def get_all_params(self):
        dict_ = {}
        param = []
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n) or "nn" in str(n):
                continue
            dict_[n] = self.nodes[n]["component"].get_params()
            param += list(self.nodes[n]["component"].get_params())
        return dict_, param

    def assign_params(self, params : dict):
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n) or "nn" in str(n):
                continue
            self.nodes[n]["component"].set_params(params[n])

    def assign_param_to_node(self, node : str, param : list):
        self.nodes[node]["component"].set_params(param)

    def assign_params(self, params : list):
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n) or "nn" in str(n):
                continue
            num_param_to_assign = len(self.nodes[n]["component"].get_params())
            self.nodes[n]["component"].set_params(params[:num_param_to_assign])
            params = params[num_param_to_assign:]

    # look for parameters which yield the input losses
    def reverse_local_loss_lookup(self, losses, method, samples, percentage_threshold_for_terminate_search=1e-01, to_plot=True, num_starting_points=5, ignore_failure=False):
        components = self.get_components().values()
        assert len(components) == len(losses), "loss input size should be equals to number of components!"

        if method == "nn_lookup":
            params = []
            num_starting_points = num_starting_points
            sample_size = num_starting_points
            final_sample = samples
            timeA = time.time()
            for loss_comp in zip(components, losses):
                param_candidates = []
                for n in range(num_starting_points):
                    component = loss_comp[0]["component"]
                    loss_target = loss_comp[1]
                    
                    re_initialise_count = 0
                    component.random_initialize_param()
                    while component.get_local_loss() > 100:
                        re_initialise_count +=1
                        component.random_initialize_param()
                        if re_initialise_count > 30:
                            break
                    
                    method = getattr(component, 'nn_function', None)
                    if callable(method):
                        temp_nn_model = copy.deepcopy(component)
                        component.descent_to_target_loss(loss_target)

                        temp_nn_model = copy.deepcopy(component.conv_model)
                        param_candidates.append(temp_nn_model)
                    else:
                        l = []
                        #print("running reverse loss lookup on other models")
                        itr = 0
                        while abs(component.get_local_loss() - loss_target) / loss_target > percentage_threshold_for_terminate_search: # threshold
                            curr_loss = component.get_local_loss()
                            l.append(curr_loss)
                            if curr_loss > loss_target:
                                component.do_one_descent_on_local()
                            else:
                                component.do_one_ascent_on_local() # can replace this simply with another custom loss function
                            itr+=1
                            if itr > 2000:
                                break
                        if not ignore_failure:
                            if abs(component.get_local_loss() - loss_target) / loss_target > 1e-01 and abs(component.get_local_loss() - loss_target) > 0.01:
                                continue
                        if list(component.get_params()) not in param_candidates:
                            param_candidates.append(list(component.get_params()))
                if len(param_candidates) == 0:
                    print(l[-5:])
                    print("loss target: ", loss_target)
                    print("loss reached: ", component.get_local_loss())
                    print("CANNOT FIND PARAMS, THROW")
                    raise Exception("cannot find param from grad descent error")
                while len(param_candidates) < sample_size:
                    param_candidates = param_candidates + param_candidates
                params.append(sample(param_candidates,sample_size)) # sample
            timeB = time.time()
            print("time taken for loss recovery: ", timeB - timeA)
            all_idx_combination = len(params) * [[x for x in range(sample_size)]]
            all_cartesian_idx = list(itertools.product(*all_idx_combination))
            a = len(list(all_cartesian_idx))
            b = final_sample
            min_to_sample = min(a, b)
            all_cartesian_idx = sample(all_cartesian_idx, min_to_sample) # sample again
            
            best_system_loss = 1e50
            best_param = None
            candidate_loss_all = []
            print(len(all_cartesian_idx))
            for idx in all_cartesian_idx: # iterate through randomly selected idx (3, 2, 5)
                candidate_param = []
                for i, cartesian_idx in enumerate(list(idx)): # i is the component index, cartesian index is which to choose

                    if list(self.get_components().items())[i][1]["component"].is_nn():
                        list(self.get_components().items())[i][1]["component"].conv_model = params[i][cartesian_idx]
                    else:
                        list(self.get_components().items())[i][1]["component"].set_params(params[i][cartesian_idx])
                candidate_system_loss = self.get_system_loss()
                candidate_loss_all.append(candidate_system_loss)
                if candidate_system_loss < best_system_loss:
                    best_system_loss = candidate_system_loss
                    best_param = candidate_param
            
            if to_plot:
                plt.hist(candidate_loss_all, bins=20)
                plt.xlabel("Sampled system loss")
                plt.ylabel("Frequency")
                plt.show()

            


            timeC = time.time()
            print("number of system calls: ", final_sample)
            print("time taken for system evaluation: ", timeC-timeB)
            print("best loss: ", best_system_loss)
            return best_system_loss

        # do gradient ascent/descent until loss is reached from current parameter configuration
        if method == "naive_climb":
            for loss_comp in zip(components, losses):
                component = loss_comp[0]["component"]
                loss_target = loss_comp[1]

                itr = 0
                while itr < 1000 and abs(component.get_local_loss() - loss_target) / loss_target > percentage_threshold_for_terminate_search: # % threshold
                    curr_loss = component.get_local_loss()
                    if curr_loss > loss_target:
                        component.do_one_descent_on_local()
                    else:
                        component.do_one_ascent_on_local() # can replace this simply with another custom loss function
                    next_loss = component.get_local_loss()
                    itr+=1
                    if abs(next_loss - curr_loss) < 1e-02:
                        break
            best_param = self.get_all_params()[1]
            self.assign_params(best_param)
            return self.get_system_loss()

        # initialise multiple parameter initialization and search for the best
        if method == "multi_search":
            #print("loss to look for: ", losses)
            params = []
            num_starting_points = num_starting_points
            sample_size = 3
            final_sample = samples
            timeA = time.time()
            for loss_comp in zip(components, losses):
                param_candidates = []
                done = 0
                for n in range(num_starting_points):
                    component = loss_comp[0]["component"]
                    loss_target = loss_comp[1]
                    component.random_initialize_param()
                    itr = 0
                    l = []
                    while abs(component.get_local_loss() - loss_target) / loss_target > percentage_threshold_for_terminate_search: # % threshold
                        curr_loss = component.get_local_loss()
                        l.append(curr_loss)
                        if curr_loss > loss_target:
                            component.do_one_descent_on_local()
                        else:
                            component.do_one_ascent_on_local() # can replace this simply with another custom loss function
                        itr+=1
                        if itr > 2000:
                            break
                    if not ignore_failure:
                        if abs(component.get_local_loss() - loss_target) / loss_target > 3e-01 and abs(component.get_local_loss() - loss_target) > 0.01:
                            continue

                    if list(component.get_params()) not in param_candidates:
                        param_candidates.append(list(component.get_params()))
                        done += 1
                    if done > sample_size:
                        break
                        
                if len(param_candidates) == 0:
                    print(l)
                    print("loss target: ", loss_target)
                    print("loss reached: ", component.get_local_loss())
                    print("CANNOT FIND PARAMS, THROW")
                    raise Exception("cannot find param from grad descent error")
                while len(param_candidates) < sample_size:
                    param_candidates = param_candidates + param_candidates
                params.append(sample(param_candidates,sample_size)) # sample
            timeB = time.time()
            print("gradient descent performed on number of components (comp * starting pt): ", len(losses) * num_starting_points)
            print("time taken for gradient descent lookup: ", timeB-timeA)
            all_idx_combination = len(params) * [[x for x in range(sample_size)]]
            all_cartesian_idx = list(itertools.product(*all_idx_combination))
            a = len(list(all_cartesian_idx))
            b = final_sample
            min_to_sample = min(a, b)
            all_cartesian_idx = sample(all_cartesian_idx, min_to_sample) # sample again
            
            best_system_loss = 1e50
            best_param = None
            count = 0
            candidate_loss_all = []
            print("checking each combination for best")
            for idx in all_cartesian_idx:
                count+=1
                candidate_param = []
                for i, cartesian_idx in enumerate(list(idx)):
                    candidate_param += params[i][cartesian_idx]
                self.assign_params(candidate_param)
                candidate_system_loss = self.get_system_loss()
                candidate_loss_all.append(candidate_system_loss)
                if candidate_system_loss < best_system_loss:
                    best_system_loss = candidate_system_loss
                    best_param = candidate_param
            timeC = time.time()
            print("number of system calls: ", final_sample)
            print("time taken for system evaluation: ", timeC-timeB)
            print("best loss: ", best_system_loss)
            if to_plot:
                candidate_loss_all = [x for x in candidate_loss_all if x < 1]
                plt.hist(candidate_loss_all, bins=20)
                plt.xlabel("Sampled system loss")
                plt.ylabel("Frequency")
                plt.show()
            self.assign_params(best_param)
            return best_system_loss
        
        if method == "block_minimization":
            params = []
            num_starting_points = 2
            for loss_comp in zip(components, losses):
                param_candidates = []
                for n in range(num_starting_points):
                    component = loss_comp[0]["component"]
                    loss_target = loss_comp[1]
                    component.random_initialize_param()
                    itr = 0
                    while itr < 500 and abs(component.get_local_loss() - loss_target) / loss_target > 1e-02: # % threshold
                        curr_loss = component.get_local_loss()
                        if curr_loss > loss_target:
                            component.do_one_descent_on_local()
                        else:
                            component.do_one_ascent_on_local() # can replace this simply with another custom loss function
                        next_loss = component.get_local_loss()
                        itr+=1
                        if abs(next_loss - curr_loss) < 1e-05:
                            break
                    if list(component.get_params()) not in param_candidates:
                        param_candidates.append(list(component.get_params()))
                if len(param_candidates) == 0:
                    print("CANNOT FIND PARAMS, THROW")
                    assert False
                params.append(param_candidates)

            # might need to assign initial params
            
            # perform block minimization
            node_names = [x for x in self.nodes if not ("Blackbox" in str(x) or "Dummy" in str(x))]
            for i in range(20): # iterations
                for node_name, param_candidates in zip(node_names, params):
                    # param_candidate is a list of params, iterate and assign best
                    best_system_loss = 1000
                    for param_candidate in param_candidates:
                        self.assign_param_to_node(node_name, param_candidate)
                        curr_loss = self.get_system_loss()
                        if curr_loss < best_system_loss:
                            best_system_loss = curr_loss
            best_param = self.get_all_params()[1]
            print(best_param)
            self.assign_params(best_param)  
            return best_system_loss             
        

    def fit_locally_partial(self, itr=50):
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n):
                continue
            for i in range(itr):
                self.nodes[n]["component"].do_one_descent_on_local()
    
    def random_initialize_param(self,seed=None):
        for n in self.nodes:
            if "Blackbox" in str(n) or "Dummy" in str(n):
                continue
            if seed is not None:
                self.nodes[n]["component"].random_initialize_param(seed)
            else:
                self.nodes[n]["component"].random_initialize_param()
            
    
    def assign_mutual_information_to_node(self, mi : dict):
        for node in mi:
            self.nodes[node]["mi"] = mi[node]
    
    def debug_loss(self):
        for x in range(10):
            self.fit_locally_partial(1)
            print("local losses: ", self.get_local_losses())
            print("system loss: ", self.get_system_loss())
            print("\n")