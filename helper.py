from GraphDecomposition.Heuristic import *
from GraphDecomposition.DirectedFunctionalGraph import *
from Components.DifferentiablePolynomial import *
from Models.ModelSinCos import *
from Models.ModelWeightedSum import *
from Models.ModelExponential import *
from Models.ModelConstant import *
from Models.ModelWeightedSumLogit import ModelWeightedSumLogit
from Models.ModelLogistic import ModelLogistic
from Models.ModelLinearRegression import ModelLinearRegression
from Models.ModelWeightedSumThree import ModelWeightedSumThree
from GraphDecomposition.MutualInformation import mutual_information_nodes_samples
from mnist.MNISTLoader import *
from Models.ModelMNIST import *

from scipy.stats import beta, uniform
from scipy.stats import norm
from scipy.stats import truncnorm
from scipy.stats import truncexpon, expon

import numpy as np
import pandas as pd
import copy
from collections import defaultdict
from sklearn.feature_selection import mutual_info_regression
import time

# single input, single output
def get_data(component : Model, input_range_lower, input_range_upper, ground_truth_param, noisy=True, seed = 10):
    np.random.seed(seed)
    # ground truth for training
    component = copy.deepcopy(component)
    component.set_params(ground_truth_param)

    X_local = torch.tensor(np.random.uniform(input_range_lower, input_range_upper, size=100))
    y_local = component.forward(X_local, noisy=noisy)  # labeling effort of A
    return X_local, y_local

# multi input, single output (default is input size 2)
def get_data_tree(component : Model, input_range_lower, input_range_upper, ground_truth_param, inputs=2, seed = 10):
    np.random.seed(seed)
    # ground truth for training
    component = copy.deepcopy(component)
    component.set_params(ground_truth_param)

    X_local = torch.tensor(np.random.uniform(input_range_lower, input_range_upper, size=(inputs,100)))
    y_local = component.forward(X_local, noisy=True, noise_std=0.2)  # labeling effort of A

    return X_local, y_local

def get_end_to_end_data(dg, gt_param, seed=10):
    np.random.seed(seed)
    #ground truth end to end data from a graph
    for node_idx in dg.nodes:
        if node_idx in gt_param:
            dg.nodes[node_idx]["component"].set_params(gt_param[node_idx])
    
    y = []
    exit = dg.get_exit()
    entry = dg.get_entry()
    X_local = np.random.uniform(0, 5, size=(100,len(entry)))

    for x in X_local:
        assert len(x) == len(entry)
        input_dict = {}
        for node_idx, input in zip(entry, x):
            input_dict[node_idx] = input
        y.append(dg.forward(input_dict, exit, perturbed_black_box=False)) # when generating data, no perturbation
    return torch.tensor(X_local), torch.tensor(y)

def create_mnist_system(ground_truth_param_mnist, noise=0.3, seed=10):  
    np.random.seed(seed)
    dg_nn = DirectedFunctionalGraph(noise)
    # white box components
    local_mnist,system_mnist = generate_data_loader(num_datapts=100,batch_size=100, start_idx=500, label_int=5)
    dg_nn.add_node("nn_1", component=ModelMNIST(local_train_loader=local_mnist, system_train_loader=system_mnist, output_size=2))

    local_mnist_nine,system_mnist_nine = generate_data_loader(num_datapts=100,batch_size=100, start_idx=1000, label_int=7)
    dg_nn.add_node("nn_5", component=ModelMNIST(local_train_loader=local_mnist_nine, system_train_loader=system_mnist_nine, output_size=2))

    dg_nn.add_node(6, component=ModelExponential())
    x,y = get_data(dg_nn.nodes[6]["component"], 0, 5, ground_truth_param_mnist[6], seed=seed)
    dg_nn.nodes[6]["component"].attach_local_data(x,y)
 
    dg_nn.add_node(2, component=ModelWeightedSum())
    x,y = get_data_tree(dg_nn.nodes[2]["component"], 0, 5, ground_truth_param_mnist[2], seed=seed)
    dg_nn.nodes[2]["component"].attach_local_data(x,y)
    
    dg_nn.add_node(4, component=ModelExponential())
    x,y = get_data(dg_nn.nodes[4]["component"], 0, 5, ground_truth_param_mnist[4], seed=seed)
    dg_nn.nodes[4]["component"].attach_local_data(x,y)

    dg_nn.add_node("Blackbox3", component=ModelWeightedSum())
    dg_nn.add_node("Blackbox7", component=ModelWeightedSum())
    dg_nn.nodes["Blackbox7"]["component"].set_params(ground_truth_param_mnist["Blackbox7"])
    
    dg_nn.add_node(8, component=ModelExponential())
    x,y = get_data(dg_nn.nodes[8]["component"], 0, 5, ground_truth_param_mnist[8], seed=seed)
    dg_nn.nodes[8]["component"].attach_local_data(x,y)
    
    dg_nn.add_node(9, component=ModelWeightedSum())
    x,y = get_data_tree(dg_nn.nodes[9]["component"], 0, 5, ground_truth_param_mnist[9], seed=seed)
    dg_nn.nodes[9]["component"].attach_local_data(x,y)
    
    dg_nn.add_node(10, component=ModelSinCos())
    x,y = get_data(dg_nn.nodes[10]["component"], 0, 1, ground_truth_param_mnist[10], seed=seed)
    dg_nn.nodes[10]["component"].attach_local_data(x,y)
    
    dg_nn.add_node(11, component=ModelSinCos())
    x,y = get_data(dg_nn.nodes[11]["component"], 0, 1, ground_truth_param_mnist[11], seed=seed)
    dg_nn.nodes[11]["component"].attach_local_data(x,y)
    
    dg_nn.add_node(12, component=DifferentiablePolynomial(lr=0.01))
    x,y = get_data(dg_nn.nodes[12]["component"], 0, 5, ground_truth_param_mnist[12], seed=seed)
    dg_nn.nodes[12]["component"].attach_local_data(x,y)
    
    dg_nn.add_node(13, component=DifferentiablePolynomial(lr=0.01))
    x,y = get_data(dg_nn.nodes[13]["component"], 0, 5, ground_truth_param_mnist[13], seed=seed)
    dg_nn.nodes[13]["component"].attach_local_data(x,y)
    
    dg_nn.add_node(14, component=ModelWeightedSum())
    x,y = get_data_tree(dg_nn.nodes[14]["component"], -1, 5, ground_truth_param_mnist[14], seed=seed)
    dg_nn.nodes[14]["component"].attach_local_data(x,y)
    
    dg_nn.add_edge(("nn_1", 11), 9)
    dg_nn.add_edge(9, 12)
    dg_nn.add_edge(12, 10)
    dg_nn.add_edge(10, 8)
    dg_nn.add_edge(8, 13)
    dg_nn.add_edge(("Blackbox7", 13), 14)
    dg_nn.add_edge("Blackbox3", 4)
    dg_nn.add_edge("nn_5", 6)
    dg_nn.add_edge((4, 6), "Blackbox7")
    dg_nn.add_edge(("nn_1", 2), "Blackbox3")

    x,y = get_end_to_end_nn_data(dg_nn, ground_truth_param_mnist, seed=seed)

    dg_nn.system_x = x
    dg_nn.system_y = y
    return dg_nn

def create_mnist_system_original(ground_truth_param_mnist, noise=0.3, seed=10):  
    np.random.seed(seed)
    dg_nn = DirectedFunctionalGraph(noise)
    # white box components
    local_mnist,system_mnist = generate_data_loader(num_datapts=100,batch_size=100, start_idx=500, label_int=5)
    dg_nn.add_node("nn_1", component=ModelMNIST(local_train_loader=local_mnist, system_train_loader=system_mnist, output_size=2))

    local_mnist_nine,system_mnist_nine = generate_data_loader(num_datapts=100,batch_size=100, start_idx=1000, label_int=7)
    dg_nn.add_node("nn_5", component=ModelMNIST(local_train_loader=local_mnist_nine, system_train_loader=system_mnist_nine, output_size=2))

    dg_nn.add_node(6, component=DifferentiablePolynomial(lr=0.01))
    x,y = get_data(dg_nn.nodes[6]["component"], 0, 5, ground_truth_param_mnist[6], seed=seed)
    dg_nn.nodes[6]["component"].attach_local_data(x,y)
 
    dg_nn.add_node(2, component=ModelWeightedSum())
    x,y = get_data_tree(dg_nn.nodes[2]["component"], 0, 5, ground_truth_param_mnist[2], seed=seed)
    dg_nn.nodes[2]["component"].attach_local_data(x,y)
    
    dg_nn.add_node(4, component=ModelExponential())
    x,y = get_data(dg_nn.nodes[4]["component"], 0, 5, ground_truth_param_mnist[4], seed=seed)
    dg_nn.nodes[4]["component"].attach_local_data(x,y)

    dg_nn.add_node("Blackbox3", component=ModelWeightedSum())
    dg_nn.add_node("Blackbox7", component=ModelWeightedSum())
    dg_nn.nodes["Blackbox7"]["component"].set_params(ground_truth_param_mnist["Blackbox7"])
    dg_nn.add_edge(("nn_1", 2), "Blackbox3")
    dg_nn.add_edge("Blackbox3", 4)
    dg_nn.add_edge("nn_5", 6)
    dg_nn.add_edge((4, 6), "Blackbox7")

    x,y = get_end_to_end_nn_data(dg_nn, ground_truth_param_mnist, seed=seed)

    dg_nn.system_x = x
    dg_nn.system_y = y
    return dg_nn

def get_end_to_end_nn_data(dg, gt_param, seed=10):
    np.random.seed(seed)
    #ground truth end to end data from a graph
    for node_idx in dg.nodes:
        if node_idx in gt_param:
            print("setting: ", gt_param[node_idx])
            dg.nodes[node_idx]["component"].set_params(gt_param[node_idx])
    
    y_system = []
    exit = dg.get_exit()
    entry = dg.get_entry()
    X_system = []

    for idx in range(100): # 100 datapoints, can adjust
        input_data = []
        input_dict = {}
        for node_idx in entry:
            if "nn" in str(node_idx): # if neural network; assume batch size is also 100
                dg.nodes[node_idx]["component"].set_oracle_mode(True)
                input = dg.nodes[node_idx]["component"].X[idx]
                input = input.unsqueeze(0)
                input = input.unsqueeze(1)
                input_data.append(input)
                input = dg.nodes[node_idx]["component"].y[idx] # hack, our model is in oracle mode
            elif "model" in str(node_idx):
                dg.nodes[node_idx]["component"].set_oracle_mode(True)
                input = dg.nodes[node_idx]["component"].X[idx]
                input_data.append(input)
                input = dg.nodes[node_idx]["component"].y[idx]
            else: # if model, or black box
                input = np.random.uniform(0, 5, dg.nodes[node_idx]["component"].inputs)
                input_data.append(input)
            input_dict[node_idx] = input
        X_system.append(input_data)
        output = dg.forward(input_dict, exit, perturbed_black_box=False)
        y_system.append(output) # when generating data, no perturbation
        for node_idx in entry:
            if dg.nodes[node_idx]["component"].oracle_mode == True:
                dg.nodes[node_idx]["component"].set_oracle_mode(False)
    return X_system, torch.tensor(y_system)

def get_body_fat_data():
    path = "health_care/body_fat/body_fat_cleaned.csv"
    data = pd.read_csv(path)
    data = data.sample(frac=1)
    y = torch.Tensor(np.array(list(data["BodyFat"])).astype('float32'))
    X = torch.nn.functional.normalize(torch.Tensor(data.drop(["BodyFat"],axis=1).to_numpy().astype('float32')))
    return X,y

def get_heart_disease_data():
    path = "health_care/heart_disease_1/heart_cleaned.csv"
    data = pd.read_csv(path)
    data = data.sample(frac=1)
    y = torch.Tensor(np.array(list(data["HeartDisease"])).astype('float32'))
    X = torch.nn.functional.normalize(torch.Tensor(data.drop(["HeartDisease"],axis=1).to_numpy().astype('float32')))
    return X,y

def get_hepatitis_data():
    path = "health_care/hepatitis/hepatitis_cleaned.csv"
    data = pd.read_csv(path)
    data = data.sample(frac=1)
    y = torch.Tensor(np.array(list(data["Category"])).astype('float32'))
    X = torch.nn.functional.normalize(torch.Tensor(data.drop(["Category"],axis=1).to_numpy().astype('float32')))
    return X,y

def get_kidney_data():
    path = "health_care/kidney/kidney_disease_risk_cleaned.csv"
    data = pd.read_csv(path)
    data = data.sample(frac=1)
    y = torch.Tensor(np.array(list(data["class_ckd"])).astype('float32'))
    X = torch.nn.functional.normalize(torch.Tensor(data.drop(["class_ckd"],axis=1).to_numpy().astype('float32')))
    return X,y

def generate_dg(ground_truth_param, noise=0.2, seed=10):
    np.random.seed(seed)
    DG = DirectedFunctionalGraph(noise=0.2)

    # white box components
    DG.add_node(1, component=DifferentiablePolynomial(lr=0.001))
    x,y = get_data(DG.nodes[1]["component"], 0, 2, ground_truth_param[1])
    DG.nodes[1]["component"].attach_local_data(x,y)

    DG.add_node(2, component=ModelSinCos())
    x,y = get_data(DG.nodes[2]["component"], -3, 6, ground_truth_param[2])
    DG.nodes[2]["component"].attach_local_data(x,y)

    DG.add_node(4, component=ModelExponential())
    x,y = get_data(DG.nodes[4]["component"], 0, 2, ground_truth_param[4])
    DG.nodes[4]["component"].attach_local_data(x,y)

    DG.add_node(7, component=ModelSinCos())
    x,y = get_data(DG.nodes[7]["component"], 0, 5, ground_truth_param[7])
    DG.nodes[7]["component"].attach_local_data(x,y)
    
    DG.add_node(8, component=ModelExponential())
    x,y = get_data(DG.nodes[8]["component"], -1, 2, ground_truth_param[8])
    DG.nodes[8]["component"].attach_local_data(x,y)
    
    DG.add_node(9, component=ModelExponential())
    x,y = get_data(DG.nodes[9]["component"], -2, 2, ground_truth_param[9])
    DG.nodes[9]["component"].attach_local_data(x,y)
    
    DG.add_node(10, component=ModelSinCos())
    x,y = get_data(DG.nodes[10]["component"], -2, 2, ground_truth_param[10])
    DG.nodes[10]["component"].attach_local_data(x,y)
    
    DG.add_node(11, component=DifferentiablePolynomial(lr=0.001))
    x,y = get_data(DG.nodes[11]["component"], 0, 5, ground_truth_param[11])
    DG.nodes[11]["component"].attach_local_data(x,y)
    
    DG.add_node(12, component=ModelSinCos())
    x,y = get_data(DG.nodes[12]["component"], 0, 5, ground_truth_param[12])
    DG.nodes[12]["component"].attach_local_data(x,y)

    # black box components
    DG.add_node("Blackbox3", component=ModelWeightedSum())
    DG.nodes["Blackbox3"]["component"].set_params(ground_truth_param["Blackbox3"])

    DG.add_node("Blackbox5", component=ModelWeightedSum())
    DG.nodes["Blackbox5"]["component"].set_params(ground_truth_param["Blackbox5"])

    DG.add_node("Blackbox6", component=ModelWeightedSum())
    DG.nodes["Blackbox6"]["component"].set_params(ground_truth_param["Blackbox6"])

    # Test warning for multiple parents
    DG.add_edge(["Blackbox6",7],"Blackbox3")
    DG.add_edge([1,2],"Blackbox3")

    DG.add_edge([4,2],"Blackbox5")

    # Test warning for singular parents
    DG.add_edge(2,4)
    DG.add_edge("Blackbox3",4)
    DG.add_edge([7,"Blackbox5"],"Blackbox6")
    
    DG.add_edge(8, 7)
    DG.add_edge(9, 8)
    DG.add_edge(10, 9)
    DG.add_edge(11, 10)
    DG.add_edge(12, 11)

    X_end, y_end = get_end_to_end_nn_data(DG, ground_truth_param)
    DG.system_x = X_end
    DG.system_y = y_end
    
    DG.random_initialize_param()
    return DG

def generate_dg_logit_output(ground_truth_param, noise=0.2, seed=10):
    np.random.seed(seed)
    DG = DirectedFunctionalGraph(noise=0.2)

    # white box components
    DG.add_node(1, component=DifferentiablePolynomial(lr=0.001))
    x,y = get_data(DG.nodes[1]["component"], 0, 2, ground_truth_param[1])
    DG.nodes[1]["component"].attach_local_data(x,y)

    DG.add_node(2, component=ModelSinCos())
    x,y = get_data(DG.nodes[2]["component"], -3, 6, ground_truth_param[2])
    DG.nodes[2]["component"].attach_local_data(x,y)

    DG.add_node(4, component=ModelExponential())
    x,y = get_data(DG.nodes[4]["component"], 0, 2, ground_truth_param[4])
    DG.nodes[4]["component"].attach_local_data(x,y)

    DG.add_node(7, component=ModelSinCos())
    x,y = get_data(DG.nodes[7]["component"], 0, 5, ground_truth_param[7])
    DG.nodes[7]["component"].attach_local_data(x,y)
    
    DG.add_node(8, component=ModelExponential())
    x,y = get_data(DG.nodes[8]["component"], -1, 2, ground_truth_param[8])
    DG.nodes[8]["component"].attach_local_data(x,y)
    
    DG.add_node(9, component=ModelExponential())
    x,y = get_data(DG.nodes[9]["component"], -2, 2, ground_truth_param[9])
    DG.nodes[9]["component"].attach_local_data(x,y)
    
    DG.add_node(10, component=ModelSinCos())
    x,y = get_data(DG.nodes[10]["component"], -2, 2, ground_truth_param[10])
    DG.nodes[10]["component"].attach_local_data(x,y)
    
    DG.add_node(11, component=DifferentiablePolynomial(lr=0.001))
    x,y = get_data(DG.nodes[11]["component"], 0, 5, ground_truth_param[11])
    DG.nodes[11]["component"].attach_local_data(x,y)
    
    DG.add_node(12, component=ModelSinCos())
    x,y = get_data(DG.nodes[12]["component"], 0, 5, ground_truth_param[12])
    DG.nodes[12]["component"].attach_local_data(x,y)

    # black box components
    DG.add_node("Blackbox3", component=ModelWeightedSum())
    DG.nodes["Blackbox3"]["component"].set_params(ground_truth_param["Blackbox3"])

    DG.add_node("Blackbox5", component=ModelWeightedSum())
    DG.nodes["Blackbox5"]["component"].set_params(ground_truth_param["Blackbox5"])

    DG.add_node("Blackbox6", component=ModelWeightedSumLogit())
    DG.nodes["Blackbox6"]["component"].set_params(ground_truth_param["Blackbox6"])

    # Test warning for multiple parents
    DG.add_edge(["Blackbox6",7],"Blackbox3")
    DG.add_edge([1,2],"Blackbox3")

    DG.add_edge([4,2],"Blackbox5")

    # Test warning for singular parents
    DG.add_edge(2,4)
    DG.add_edge("Blackbox3",4)
    DG.add_edge([7,"Blackbox5"],"Blackbox6")
    
    DG.add_edge(8, 7)
    DG.add_edge(9, 8)
    DG.add_edge(10, 9)
    DG.add_edge(11, 10)
    DG.add_edge(12, 11)

    X_end, y_end = get_end_to_end_nn_data(DG, ground_truth_param)
    DG.system_x = X_end
    DG.system_y = y_end
    
    DG.random_initialize_param()
    return DG

'''
Given a 1) DG, 2) a set of sub-system 3)ground truth param,
Generate a new DG with end to end data inside
'''
def generate_sub_system(sub_system : set, DG : DirectedFunctionalGraph, ground_truth_param : dict) -> DirectedFunctionalGraph:
    sub_system_extended = DG.generate_sub_system(sub_system)
    graph_temp = copy.deepcopy(DG)
    graph_temp.retain_nodes(sub_system_extended)
    relabel_nodes = {}
    for n in graph_temp.nodes:
        if n not in sub_system:
            graph_temp.nodes[n]["component"] = ModelConstant()
            relabel_nodes[n] = str(n) + "Dummy"
            graph_temp = nx.relabel_nodes(graph_temp, relabel_nodes)
    
    # reassign parents to proper name
    for n in graph_temp.nodes:
        if "parents" in graph_temp.nodes[n]:
            new_parents = []
            for parent in graph_temp.nodes[n]["parents"]:
                if parent in relabel_nodes:
                    new_parents.append(relabel_nodes[parent])
                else:
                    new_parents.append(parent)
            graph_temp.nodes[n]["parents"] = new_parents
            

    X,y = get_end_to_end_data(graph_temp,ground_truth_param)
    graph_temp.system_x = X
    graph_temp.system_y = y
    return graph_temp

'''
Do gradient descent on every non-black box component and plot the losses, along with system loss
'''
def show_system_loss_from_grad_descent(DG, itr=500, y_min=0.0, y_max=1.0, plot=False):
    losses = defaultdict(list)
    num_itr = itr
    # gradient descent of individual
    for i in range(num_itr):
        last_loss = []
        for node_idx in DG.nodes:

            if ("Dummy" in str(node_idx)) or ("Blackbox" in str(node_idx)):
                continue
            comp = DG.nodes[node_idx]["component"]
            comp.do_one_descent_on_local()
            local_loss = comp.get_local_loss().detach()
            losses[node_idx].append(local_loss)
            last_loss.append(local_loss)
        losses["system"].append(DG.get_system_loss())
    print("final system loss: ", DG.get_system_loss())
    if plot:
        for l in losses:
            loss = (np.array([x.cpu() for x in losses[l]]))
            if l == "system":
                plt.plot(range(num_itr), loss, label= str(l)+" loss", c="black")
            else:
                plt.plot(range(num_itr), loss, label="component " + str(l)+" loss")
        plt.legend(loc="upper right")
        plt.ylim(y_min,y_max)
        plt.ylabel("loss")
        plt.xlabel("GD iterations")
        #plt.title("component and system loss with local gradient descent")
        plt.show()
    return last_loss, losses

def find_statistics_from_samples(samples, method="scipy"):

    # use scipy to estimate exponential dist
    res = truncexpon.fit(samples, fb=max(samples), fscale=1, method="mm")
    print(res)
    b, loc, scale = res[0], res[1], res[2]
    lln_exp = np.sum(np.log(0.001 + truncexpon(b,loc,scale).pdf(samples)))

    # use our method to estimate uniform
    k = len(samples)
    width_estimate =  (max(samples) - min(samples)) * (k+1)/(k-1)
    lower_bound_estimate = min(samples) - (width_estimate / (k+1))
    lln_uni = np.sum(np.log(0.001 + uniform(lower_bound_estimate,width_estimate).pdf(samples)))

    if lln_exp > lln_uni:
        return loc, truncexpon(b,loc,scale)

    # if distribution == "uniform":
    #     minimum = min(samples)
    #     maximum = max(samples)
    #     width_estimate = (maximum- minimum) * (k + 1)/(k - 1)
    #     minimum_bound_estimate = minimum - (width_estimate/(k+1))
    #     variance_of_minimum = k/((k+2)*(k+1)**2 )
    return minimum, maximum, minimum_bound_estimate, width_estimate, variance_of_minimum
'''
Find M.I of each component loss w.r.t system loss
via random sampling (uniform w.r.t gradient descent loss)
'''
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
            sampled_local_loss = sample(list(local_node_loss), 1)[0]
            sampled_param = samples[node_name][sampled_local_loss]
            dg.assign_param_to_node(node_name, sampled_param)
        

    all_local_losses = np.stack(all_local_losses)
    mi = mutual_info_regression(all_local_losses, all_system_losses, n_neighbors=10)
    return mi

'''
Compute M.I of each component and assign to each node of DG
'''
def assign_mi(DG : DirectedFunctionalGraph):
    DG.random_initialize_param()
    mi = get_mi(DG)
    mi_dict = {k:v for k,v in zip(list(DG.get_components().keys()), mi)}
    DG.assign_mutual_information_to_node(mi_dict)
    print("MI: ", mi)
    return DG

import random
from numpy import genfromtxt
from SearchAlgorithm.skeleton import BO_skeleton, BO_graph, BO_graph_local_loss

def run_experiments_all(system_param : dict, data_generation_seed : int, vanilla_bo_trials : int, vanilla_bo_iterations: int, vanilla_bo_output_dir, vanilla_bo_all_seeds_dir,
                        our_bo_trial : int, ours_bo_bound_size: int, our_bo_iterations : int, our_bo_samples: list, our_bo_output_dir, our_bo_search_method):
    DG = generate_dg(system_param, data_generation_seed)
    nx.draw_networkx(DG)

    #grad descent
    DG.random_initialize_param(data_generation_seed)
    lower_bound_local_loss, all_loss = show_system_loss_from_grad_descent(DG, itr=500, plot=True)
    lower_bound_local_loss = [x.detach().numpy().tolist() for x in lower_bound_local_loss]

    # vanilla BO

    vanilla_all_trials = []

    trials = 100
    seeds = []    
    for x in range(trials):
        print("trial of vanilla BO: ", x)
        r1 = 5555
        if len(vanilla_all_trials) >= vanilla_bo_trials:
            break
        try:
            DG.random_initialize_param(r1)
            if (DG.get_system_loss() > 100):
                continue
            all_best_losses, _, _ = BO_graph(DG,printout=True,iteration=vanilla_bo_iterations)
            vanilla_all_trials.append(all_best_losses)
            seeds.append(r1)
        except:
            print("exception in vanilla BO")
            continue

    vanilla_all_trials = np.array(vanilla_all_trials)
    np.savetxt(vanilla_bo_output_dir, vanilla_all_trials)
    print("seeds: ")
    print(seeds)
    np.savetxt(vanilla_bo_all_seeds_dir, seeds)
    run_our_bo(DG, lower_bound_local_loss, seeds,  our_bo_trial, ours_bo_bound_size, our_bo_iterations, our_bo_samples, our_bo_output_dir, our_bo_search_method)

def create_healthcare_system(param, noise=0.5, seed=11):  
    np.random.seed(seed)
    dg_nn = DirectedFunctionalGraph(noise)
    
    # white box components
    dg_nn.add_node("model_heart_disease", component=ModelLogistic(21))
    x,y = get_heart_disease_data()
    dg_nn.nodes["model_heart_disease"]["component"].attach_local_data(x,y)
 
    dg_nn.add_node("model_liver_hep", component=ModelLogistic(14))
    x,y = get_hepatitis_data()
    
    dg_nn.nodes["model_liver_hep"]["component"].attach_local_data(x,y)
    
    dg_nn.add_node("model_kidney", component=ModelLogistic(146))
    x,y = get_kidney_data()
    dg_nn.nodes["model_kidney"]["component"].attach_local_data(x,y)
    
    dg_nn.add_node("model_body_fat", component=ModelLinearRegression(15))
    x,y = get_body_fat_data()
    dg_nn.nodes["model_body_fat"]["component"].attach_local_data(x,y)

    dg_nn.add_node("Blackbox_DoctorA", component=ModelWeightedSum())
    dg_nn.add_node("Blackbox_DoctorB", component=ModelWeightedSumThree())
    
    dg_nn.add_node("model_aggregate", component=ModelWeightedSum())
    x,y = get_data_tree(dg_nn.nodes["model_aggregate"]["component"], 0, 2, param["model_aggregate"])
    dg_nn.nodes["model_aggregate"]["component"].attach_local_data(x,y)
    
    dg_nn.add_edge(("model_liver_hep", "model_kidney", "model_body_fat"), "Blackbox_DoctorB")
    dg_nn.add_edge(("model_heart_disease", "model_body_fat"), "Blackbox_DoctorA")
    dg_nn.add_edge(("Blackbox_DoctorA", "Blackbox_DoctorB"), "model_aggregate")

    x,y = get_end_to_end_nn_data(dg_nn, param, seed=seed)

    dg_nn.system_x = x
    dg_nn.system_y = y
    return dg_nn

def run_our_bo(DG : DirectedFunctionalGraph, lower_bound_local_loss, seeds,  our_bo_trial : int, ours_bo_bound_size: int, our_bo_iterations, our_bo_samples: list, our_bo_output_dir, our_bo_search_method):
    samples = our_bo_samples
    seeds = copy.deepcopy(list(seeds))
    for s in samples:
        for itr in our_bo_iterations:
            seed_for_trial = copy.deepcopy(list(seeds))
            print("samples: ", s)
            loss_space_bo_all_trials = []
            for x in range(200):
                print("number of attempts: ", x)
                print("trial of our BO (successful): ", len(loss_space_bo_all_trials))
                if len(loss_space_bo_all_trials) >= our_bo_trial:
                    break
                
                try:
                    torch.manual_seed(int(seed_for_trial[0]))
                    DG.random_initialize_param(int(seed_for_trial[0]))
                    print("system loss: ", DG.get_system_loss())
                    # BO with local loss -> system loss
                    print("bounds: ", np.array(lower_bound_local_loss))
                    bounds = torch.tensor([np.array(lower_bound_local_loss) * 1.1, np.array(lower_bound_local_loss) * ours_bo_bound_size])
                    all_best_losses_ours, best_param = BO_graph_local_loss(DG, bounds, our_bo_search_method, s, printout=True, iteration=itr, ignore_error=True)
                    loss_space_bo_all_trials.append(all_best_losses_ours)

                except Exception as e:
                    print(e)
                    print("IN TROUBLE NOW. TERMINATE AND FIND OUT WHY???")
                    continue



            loss_space_bo_all_trials = np.array(loss_space_bo_all_trials)
            file_name = our_bo_output_dir + "_" + str(s) + "_" + str(itr) +".csv"
            np.savetxt(file_name, loss_space_bo_all_trials)
        

def run_our_bo_same_queries(DG : DirectedFunctionalGraph, lower_bound_local_loss, seeds,  our_bo_trial : int, ours_bo_bound_size: int, our_bo_iterations : list, our_bo_samples: list, our_bo_output_dir, our_bo_search_method):
    samples = our_bo_samples
    seeds = copy.deepcopy(list(seeds))
    for s,itr in zip(samples,our_bo_iterations):
        seed_for_trial = copy.deepcopy(list(seeds))
        print("samples: ", s)
        loss_space_bo_all_trials = []
        for x in range(200):
            print("number of attempts: ", x)
            print("trial of our BO (successful): ", len(loss_space_bo_all_trials))
            if len(loss_space_bo_all_trials) >= our_bo_trial:
                break
            
            try:
                torch.manual_seed(int(seed_for_trial[0]))
                DG.random_initialize_param(int(seed_for_trial[0]))
                print("system loss: ", DG.get_system_loss())
                # BO with local loss -> system loss
                print("bounds: ", np.array(lower_bound_local_loss))
                bounds = torch.tensor([np.array(lower_bound_local_loss), np.array(lower_bound_local_loss) * ours_bo_bound_size])
                all_best_losses_ours, best_param = BO_graph_local_loss(DG, bounds, our_bo_search_method, s, printout=True, iteration=itr)
                loss_space_bo_all_trials.append(all_best_losses_ours)
                seed_for_trial.pop(0)
            except:
               continue



        loss_space_bo_all_trials = np.array(loss_space_bo_all_trials)
        file_name = our_bo_output_dir + "_" + str(s) + ".csv"
        np.savetxt(file_name, loss_space_bo_all_trials)
        

def run_experiments_all_same_queries(system_param : dict, data_generation_seed : int, vanilla_bo_trials : int, vanilla_bo_iterations: int, vanilla_bo_output_dir, vanilla_bo_all_seeds_dir,
                        our_bo_trial : int, ours_bo_bound_size: int, our_bo_iterations : list, our_bo_samples: list, our_bo_output_dir, our_bo_search_method):
    DG = generate_dg(system_param, data_generation_seed)
    nx.draw_networkx(DG)

    #grad descent
    DG.random_initialize_param(data_generation_seed)
    lower_bound_local_loss, all_loss = show_system_loss_from_grad_descent(DG, itr=500, plot=True)
    lower_bound_local_loss = [x.detach().numpy().tolist() for x in lower_bound_local_loss]

    # vanilla BO

    vanilla_all_trials = []

    trials = 100
    seeds = []    
    for x in range(trials):
        print("trial of vanilla BO: ", x)
        r1 = random.randint(0, 100000)
        if len(vanilla_all_trials) >= vanilla_bo_trials:
            break
        try:
            DG.random_initialize_param(r1)
            if (DG.get_system_loss() > 100):
                continue
            all_best_losses, _, _ = BO_graph(DG,printout=True,iteration=vanilla_bo_iterations)
            vanilla_all_trials.append(all_best_losses)
            seeds.append(r1)
        except:
            print("exception in vanilla BO")
            continue

    vanilla_all_trials = np.array(vanilla_all_trials)
    np.savetxt(vanilla_bo_output_dir, vanilla_all_trials)
    print("seeds: ")
    print(seeds)
    np.savetxt(vanilla_bo_all_seeds_dir, seeds)
    run_our_bo_same_queries(DG, lower_bound_local_loss, seeds,  our_bo_trial, ours_bo_bound_size, our_bo_iterations, our_bo_samples, our_bo_output_dir, our_bo_search_method)
