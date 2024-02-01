import math
import numpy as np
import matplotlib.pyplot as plt
import torch
import copy
from collections import defaultdict

from Components.ConditionalNormalDistribution import ConditionalNormalDistribution
from Components.DifferentiablePolynomial import DifferentiablePolynomial
from Models.ModelExponential import ModelExponential
from Models.ModelSinCos import ModelSinCos
from Models.ModelLogistic import ModelLogistic
from Models.ModelSigmoid import ModelSigmoid
from Composition.SequentialSystem import SequentialSystem
from SearchAlgorithm.skeleton import BO_skeleton, BO_graph, BO_graph_local_loss, BO_graph_turbo

from GraphDecomposition.DirectedFunctionalGraph import DirectedFunctionalGraph
from Components.DifferentiablePolynomial import DifferentiablePolynomial
from Models.ModelSinCos import ModelSinCos
from Models.ModelConstant import ModelConstant
from Models.ModelWeightedSum import ModelWeightedSum
from Models.ModelExponential import ModelExponential

from GraphDecomposition.Heuristic import *
from helper import *
from Plotting.HeatMapLossFunction import *

from Models.ModelMNIST import ModelMNIST
from mnist.MNISTLoader import *
from helper import *
import time
import json

import warnings
warnings.filterwarnings("ignore")

f = open('params/mnist_large.json')
param = json.load(f)

ground_truth_param_mnist = {"Blackbox7": np.array([0.8,0.8]), 6 : np.array([0.7, 0.9]), 2: np.array([0.5, 0.7]), 12: np.array([0.5, 0.5]), "Blackbox3":np.array([1.2, 0.8]), 4:np.array([-0.3, 0.5]),
                            8 : np.array([0.7, 0.9]), 9 : np.array([-0.7, 0.9]), 10 : np.array([0.7, 0.9]), 11 : np.array([-0.4, 0.8]), 12 : np.array([-0.4, 0.8, 0.1]), 13 : np.array([-0.4, 0.8, 0.1]), 14:np.array([0.5, 0.8])}

ground_truth_param_mnist_original = {"Blackbox7": np.array([0.8,0.8]), 6 : np.array([0.7, 0.9, -0.3]), 2: np.array([0.5, 0.7]), 12: np.array([0.5, 0.5]), "Blackbox3":np.array([1.2, 0.8]), 4:np.array([-0.3, 0.5]),
                            8 : np.array([0.7, 0.9]), 9 : np.array([-0.7, 0.9]), 10 : np.array([0.7, 0.9]), 11 : np.array([-0.4, 0.8]), 12 : np.array([-0.4, 0.8, 0.1]), 13 : np.array([-0.4, 0.8, 0.1]), 14:np.array([0.5, 0.8])}

data_generation_seed = param["data_generation_seed"]
perturbation_noise = param["perturbation_noise"]
noise_std = param["noise_std"]

if param["mnist_size"] == "small":
    dg_nn = create_mnist_system_original(ground_truth_param_mnist_original,  noise=perturbation_noise, seed=data_generation_seed)
else:
    dg_nn = create_mnist_system(ground_truth_param_mnist,  noise=perturbation_noise, seed=data_generation_seed)

dg_nn.noise_std = noise_std
dg_nn.to_perturb = True
seed = param["starting_seed"]
seeds = [seed]

# ABOLLO
our_bo_trial = param["abollo_trial"] # trials
ours_bo_bound_size = param["abollo_bound_size"] # how much to multiply lower bound by
our_bo_iterations = param["abollo_bo_iteration"] # bo iteration
our_bo_samples = param["abollo_sample_size"] # k samples
our_bo_output_dir = "result/" + param["mnist_size"] + "_mnist_ABOLLO_data_seed_" + str(data_generation_seed) + "_starting_grad_seed_" + str(seed) + "_noise_pos_" + str(perturbation_noise) + "_std_" + str(noise_std) + ".csv"
our_bo_search_method = param["abollo_search_method"]

# gradient descent of local
gradient_system_losses_trials = []
lower_bound_local_loss = None
for s in seeds:
    now = time.time()
    print("doing grad descent")
    dg_nn.random_initialize_param(s)
    lower_bound_local_loss, all_loss = show_system_loss_from_grad_descent(dg_nn, itr=500, plot=False)
    lower_bound_local_loss = [x.detach().cpu().numpy() for x in lower_bound_local_loss]
    gradient_system_losses_trials.append(min(all_loss["system"]))
    later = time.time()
    print("time taken: ", later - now)

np.savetxt("result/mnist_gradient.csv", gradient_system_losses_trials)

# vanilla
vanilla_bo_trials = param["vanilla_bo_trial"]
vanilla_bo_iterations = param["vanilla_bo_iteration"]

for b in param["vanilla_bo_bounds"]:
    vanilla_all_trials = []

    trials = 100  
    for x in range(trials):
        print("trial of vanilla BO: ", x)
        r1 = seed
        if len(vanilla_all_trials) >= vanilla_bo_trials:
            break
        try:
            dg_nn.random_initialize_param(r1)
            for x in range(100):
                dg_nn.nodes["nn_1"]["component"].do_one_descent_on_local()
                dg_nn.nodes["nn_5"]["component"].do_one_descent_on_local()
            
            all_best_losses, _, _ = BO_graph(dg_nn,printout=True, lower_bound = -b, upper_bound=b, iteration=vanilla_bo_iterations)
            vanilla_all_trials.append(all_best_losses)
            seeds.append(r1)
        except:
            print("exception in vanilla BO")
            continue
    vanilla_bo_output_dir = "result/" + param["mnist_size"] + "_mnist_VANILLA_BO_data_seed_" + str(data_generation_seed) + "_starting_grad_seed_" + str(seed) + "_noise_pos_" + str(perturbation_noise) + "_std_" + str(noise_std) + "_bounds_" + str(b) + ".csv"
    np.savetxt(vanilla_bo_output_dir, vanilla_all_trials)

# TURBO
for b in param["turbo_bo_bounds"]:   
    vanilla_all_trials = [] 
    trials = 100
    for x in range(trials):
        if len(vanilla_all_trials) >= 5:
            break

        now = time.time()
        dg_nn.random_initialize_param(seed)
        for x in range(100):
            dg_nn.nodes["nn_1"]["component"].do_one_descent_on_local()
            dg_nn.nodes["nn_5"]["component"].do_one_descent_on_local()
        batch_size = param["turbo_batch_size"]
        all_best_losses, _, _ = BO_graph_turbo(dg_nn,printout=True,iteration=100,to_normalize_y=True,lower_bound=-b, upper_bound=b, batch_size=batch_size)
        
        vanilla_all_trials.append(all_best_losses)
        later = time.time()
        difference = (later - now)
        print("total time taken: ", difference)

    max_len = max([len(x) for x in vanilla_all_trials])
    for x in range(0, len(vanilla_all_trials)):
        while len(vanilla_all_trials[x]) < max_len:
            vanilla_all_trials[x].append(vanilla_all_trials[x][-1])

    vanilla_bo_output_dir = "result/" + param["mnist_size"] + "_mnist_TURBO_BO_data_seed_" + str(data_generation_seed) + "_starting_grad_seed_" + str(seed) + "_noise_pos_" + str(perturbation_noise) + "_std_" + str(noise_std) + "_bounds_" + str(b) + ".csv"
    vanilla_all_trials = np.array(vanilla_all_trials)
    np.savetxt(vanilla_bo_output_dir, vanilla_all_trials)

run_our_bo(dg_nn, lower_bound_local_loss, seeds,  our_bo_trial, ours_bo_bound_size, our_bo_iterations, our_bo_samples, our_bo_output_dir, our_bo_search_method)