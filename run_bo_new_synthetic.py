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
from SearchAlgorithm.skeleton import BO_skeleton, BO_graph, BO_graph_turbo, BO_graph_local_loss

from GraphDecomposition.DirectedFunctionalGraph import DirectedFunctionalGraph
from Components.DifferentiablePolynomial import DifferentiablePolynomial
from Models.ModelSinCos import ModelSinCos
from Models.ModelConstant import ModelConstant
from Models.ModelWeightedSum import ModelWeightedSum
from Models.ModelExponential import ModelExponential

from GraphDecomposition.Heuristic import *
from helper import *
from Plotting.HeatMapLossFunction import *
import botorch
from numpy import genfromtxt

botorch.settings.debug = False

import random 
import json

ground_truth_param = {1 : np.array([0.7, 0.9, -0.5]), 2: np.array([0.3, 0.7]), 8: np.array([-0.2, -0.2]), 9: np.array([-0.5, 0.5]), 10: np.array([-0.2, 0.1]), 11: np.array([-0.3, 0.3, 0.2]),
                      12: np.array([0.1, 0.1]), "Blackbox3":np.array([1.2, 0.8]), 4:np.array([1.1, -0.5]), "Blackbox5":np.array([0.7, -0.5]),
                      "Blackbox6": np.array([0.7, 1.1]), 7: np.array([0.7, -0.5])}

import warnings
warnings.filterwarnings("ignore")

f = open('params/synthetic.json')
param = json.load(f)

data_generation_seed = param["data_generation_seed"]
perturbation_noise = param["perturbation_noise"]
noise_std = param["noise_std"]
dg = generate_dg(ground_truth_param, data_generation_seed)
dg.noise_std = noise_std
dg.to_perturb = True
seed = param["starting_seed"]
seeds = [seed]

# ABOLLO
our_bo_trial = param["abollo_trial"] # trials
ours_bo_bound_size = param["abollo_bound_size"] # how much to multiply lower bound by
our_bo_iterations = param["abollo_bo_iteration"] # bo iteration
our_bo_samples = param["abollo_sample_size"] # k samples
our_bo_output_dir = "result/" + "_synthetic_ABOLLO_data_seed_" + str(data_generation_seed) + "_starting_grad_seed_" + str(seed) + "_noise_pos_" + str(perturbation_noise) + "_std_" + str(noise_std)
our_bo_search_method = param["abollo_search_method"]

# vanilla
vanilla_bo_trials = param["vanilla_bo_trial"]
vanilla_bo_iterations = param["vanilla_bo_iteration"]

# gradient descent of local
gradient_system_losses_trials = []
lower_bound_local_loss = None
for s in seeds:
    now = time.time()
    print("doing grad descent")
    dg.random_initialize_param(s)
    lower_bound_local_loss, all_loss = show_system_loss_from_grad_descent(dg, itr=500, plot=False)
    lower_bound_local_loss = [x.detach().cpu().numpy() for x in lower_bound_local_loss]
    gradient_system_losses_trials.append(min(all_loss["system"]))
    later = time.time()
    print("time taken: ", later - now)

np.savetxt("result/gradient.csv", gradient_system_losses_trials)

# vanilla
vanilla_bo_trials = param["vanilla_bo_trial"]
vanilla_bo_iterations = param["vanilla_bo_iteration"]

print(param["vanilla_bo_bounds"])
for b in param["vanilla_bo_bounds"]:
    vanilla_all_trials = []

    trials = 100  
    for x in range(trials):
        print("trial of vanilla BO: ", x)
        r1 = seed
        if len(vanilla_all_trials) >= vanilla_bo_trials:
            break
        try:
            dg.random_initialize_param(r1)
            all_best_losses, _, _ = BO_graph(dg,printout=True, lower_bound = -b, upper_bound=b, iteration=vanilla_bo_iterations)
            vanilla_all_trials.append(all_best_losses)
            seeds.append(r1)
        except:
            print("exception in vanilla BO")
            continue
    vanilla_bo_output_dir = "result/" + "_synthetic_VANILLA_BO_data_seed_" + str(data_generation_seed) + "_starting_grad_seed_" + str(seed) + "_noise_pos_" + str(perturbation_noise) + "_std_" + str(noise_std) + "_bounds_" + str(b) + ".csv"
    np.savetxt(vanilla_bo_output_dir, vanilla_all_trials)

# TURBO
for b in param["turbo_bo_bounds"]:   
    vanilla_all_trials = [] 
    trials = 100
    for x in range(trials):
        if len(vanilla_all_trials) >= vanilla_bo_trials:
            break

        now = time.time()
        dg.random_initialize_param(800)
        batch_size = param["turbo_batch_size"]
        all_best_losses, _, _ = BO_graph_turbo(dg,printout=True,iteration=100,to_normalize_y=True,lower_bound=-b, upper_bound=b, batch_size=batch_size)
        
        vanilla_all_trials.append(all_best_losses)
        later = time.time()
        difference = (later - now)
        print("total time taken: ", difference)

    max_len = max([len(x) for x in vanilla_all_trials])
    for x in range(0, len(vanilla_all_trials)):
        while len(vanilla_all_trials[x]) < max_len:
            vanilla_all_trials[x].append(vanilla_all_trials[x][-1])

    vanilla_bo_output_dir = "result/" + "_synthetic_TURBO_BO_data_seed_" + str(data_generation_seed) + "_starting_grad_seed_" + str(seed) + "_noise_pos_" + str(perturbation_noise) + "_std_" + str(noise_std) + "_bounds_" + str(b) + ".csv"
    vanilla_all_trials = np.array(vanilla_all_trials)
    np.savetxt(vanilla_bo_output_dir, vanilla_all_trials)

run_our_bo(dg, lower_bound_local_loss, seeds,  our_bo_trial, ours_bo_bound_size, our_bo_iterations, our_bo_samples, our_bo_output_dir, our_bo_search_method)

