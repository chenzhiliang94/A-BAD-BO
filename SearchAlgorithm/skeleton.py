import torch
from botorch.models import SingleTaskGP, HeteroskedasticSingleTaskGP, KroneckerMultiTaskGP, MultiTaskGP, FixedNoiseMultiTaskGP, FixedNoiseGP, HeteroskedasticSingleTaskGP
from botorch.fit import fit_gpytorch_mll
from gpytorch.mlls import ExactMarginalLogLikelihood, SumMarginalLogLikelihood, MarginalLogLikelihood
from botorch.optim import optimize_acqf
from Composition.SequentialSystem import SequentialSystem
from botorch.acquisition import UpperConfidenceBound
from SearchAlgorithm.AcquisitionFunction import *
from SearchAlgorithm.turbo import *
import numpy as np
import time

from GraphDecomposition.DirectedFunctionalGraph import *

dtype = torch.float32

def BO_skeleton(system: SequentialSystem, objective="system", model="single_task_gp", printout=False):
    parameter_trials = torch.tensor((np.array(system.get_parameters())).flatten(), dtype=dtype).unsqueeze(0)

    next_param = torch.DoubleTensor([[1.0, 1.0, 1.0, 1.0]])
    target = []
    input_param = next_param
    best_param = None
    best_loss_tuples = None
    best_objective = -10000
    for i in range(30):
        if printout:
            print("BO iteration: ", i)
            print("Current best objective: ", best_objective)
        system.assign_parameters(next_param)
        if objective == "all":
            current_loss = - system.compute_system_loss() - sum(system.compute_local_loss())
        if objective == "all":
            current_loss = - system.compute_system_loss()
        if current_loss > best_objective:
            best_objective = current_loss
            best_param = system.get_parameters()
            best_loss_tuples = system.compute_local_loss(), system.compute_system_loss()

        UCB = None
        gp = None
        if model == "single_task_gp":
            target.append([- system.compute_system_loss() - sum(system.compute_local_loss())])
            Y = torch.DoubleTensor(target)

            # parameterize standardization and model
            # target = standardize(target)
            gp = SingleTaskGP(input_param.double(), Y.double())
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll);
            UCB = UpperConfidenceBound(gp, beta=1)

        if model == "multi_task_gp_bonilla":
            target.append([-system.compute_local_loss()[0], -system.compute_local_loss()[1], -system.compute_system_loss()])
            Y = torch.DoubleTensor(target)

            # parameterize standardization and model
            # target = standardize(target)
            gp = KroneckerMultiTaskGP(input_param, Y)
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll);
            UCB = ScalarizedUpperConfidenceBound(gp, beta=1, weights=torch.tensor([1.0, 1.0, 1.0]).double())

        bounds = torch.stack([torch.ones(parameter_trials.shape[-1])*0, 2*torch.ones(parameter_trials.shape[-1])])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20,
        )
        next_param = candidate
        input_param = torch.cat((input_param, next_param), 0)
    return parameter_trials, best_loss_tuples, best_param

def BO_graph(system : DirectedFunctionalGraph, printout=True, iteration=50, lower_bound = -1, upper_bound=1,to_normalize_y=False):
    for node in system.nodes:
        if "Blackbox" in str(node):
            continue
    parameter_trials = torch.tensor((np.array(system.get_all_params()[1])).flatten(), dtype=dtype).unsqueeze(0)

    next_param = torch.DoubleTensor([system.get_all_params()[1]])
    target = []
    input_param = next_param
    best_param = None
    best_objective = -10000
    all_best_losses = []
    for i in range(iteration):
        now = time.time()
        if printout:
            print("BO iteration: ", i)
            print("Current best objective: ", best_objective)
        system.assign_params(next_param[0])
        current_loss = - system.get_system_loss()
        print("current loss: ", current_loss)
        if current_loss > best_objective:
            best_objective = current_loss
            best_param = system.get_all_params()[1]
        all_best_losses.append(best_objective)
        
        UCB = None
        gp = None

        target.append([current_loss])
        Y = torch.DoubleTensor(target)
        if to_normalize_y:
            Y = torch.nn.functional.normalize(Y)

        # parameterize standardization and model
        # target = standardize(target)
        gp = SingleTaskGP(input_param, Y)
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fitting_error_count = 0
        try:
            fit_gpytorch_mll(mll)
        except:
            fitting_error_count+=1
        UCB = UpperConfidenceBound(gp, beta=0.1)

        bounds = torch.stack([torch.ones(parameter_trials.shape[-1]) * lower_bound, upper_bound * torch.ones(parameter_trials.shape[-1])])
        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20,
        )
        next_param = candidate
        input_param = torch.cat((input_param, next_param), 0)
        later = time.time()
        difference = (later - now)
        print("time taken for one BO iteration: ", difference)
    print(fitting_error_count)
    return all_best_losses, input_param, best_param

def BO_graph_turbo(system : DirectedFunctionalGraph, printout=True, iteration=50, lower_bound = -1, upper_bound=1, batch_size=4, to_normalize_y=False):
    parameter_trials = torch.tensor((np.array(system.get_all_params()[1])).flatten(), dtype=dtype).unsqueeze(0)
    def get_initial_points(dim, n_pts, seed=0):
        sobol = SobolEngine(dimension=dim, scramble=True, seed=seed)
        X_init = sobol.draw(n=n_pts).to(dtype=torch.double, device=device)
        return X_init

    all_best_losses = []
    X_turbo = torch.tensor((np.array(system.get_all_params()[1])).flatten(), dtype=torch.double).unsqueeze(0)
    X_turbo = get_initial_points(parameter_trials.shape[-1], 1, seed=1)
    Y_turbo = torch.tensor(
    [- system.get_system_loss_with_params(x) for x in X_turbo], dtype=torch.double, device="cpu").unsqueeze(-1)
    dim = len(system.get_all_params()[1])

    state = TurboState(dim, batch_size=batch_size)

    while not state.restart_triggered:  # Run until TuRBO converges
        # Fit a GP model
        if to_normalize_y:
            train_Y = torch.nn.functional.normalize(Y_turbo).double()
        else:
            train_Y = Y_turbo.double()

        likelihood = GaussianLikelihood()
        covar_module = ScaleKernel(  # Use the same lengthscale prior as in the TuRBO paper
            MaternKernel(
                nu=2.5, ard_num_dims=dim, lengthscale_constraint=Interval(0.005, 0.5)
            )
        )

        model = SingleTaskGP(
            X_turbo, train_Y, covar_module=covar_module, likelihood=likelihood
        )
        mll = ExactMarginalLogLikelihood(model.likelihood, model)

        # Do the fitting and acquisition function optimization inside the Cholesky context
        max_cholesky_size = float("inf")
        with gpytorch.settings.max_cholesky_size(max_cholesky_size):
            # Fit the model
            try:
                fit_gpytorch_mll(mll)
            except:
                print("error in fitting")

            # Create a batch
            X_next = generate_batch(
                state=state,
                model=model,
                X=X_turbo,
                Y=train_Y,
                batch_size=batch_size,
                n_candidates=20,
                num_restarts=5,
                raw_samples=24,
                acqf="ts",
            )
        bounds = torch.stack([torch.ones(parameter_trials.shape[-1]) * lower_bound, upper_bound * torch.ones(parameter_trials.shape[-1])])
        # print("proposed batch: ")
        # print(X_next)
        # print([unnormalize(x, bounds) for x in X_next])
        Y_next = torch.tensor(
            [ - system.get_system_loss_with_params(unnormalize(x, bounds)) for x in X_next], dtype=dtype, device=device
        ).unsqueeze(-1)
        
        # Update state
        state = update_state(state=state, Y_next=Y_next)

        # Append data
        X_turbo = torch.cat((X_turbo, X_next), dim=0)
        Y_turbo = torch.cat((Y_turbo, Y_next), dim=0)

        # Print current status
        print(
            f"{len(X_turbo)}) Best value: {state.best_value:.2e}, TR length: {state.length:.2e}"
        )
        for x in range(batch_size):
            all_best_losses.append(state.best_value)

    return all_best_losses, None, None


def BO_graph_local_loss(system : DirectedFunctionalGraph, bounds: torch.tensor, method, samples, printout=True, num_starting_points= 5, to_normalize_y=False, iteration=50, ignore_error=False):    
    next_param = system.get_local_losses()
    target = [[-system.get_system_loss()]]

    input_param = next_param
    best_param = None
    best_objective = -10000
    all_best_losses = []
    for i in range(iteration):
        now = time.time()
        print("BO iteration: ", i)
        print("Current best objective: ", best_objective)

        ###
        # assign param which reverse maps to local loss
        current_loss = - system.get_system_loss()
        if i != 0:
            current_loss  = -system.reverse_local_loss_lookup(next_param[0], method, samples, to_plot=printout, num_starting_points=num_starting_points, ignore_failure=ignore_error)
        ###
        # current_loss = - system.get_system_loss()
        if current_loss > best_objective:
            best_objective = current_loss
            best_param = system.get_all_params()[1] 
        all_best_losses.append(best_objective)
        
        UCB = None
        gp = None
        print("target loss:", next_param[0])
        input_param = torch.cat((input_param, system.get_local_losses()), 0)
        target.append([current_loss])
        Y = torch.DoubleTensor(target)
        if to_normalize_y:
            Y = torch.nn.functional.normalize(Y)

        # parameterize standardization and model
        # target = standardize(target)

        gp = SingleTaskGP(input_param.double(), Y.double())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fitting_error_count = 0
        try:
            fit_gpytorch_mll(mll)
        except:
            fitting_error_count+=1
        UCB = UpperConfidenceBound(gp, beta=0.1)

        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds.double(), q=1, num_restarts=1, raw_samples=20,
        )
        next_param = candidate
        if printout:
            print("candidate param: ", next_param)
        later = time.time()
        difference = (later - now)
        print("time taken for one BO iteration: ", str(difference))
    print(fitting_error_count)
    return all_best_losses, best_param, candidate_loss_all

def BO_graph_local_loss_with_explicit_noise(system : DirectedFunctionalGraph, bounds: torch.tensor, method, samples, num_starting_points = 10, printout=True, iteration=50):
    all_params = []
    for node in system.nodes:
        if "Dummy" in str(node) or "Blackbox" in str(node):
            continue
    
    next_param = system.get_local_losses()
    target = [[-system.get_system_loss()]]

    input_param = next_param
    best_param = None
    best_objective = -10000
    all_best_losses = []
    for i in range(iteration):
        now = time.time()
        if printout:
            print("BO iteration: ", i)
            print("Current best objective: ", best_objective)

        ###
        # assign param which reverse maps to local loss
        current_loss = - system.get_system_loss()
        if i != 0:
            current_loss = - system.reverse_local_loss_lookup(next_param[0], method, samples, num_starting_points=num_starting_points)
        ###

        # current_loss = - system.get_system_loss()
        if current_loss > best_objective:
            best_objective = current_loss
            best_param = system.get_all_params()[1] 
        all_best_losses.append(best_objective)
        
        UCB = None
        gp = None
        print("target loss:", next_param[0])
        input_param = torch.cat((input_param, system.get_local_losses()), 0)
        target.append([current_loss])
        Y = torch.DoubleTensor(target)

        # parameterize standardization and model
        # target = standardize(target)
        train_Yvar = 0.01 * torch.rand_like(Y)
        gp = HeteroskedasticSingleTaskGP(input_param.double(), Y.double(), train_Yvar.double())
        mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
        fitting_error_count = 0
        try:
            fit_gpytorch_mll(mll)
        except:
            fitting_error_count+=1
        UCB = UpperConfidenceBound(gp, beta=0.1)

        candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds.double(), q=1, num_restarts=1, raw_samples=20,
        )
        next_param = candidate
        if printout:
            print("candidate param: ", next_param)
        later = time.time()
        difference = (later - now)
        print("time taken for one BO iteration: ", str(difference))
    print(fitting_error_count)
    return all_best_losses, best_param