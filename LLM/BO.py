import os
import random

from LLM.helper import *
from botorch.fit import fit_gpytorch_mll
from SearchAlgorithm.AcquisitionFunction import *
from SearchAlgorithm.turbo import *

os.environ["TOKENIZERS_PARALLELISM"] = "false"
id2label = {0: "NEGATIVE", 1: "POSITIVE"}
label2id = {"NEGATIVE": 0, "POSITIVE": 1} 

def run_BO_GP_UCB(BO_iterations, trials, subtasks, contamination_rate):
    
    trial_accuracy = []
    for trial in range(trials): 
        model_example_picker = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
        ).to("cuda")
        model_prompt_picker = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
        ).to("cuda")
        all_best_accuracy = []
        X = []
        y = []
        x = torch.ones([3072])
        best_accuracy = -1
        for i in range(BO_iterations):
            start = time.time()
            
            # replace with last layer weight copy with BO
            with torch.no_grad():
                input = x[:1536].reshape([2,768])
                dict(model_example_picker.named_parameters())["classifier.weight"].data.copy_(input)
                input = x[1536:].reshape([2,768])
                dict(model_prompt_picker.named_parameters())["classifier.weight"].data.copy_(input)
            
            accuracy = get_accuracy_for_each_subtask(model_example_picker, model_prompt_picker, subtasks, contamination_rate)
            
            current_accuracy = sum(accuracy.values())/len(accuracy.values())
            best_accuracy = max(best_accuracy, current_accuracy)
            all_best_accuracy.append(best_accuracy)
            
            X.append(x)
            y.append([current_accuracy])
            gp = SingleTaskGP(torch.Tensor(torch.stack(X)), torch.Tensor(y))
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            UCB = UpperConfidenceBound(gp, beta=0.1)
            bounds = torch.stack([torch.ones(3072) * -1, 1 * torch.ones(3072)])
            candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20)
            x = candidate[0] # size 2 * 2 * 768
            end = time.time()
            print("time taken for one BO iteration: ", end-start)
            print("current best accuracy: ", best_accuracy)
        print("all best accuracy:", all_best_accuracy)
        trial_accuracy.append(all_best_accuracy)
    
    result = np.array(trial_accuracy)
    return result

def get_accuracy(x, model_example_picker, model_prompt_picker, sub_tasks, contamination_rate):
    # replace with last layer weight copy with BO
    with torch.no_grad():
        input = x[:1536].reshape([2,768])
        dict(model_example_picker.named_parameters())["classifier.weight"].data.copy_(input)
        input = x[1536:].reshape([2,768])
        dict(model_prompt_picker.named_parameters())["classifier.weight"].data.copy_(input)
    
    accuracy =  get_accuracy_for_each_subtask(model_example_picker, model_prompt_picker, sub_tasks, contamination_rate)
    acc = sum(accuracy.values())/len(accuracy.values())
    return acc

def llm_turbo(sub_tasks, contamination_rate, to_normalize_y=True, total_trials=3, iterations=50, batch_size=1):
    trial_acc=[]
    for trials in range(total_trials): 
        model_example_picker = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
        ).to("cuda")
        model_prompt_picker = AutoModelForSequenceClassification.from_pretrained(
            "distilbert-base-uncased", num_labels=2, id2label=id2label, label2id=label2id
        ).to("cuda")
        all_best_accuracy = []
        dim = 3072
        
        # initial value
        x = torch.ones([dim])
        X_turbo = torch.tensor(x, dtype=torch.double).unsqueeze(0)
        Y_turbo = torch.tensor(
        [get_accuracy(x, model_example_picker, model_prompt_picker, sub_tasks, contamination_rate) for x in X_turbo], dtype=torch.double, device="cpu").unsqueeze(-1)
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
            bounds = torch.stack([torch.ones(dim) * -1, 1 * torch.ones(dim)])
            Y_next = torch.tensor(
                [ get_accuracy(unnormalize(x, bounds), model_example_picker, model_prompt_picker, sub_tasks, contamination_rate) for x in X_next],device="cpu"
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
            print("all y (including 1st iteration): ", Y_turbo)
            print("all best acc: ", all_best_accuracy)
            for x in range(batch_size):
                all_best_accuracy.append(state.best_value)
            print("is restart triggered: ", state.restart_triggered)
            if len(all_best_accuracy) >= iterations:
                print("breaking because iteration reached.")
                print(iterations)
                break
    trial_acc.append(all_best_accuracy)
    return trial_acc

def llm_abollo(sub_tasks, bounds = torch.stack([torch.ones(2) * 0.01, torch.ones(2) * 0.1]),
                    down_sample_size = 0.05, BO_iteration=10, trials=3, to_use_specific_model = False, sample_size = 1, epochs = 10, contamination_rate=0.9):
    
    trials_best_accuracy = []
    for trial in range(trials):
        target_loss = [0.5, 0.5]
        best_accuracy = -1
        all_best_accuracy = []
        X = []
        y = []

        for i in range(BO_iteration):
            start = time.time()

            model_to_train_on = "all_task_combined"
            overall_accuracy = 0.0
            for k in range(sample_size):
                model_example_picker = train_model(model_to_train_on, "example_picker", target_loss[0], epochs=epochs, to_down_sample=True, down_sample_size=down_sample_size)
                model_prompt_picker = train_model(model_to_train_on, "prompt_picker", target_loss[1], epochs=epochs, to_down_sample=True, down_sample_size=down_sample_size)
                accuracy = get_accuracy_for_each_subtask(model_example_picker, model_prompt_picker, sub_tasks, contamination_rate=contamination_rate)
                acc_sampled = sum(accuracy.values())/len(accuracy.values())
                print("accuracy in each sample k: ", acc_sampled)
                overall_accuracy = max(overall_accuracy, acc_sampled)


            if overall_accuracy > best_accuracy:
                best_accuracy = overall_accuracy
            print("X: ", target_loss)
            print("current accuracy: ", overall_accuracy)
            print("current best: ", best_accuracy)
            all_best_accuracy.append(best_accuracy)
            
            X.append(target_loss)
            y.append([overall_accuracy])
            gp = SingleTaskGP(torch.DoubleTensor(X), torch.DoubleTensor(y).double())
            mll = ExactMarginalLogLikelihood(gp.likelihood, gp)
            fit_gpytorch_mll(mll)
            UCB = UpperConfidenceBound(gp, beta=1)
            candidate, acq_value = optimize_acqf(
            UCB, bounds=bounds, q=1, num_restarts=1, raw_samples=20)
            target_loss = candidate[0]
            end = time.time()
            print("time taken for one BO iteration: ", end-start)
        print("finished a trial of ABOLLO LLM pipeline. X, y: ")
        print(X)
        print(y)
        trials_best_accuracy.append(all_best_accuracy)
    return trials_best_accuracy