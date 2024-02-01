from LLM.BO import *
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

sub_tasks = ['antonyms', 'diff', 'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']
import warnings
warnings.filterwarnings("ignore")

f = open('params/LLM.json')
param = json.load(f)

# system related parameters
down_sample_size = param["down_sample_size"]
to_use_specific_model = False
contamination_rate=param["contamination_rate"]
epochs = param["epochs"]

# ABOLLO related parameters
total_trials = param["abollo_trial"] # trials
abollo_iterations = param["abollo_bo_iteration"] # bo iteration
samples = param["abollo_sample_size"] # k samples
bounds = torch.stack([torch.ones(2) * param["abollo_lower_bound"], torch.ones(2) * param["abollo_upper_bound"]])

# vanilla parameters
vanilla_bo_trials = param["vanilla_bo_trial"]
vanilla_bo_iterations = param["vanilla_bo_iteration"]

# turbo parameters
turbo_bo_trials=param["vanilla_bo_trial"]
turbo_bo_iterations=param["vanilla_bo_trial"]

contamination_rate = 0.5
# ABOLLO
for sample_k in samples:
    for itr in abollo_iterations: 
        task = "llm_pipeline_k" + str(sample_k) + "_itr_" + str(itr)
        units = sample_k * epochs * down_sample_size / 0.25
        print("predicted total time taken (hours): ", str(itr * units * 2400 / 3600))
        loss_space_bo_all_trials = llm_abollo(sub_tasks, bounds = bounds,
                            down_sample_size = down_sample_size, BO_iteration=itr, trials=total_trials, to_use_specific_model = to_use_specific_model,
                            sample_size=sample_k, epochs=epochs, contamination_rate=contamination_rate)
        results = np.array(loss_space_bo_all_trials)
        output_dir = "result/llm/" + str(task) + ".csv"
        np.savetxt(output_dir, results)

# turbo
# bo_all_trials = llm_turbo(sub_tasks, contamination_rate, to_normalize_y=True, total_trials=vanilla_bo_trials, iterations=turbo_bo_iterations)
# max_len = max([len(x) for x in bo_all_trials])
# for x in range(0, len(bo_all_trials)):
#     while len(bo_all_trials[x]) < max_len:
#         bo_all_trials[x].append(bo_all_trials[x][-1])
# output_dir = "result/llm_turbo.csv"
# results = np.array(bo_all_trials)
# np.savetxt(output_dir, results)

# # vanilla BO with GP UCB
# result = run_BO_GP_UCB(vanilla_bo_iterations, vanilla_bo_trials, sub_tasks)
# output_dir = "result/llm_vanilla_bo.csv"
# result = np.array(result)
# np.savetxt(output_dir, result)