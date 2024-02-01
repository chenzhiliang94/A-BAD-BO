import os
import random

from LLM.helper import *
from LLM.BO import run_BO_GP_UCB


os.environ["TOKENIZERS_PARALLELISM"] = "false"
sub_tasks = ['antonyms', 'diff', 'first_word_letter',
             'informal_to_formal', 'larger_animal', 'letters_list', 'taxonomy_animal', 'negation', 'num_to_verbal',
             'active_to_passive', 'singular_to_plural', 'rhymes',
             'second_word_letter', 'sentence_similarity', 'sentiment', 'orthography_starts_with',
             'sum', 'synonyms', 'translation_en-de', 'translation_en-es',
             'translation_en-fr', 'word_in_context']
#openai.api_key = ""

BO_iterations=50
trials=2
result = run_BO_GP_UCB(BO_iterations, trials, sub_tasks, contamination_rate=0.5)
output_dir = "result/llm_vanilla_bo.csv"
result = np.array(result)
np.savetxt(output_dir, result)