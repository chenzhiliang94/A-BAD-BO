This repository contains experiments for A-BAD-BO algorithm for optimizing systems in this paper: [Towards AutoAI: Optimizing a Machine Learning System with Black-box and Differentiable Components](https://openreview.net/forum?id=JVhUR8q27o&referrer=%5BAuthor%20Console%5D(%2Fgroup%3Fid%3DICML.cc%2F2024%2FConference%2FAuthors%23your-submissions))

To cite the paper, please use the following:
```
@inproceedings{
chen2024towards,
title={Towards Auto{AI}: Optimizing a Machine Learning System with Black-box and Differentiable Components},
author={Zhiliang Chen and Chuan-Sheng Foo and Bryan Kian Hsiang Low},
booktitle={Forty-first International Conference on Machine Learning},
year={2024},
url={https://openreview.net/forum?id=JVhUR8q27o}
}
```
## Requirements
`pip3 install -r requirements.txt`

## Running experiments for synthetic, mnist and healthcare system (each command will run A-BAD-BO with other baselines). No data download needed. Results will be stored in result/ directory.
- `python3 run_bo_new_healthcare.py`
- `python3 run_bo_new_mnist.py`
- `python3 run_bo_new_synthetic.py`
  
## Installing data for LLM system
- First, download the folder from https://github.com/keirp/automatic_prompt_engineer/tree/main/experiments/data/instruction_induction (the folder downloaded is instruction_induction/...) into this repository.
- `cd instruction_induction && mkdir prompt_picker_data && mkdir example_picker_data`
- `python3 get_llm_data.py` to generate the data needed for our LLM system used in our paper.

## Running experiments for LLM prompt engineering system (make sure data for LLM system is installed in previous step)
First, make sure you have a working openai API key and set it on the command line.
- `python3 run_BO_LLM_vanilla.py` (optimize LLM pipeline with vanilla BO with GP-LCB)
- `python3 run_BO_LLM_turbo.py` (optimize LLM pipeline with TuRBO)
- `python3 run_LLM_pipeline.py` (optimize LLM pipeline with A-BAD-BO, our algorithm)
