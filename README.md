# LTLZinc generator and benchmark

This repository contains the generator and datasets for the [LTLZinc Benchmark](https://duckduckgo.com).

Task annotations are available in the release section, data can be downloaded from pytorch datasets, using the `download_images.py` utility.
Baseline experiments described in the paper are available in the `baselines` subfolder.

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Introduction
TODO

## Datasets
TODO

## LTLZinc generator
TODO

## Examples
TODO

## Requirements

```
flloat==0.3.0
minizinc==0.9.0
numpy==2.0.0
pandas==2.2.2
PyYAML==6.0.1
PyYAML==6.0.1
sympy==1.12
torch==2.3.0a0+git39ae4d8
torchvision==0.18.1
tqdm==4.66.1
```

For the generator, install MiniZinc system-wide (e.g., with `apt-get install minizinc`).
Baseline experiments have different requirements (see `baselines/sequential_tasks/README.md` and `baselines/incremental_tasks/README.md`).


## How to use the generator (with the ready-made script)
1. Run `python download_images.py` to download the datasets, alternatively, put your images in `data/SPLIT_NAME/DATASET_NAME/CLASS/`
   The generator assumes a **multi-class single-label** image classification setting for each dataset
2. Create one or more configuration files and place them in `benchmark/tasks/specs/` (see `example_config.yml` for the syntax).
   Make sure the `splits` and `types` fields are coherent with your dataset structure
3. Run `python main.py` and get yourself a coffee, this will take a while

NOTE: The generator could in theory be used in an online fashion, however, as extracting solutions from the CSP problems is slow, we discourage calling it from within a training loop.

## How to use the datasets (Sequence-mode tasks)
1. Run `python download_images.py` to download the datasets
2. Load dataset annotations from `tasks/TASK_NAME/sequence_SPLIT.csv`
3. Load `tasks/TASK_NAME/domain_values.yml` and `tasks/TASK_NAME/mappings.yml` to map annotations to  values
4. If you need background knowledge, exploit the content of `tasks/TASK_NAME/automaton.yml`

| Column           | Content                                                                           |
|------------------|-----------------------------------------------------------------------------------|
| `sample_id`      | Unique identifier                                                                 |
| `seq_id`         | Sequence identifier (group rows by this value for a single sequence)              |
| `time`           | Time step                                                                         |
| `accepted`       | Sequence label (duplicated for each time step of the sequence)                    |
| `transition`     | Couple `previous_state->next_state`                                               |
| `img_STREAMNAME` | Path for the image corresponding to `STREAMNAME` at the given time step           |
| `out_STREAMNAME` | Label for `STREAMNAME` at the given time step. `STREAMNAME` is an output variable |
| `var_STREAMNAME` | Label for `STREAMNAME` at the given time step. `STREAMNAME` is an input variable  |
| `p_ID`           | Label for constraint `p_ID`                                                       |

Labels between parentheses (e.g., `var_x = (8)`) mean that the true label is known, but irrelevant for that time step.
Ideally, you should not train on these values, but you can measure accuracy on them if you want.

## How to use the datasets (Incremental-mode tasks)
1. Run `python download_images.py` to download the datasets
2. Load dataset annotations from `tasks/TASK_NAME/incremental_SPLIT.csv`
3. Load `tasks/TASK_NAME/domain_values.yml` and `tasks/TASK_NAME/mappings.yml` to map annotations to  values
4. The content of `tasks/TASK_NAME/automaton.yml` can be used to plan some continual learning strategy
   (e.g., to determine a set of candidates for experience replay in an informed fashion)

Columns are the same as Sequence-mode tasks, with the addition of `task_id`, which groups samples in a task-incremental way.
