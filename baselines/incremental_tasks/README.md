
# Class incremental experiments for LTLZinc
Baseline experiments for class incremental tasks 1-4 presented in our paper.

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Description

## Requirements

```
numpy==2.2.4
pandas==2.2.3
PyYAML==6.0.1
PyYAML==6.0.2
quadprog==0.1.13
torch==2.3.0a0+git372d078
torcheval==0.0.7
torchvision==0.21.0
tqdm==4.66.2

```

## Usage

| Argument                 | Values                                                | Default         | Description                                                                                              |
|--------------------------|-------------------------------------------------------|-----------------|----------------------------------------------------------------------------------------------------------|
| --prefix_path            | str                                                   | ../..           | Path to the root of the project                                                                          |
| --annotations_path       | str                                                   | benchmark/tasks | Path to the annotation folder, relative to root                                                          |
| --output_path            | str                                                   | outputs         | Path to the output folder, relative to root                                                              |
| --use_random_samples     | bool                                                  | False           | Ignore annotated image paths and sample new ones with the same label                                     |
| --task                   | str                                                   | task1           | Name of the task to solve                                                                                |
| --task_seed              | int                                                   | 12345           | Task seed to use                                                                                         |
| --lr                     | float                                                 | -0.0001         | Learning rate (positive: SGD, negative: Adam)                                                            |
| --epochs                 | int                                                   | 10              | Training epochs                                                                                          |
| --shuffle                | bool                                                  | True            | Shuffle training samples before each epoch                                                               |
| --batch_size             | int                                                   | 32              | Batch size                                                                                               |
| --buffer_batch_size      | int                                                   | 4               | Batch size for experience replay                                                                         |
| --eval_batch_size        | int                                                   | 128             | Batch size for evaluation                                                                                |
| --supervision_lambda     | float                                                 | 1.0             | Weight for direct supervision                                                                            |
| --replay_lambda          | float                                                 | 1.0             | Weight for experience replay loss                                                                        |
| --distil_lambda          | float                                                 | 1.0             | Weight for distillation loss                                                                             |
| --grad_clipping          | float                                                 | 0.0             | Norm for gradient clipping, disable if 0.0                                                               |
| --train_on_irrelevant    | bool                                                  | False           | Apply the loss also on irrelevant annotations                                                            |
| --buffer_size            | int                                                   | 200             | Replay buffer size                                                                                       |
| --buffer_type            | none, ring, reservoir                                 | none            | Type of replay buffer in {'none', 'ring', 'reservoir'}                                                   |
| --buffer_loss            | ce, gem                                               | ce              | Type of experience replay loss in {'ce', 'gem'}                                                          |
| --distillation_source    | none, batch, buffer, both                             | none            | Data source for distillation in {'none', 'batch', 'buffer', 'both'} (distillation is disabled if 'none') |
| --modular_net            | bool                                                  | False           | Allocate a different layer for each concept                                                              |
| --backbone_module        | squeezenet11, shufflenetv2x05, googlenet, densenet121 | cnn             | Perceptual backbone in {'squeezenet11', 'shufflenetv2x05', 'googlenet', 'densenet121'}                   |
| --pretrained_weights     | bool                                                  | False           | If true, initialize backbone with ImageNet weights                                                       |
| --emb_size               | int                                                   | 128             | Embedding size of the backbone                                                                           |
| --hidden_size            | int                                                   | 32              | Embedding size of each of the hidden layers                                                              |
| --knowledge_availability | none, predicates, states                              | none            | Knowledge availability in {'none', 'predicates', 'states'}                                               |
| --seed                   | str                                                   | -1              | Integer seed for random generator (if negative, use timestamp)                                           |
| --epoch_timeout          | int                                                   | 0               | Timeout for each epoch, in minutes (disable if 0)                                                        |
| --device                 | str                                                   | cpu             | Device to use, no effect if environment variable DEVICE is set                                           |
| --save                   | bool                                                  | False           | Save network weights and results locally                                                                 |
| --abort_irrelevant       | bool                                                  | True            | Abort irrelevant combinations of hyper-parameters (useful when sweeping grid searches)                   |
| --verbose                | 0, 1, 2, 3, 4                                         | 1               | Amount of output (0: no output, 1: task summary, 2: epoch summary, 3: metrics eval summary, 4: full)     |
| --wandb_project          | str                                                   |                 | Use W&B, this is the project name                                                                        |
| --eval_train             | bool                                                  | False           | If true, compute metrics also for the training set (slow)                                                |
| --eps                    | float                                                 | 0.001           | Epsilon for numerical stability                                                                          |
| --neginf                 | float                                                 | -10000000000.0  | Negative infinity for numerical stability                                                                |
| --vanishing_threshold    | float                                                 | 1e-05           | Threshold triggering a vanishing gradient tag                                                            |
| --exploding_threshold    | float                                                 | 10000000.0      | Threshold triggering an exploding gradient tag                                                           |
| --std_threshold          | float                                                 | 200.0           | Threshold triggering an high gradient standard deviation tag                                             |
| --heartbeat              | int                                                   | 10              | Heartbeat for the watchdog timer in seconds                                                              |

