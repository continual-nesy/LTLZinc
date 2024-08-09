
# Sequential experiments for LTLZinc
Baseline experiments for tasks 1-8 presented in our paper.


[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Description

## Requirements

```
Arpeggio==2.0.2
nnf==0.4.1
numpy==2.0.0
pandas==2.2.2
PyYAML==6.0.1
PyYAML==6.0.1
scallopy==0.1.0
torch==2.3.0a0+git39ae4d8
torcheval==0.0.7
torchvision==0.18.1
tqdm==4.66.1

```

You need to have the [DSHARP knowledge compiler](https://github.com/QuMuLab/dsharp) executable available from your `$PATH`.
```commandline
apt-get install libgmp-dev
git clone https://github.com/QuMuLab/dsharp
cd dsharp
mv Makefile_gmp Makefile
make
mv dsharp $CONDA_PREFIX/bin/
```

## Usage

| Argument                | Values                                                                                   | Default         | Description                                                                                       |
|-------------------------|------------------------------------------------------------------------------------------|-----------------|---------------------------------------------------------------------------------------------------|
| --prefix_path           | str                                                                                      | ../..           | Path to the root of the project                                                                   |
| --annotations_path      | str                                                                                      | benchmark/tasks | Path to the annotation folder, relative to root                                                   |
| --output_path           | str                                                                                      | outputs         | Path to the output folder, relative to root                                                       |
| --task                  | str                                                                                      | task1           | Name of the task to solve                                                                         |
| --lr                    | float                                                                                    | -0.001          | Learning rate (positive: SGD, negative: Adam)                                                     |
| --epochs                | int                                                                                      | 10              | Training epochs                                                                                   |
| --pretraining_epochs    | int                                                                                      | 0               | Pre-training epochs, these will apply only supervision loss with a fixed lambda=1.0               |
| --batch_size            | int                                                                                      | 32              | Batch size                                                                                        |
| --supervision_lambda    | float                                                                                    | 0.0             | Weight for direct supervision                                                                     |
| --constraint_lambda     | float                                                                                    | 0.0             | Weight for constraint supervision                                                                 |
| --successor_lambda      | float                                                                                    | 0.0             | Weight for next-state supervision                                                                 |
| --sequence_lambda       | float                                                                                    | 1.0             | Weight for end-to-end sequence supervision                                                        |
| --sequence_loss         | bce, fuzzy                                                                               | bce             | Type of loss function for sequence classification in {'bce', 'fuzzy'}                             |
| --teacher_forcing       | bool                                                                                     | False           | Force ground truth next state during training (False) or let the network evolve on its own (True) |
| --detach_previous_state | bool                                                                                     | False           | During backpropagation detach the computation graph at each time-step                             |
| --train_on_irrelevant   | bool                                                                                     | False           | Apply the loss also on irrelevant annotations                                                     |
| --grad_clipping         | float                                                                                    | 0.0             | Norm for gradient clipping, disable if 0.0                                                        |
| --backbone_module       | independent, dataset                                                                     | dataset         | Perceptual backbone in {'independent', 'dataset'}                                                 |
| --constraint_module     | mlp:NUM_HIDDEN_NEURONS, scallop:PROVENANCE:TRAIN_K:TEST_K                                | mlp:8           | Constraint module                                                                                 |
| --dfa_module            | mlp:HIDDEN_NEURONS, gru:HIDDEN_NEURONS, scallop:PROVENANCE:TRAIN_K:TEST_K, prob, logprob | mlp:8           | Automaton module                                                                                  |
| --scallop_e2e           | bool                                                                                     | False           | Use an End-to-End Scallop program for constraints and automaton, instead of two disjoint programs |
| --seed                  | str                                                                                      | -1              | Integer seed for random generator (if negative, use timestamp)                                    |
| --epoch_timeout         | int                                                                                      | 0               | Timeout for each epoch, in minutes (disable if 0)                                                 |
| --device                | str                                                                                      | cpu             | Device to use, no effect if environment variable DEVICE is set                                    |
| --save                  | bool                                                                                     | False           | Save network weights and results locally                                                          |
| --abort_irrelevant      | bool                                                                                     | True            | Abort irrelevant combinations of hyper-parameters (useful when sweeping grid searches)            |
| --wandb_project         | str                                                                                      |                 | Use W&B, this is the project name                                                                 |
| --eps                   | float                                                                                    | 1e-07           | Epsilon for numerical stability                                                                   |
| --neginf                | float                                                                                    | -10000000000.0  | Negative infinity for numerical stability                                                         |
| --overfitting_threshold | float                                                                                    | 0.5             | Threshold triggering an overfitting tag                                                           |
| --vanishing_threshold   | float                                                                                    | 1e-05           | Threshold triggering a vanishing gradient tag                                                     |
| --exploding_threshold   | float                                                                                    | 10000000.0      | Threshold triggering an exploding gradient tag                                                    |
| --std_threshold         | float                                                                                    | 200.0           | Threshold triggering an high gradient standard deviation tag                                      |
| --rnd_threshold         | float                                                                                    | 0.1             | Threshold triggering a random guessing tag                                                        |
| --mp_threshold          | float                                                                                    | 0.1             | Threshold triggering a most probable guessing tag                                                 |
| --heartbeat             | int                                                                                      | 10              | Heartbeat for the watchdog timer in seconds                                                       |

