# LTLZinc generator and benchmark

This repository contains the generator and datasets for the [LTLZinc Benchmark](https://arxiv.org/abs/2505.05106).

Task annotations are available in the release section, data can be downloaded from pytorch datasets, using the `download_images.py` utility.
Baseline experiments described in the paper are available in the `baselines` subfolder.

[![License: CC BY-NC-SA 4.0](https://licensebuttons.net/l/by-nc-sa/4.0/80x15.png)](https://creativecommons.org/licenses/by-nc-sa/4.0/)

## Introduction
One of the goals of neuro-symbolic artificial intelligence is to exploit background knowledge to improve the performance of learning tasks. However, most of the existing frameworks focus on the simplified scenario where knowledge does not change over time and does not cover the temporal dimension. 
LTLZinc is a benchmarking framework for knowledge-driven learning and reasoning over time, capable of generating arbitrary multi-channel sequences of images from an user-defined temporal specification.

Generated tasks can be used for **sequence-based learning and reasoning** (e.g., sequence classification with relational knwoledge, distant supervision, constraint induction over time, sequence induction, etc.) and **incremental learning** (e.g., curriculum learning, past/future-aware incremental learning, long-horizon continual learning, etc.).

## LTLZinc generator
The generator consists of a finite-domain constraint solver (MiniZinc) and an automaton-based sampler, which jointly generate datasets for temporal-related tasks by sampling a collection of image classification datasets. 

Each task is characterized by four main components:
- One or more finite domains (data types), mappping sets of images to classes
- A set of channels, mapping domains to perceptual stimuli (the inputs of the agent under training)
- A vocabulary of constraints, describing properties between channels
- A linear temporal logic formula over finite traces (LTLf), specifying which constraints hold/are violated at specific times.

Tasks are specified via `.yaml` files, the file `example_config.yaml` documents both mandatory and optional fields.

## Examples

### It will always be the case that $Y$ is less than $Z$ if and only if in two steps ahead, $V$, $W$ and $X$ all belong to the same class (Sequential)
```
mode: sequential

length: [10, 20]

splits:
  train: {path: "data/train", samples: 320}
  val:   {path: "data/train", samples: 40}
  test:  {path: "data/test", samples: 40}

minizinc_prefix: |
  include "all_equal.mzn";

predicates:
  "p(A,B)": "A < B"
  "q(A,B,C)": "(all_equal([A,B,C]))"

formula: "G (p(Y,Z) <-> X X q(V,W,X) ) "

types:
  fmnist_t:
    bag: "fmnist/bag"
    boot: "fmnist/boot"
    coat: "fmnist/coat"
    dress: "fmnist/dress"
    pullover: "fmnist/pullover"
    sandal: "fmnist/sandal"
    shirt: "fmnist/shirt"
    sneaker: "fmnist/sneaker"
    top: "fmnist/top"
    trouser: "fmnist/trouser"

  fmnist_5_t:
    sandal: "fmnist/sandal"
    shirt: "fmnist/shirt"
    sneaker: "fmnist/sneaker"
    top: "fmnist/top"
    trouser: "fmnist/trouser"

domains:
  Y: fmnist_t
  Z: fmnist_t
  V: fmnist_5_t
  W: fmnist_5_t
  X: fmnist_5_t

streams:
  V: v
  W: w
  X: x
  Y: y
  Z: z

avoid_states:
  accepting_sinks: False
  rejecting_sinks: False
  self_loops: { "linear": 0.1 }

truncate_on_sinks:
  accepting: False
  rejecting: False

seed: 12345

```

### The sequence alternates between satisfying $W + X = Y + Z$ and violating it  (Sequential)
```
mode: sequential

length: [10, 20]

splits:
  train: {path: "data/train", samples: 320}
  val:   {path: "data/train", samples: 40}
  test:  {path: "data/test", samples: 40}

minizinc_prefix: ""

predicates:
  "p(A,B,C,D)": "A + B = C + D"

formula: "G (p(W,X,Y,Z) <-> WX !p(W,X,Y,Z))"


types:
  mnist_t:
    0: "mnist/0"
    1: "mnist/1"
    2: "mnist/2"
    3: "mnist/3"
    4: "mnist/4"
    5: "mnist/5"
    6: "mnist/6"
    7: "mnist/7"
    8: "mnist/8"
    9: "mnist/9"

domains:
  W: mnist_t
  X: mnist_t
  Y: mnist_t
  Z: mnist_t

streams:
  W: w
  X: x
  Y: y
  Z: z


avoid_states:
  accepting_sinks: { "linear": 0.01 }
  rejecting_sinks: { "linear": 0.01 }
  self_loops: { "linear": 0.1 }

truncate_on_sinks:
  accepting: False
  rejecting: False

seed: 12345
```

### Plant labels are observed only within a single task out of 30 total (Incremental)

**Note:** This specification contains features still under development.

```
mode: incremental

length: 30

splits:
  train: {path: "data/train", samples: 800}
  val:   {path: "data/train", samples: 100}
  test:  {path: "data/test", samples: 100}

minizinc_prefix: |
  set of cifar100_t: acquatic_mammals = {beaver, dolphin, otter, seal, whale};
  set of cifar100_t: fish = {aquarium_fish, flatfish, ray, shark, trout};
  set of cifar100_t: flowers = {orchid, poppy, rose, sunflower, tulip};
  set of cifar100_t: food_containers = {bottle, bowl, can, cup, plate};
  set of cifar100_t: fruit_and_vegetables = {apple, mushroom, orange, pear, sweet_pepper};
  set of cifar100_t: household_electrical_devices = {clock, keyboard, lamp, telephone, television};
  set of cifar100_t: household_furniture = {bed, chair, couch, table, wardrobe};
  set of cifar100_t: insects = {bee, beetle, butterfly, caterpillar, cockroach};
  set of cifar100_t: large_carnivores = {bear, leopard, lion, tiger, wolf};
  set of cifar100_t: large_manmade_outdoors_things = {bridge, castle, house, road, skyscraper};
  set of cifar100_t: large_natural_outdoors_scenes = {cloud, forest, mountain, plain, sea};
  set of cifar100_t: large_omnivores_and_herbivores = {camel, cattle, chimpanzee, elephant, kangaroo};
  set of cifar100_t: medium_sized_mammals = {fox, porcupine, possum, raccoon, skunk};
  set of cifar100_t: non_insect_invertebrates = {crab, lobster, snail, spider, worm};
  set of cifar100_t: people = {baby, boy, girl, man, woman};
  set of cifar100_t: reptiles = {crocodile, dinosaur, lizard, snake, turtle};
  set of cifar100_t: small_mammals = {hamster, mouse, rabbit, shrew, squirrel};
  set of cifar100_t: trees = {maple_tree, oak_tree, palm_tree, pine_tree, willow_tree};
  set of cifar100_t: vehicles_1 = {bicycle, bus, motorcycle, pickup_truck, train};
  set of cifar100_t: vehicles_2 = {lawn_mower, rocket, streetcar, tank, tractor};
  

predicates:
  "animals(X)": "X in acquatic_mammals union fish union insects union large_carnivores union large_omnivores_and_herbivores union medium_sized_mammals union non_insect_invertebrates union people union reptiles union small_mammals"
  "plants(X)": "X in flowers union fruit_and_vegetables union trees"
  "inanimate(X)": "X in food_containers union household_electrical_devices union household_furniture union large_manmade_outdoors_things union large_natural_outdoors_scenes union vehicles_1 union vehicles_2"


formula: "(!plants(X) U (plants(X) & WX G !plants(X)))"

#  Plants observed only once.

types:
  cifar100_t:
    apple: "cifar100/apple"
    aquarium_fish: "cifar100/aquarium_fish"
    baby: "cifar100/baby"
    bear: "cifar100/bear"
    beaver: "cifar100/beaver"
    bed: "cifar100/bed"
    bee: "cifar100/bee"
    beetle: "cifar100/beetle"
    bicycle: "cifar100/bicycle"
    bottle: "cifar100/bottle"
    bowl: "cifar100/bowl"
    boy: "cifar100/boy"
    bridge: "cifar100/bridge"
    bus: "cifar100/bus"
    butterfly: "cifar100/butterfly"
    camel: "cifar100/camel"
    can: "cifar100/can"
    castle: "cifar100/castle"
    caterpillar: "cifar100/caterpillar"
    cattle: "cifar100/cattle"
    chair: "cifar100/chair"
    chimpanzee: "cifar100/chimpanzee"
    clock: "cifar100/clock"
    cloud: "cifar100/cloud"
    cockroach: "cifar100/cockroach"
    couch: "cifar100/couch"
    crab: "cifar100/crab"
    crocodile: "cifar100/crocodile"
    cup: "cifar100/cup"
    dinosaur: "cifar100/dinosaur"
    dolphin: "cifar100/dolphin"
    elephant: "cifar100/elephant"
    flatfish: "cifar100/flatfish"
    forest: "cifar100/forest"
    fox: "cifar100/fox"
    girl: "cifar100/girl"
    hamster: "cifar100/hamster"
    house: "cifar100/house"
    kangaroo: "cifar100/kangaroo"
    keyboard: "cifar100/keyboard"
    lamp: "cifar100/lamp"
    lawn_mower: "cifar100/lawn_mower"
    leopard: "cifar100/leopard"
    lion: "cifar100/lion"
    lizard: "cifar100/lizard"
    lobster: "cifar100/lobster"
    man: "cifar100/man"
    maple_tree: "cifar100/maple_tree"
    motorcycle: "cifar100/motorcycle"
    mountain: "cifar100/mountain"
    mouse: "cifar100/mouse"
    mushroom: "cifar100/mushroom"
    oak_tree: "cifar100/oak_tree"
    orange: "cifar100/orange"
    orchid: "cifar100/orchid"
    otter: "cifar100/otter"
    palm_tree: "cifar100/palm_tree"
    pear: "cifar100/pear"
    pickup_truck: "cifar100/pickup_truck"
    pine_tree: "cifar100/pine_tree"
    plain: "cifar100/plain"
    plate: "cifar100/plate"
    poppy: "cifar100/poppy"
    porcupine: "cifar100/porcupine"
    possum: "cifar100/possum"
    rabbit: "cifar100/rabbit"
    raccoon: "cifar100/raccoon"
    ray: "cifar100/ray"
    road: "cifar100/road"
    rocket: "cifar100/rocket"
    rose: "cifar100/rose"
    sea: "cifar100/sea"
    seal: "cifar100/seal"
    shark: "cifar100/shark"
    shrew: "cifar100/shrew"
    skunk: "cifar100/skunk"
    skyscraper: "cifar100/skyscraper"
    snail: "cifar100/snail"
    snake: "cifar100/snake"
    spider: "cifar100/spider"
    squirrel: "cifar100/squirrel"
    streetcar: "cifar100/streetcar"
    sunflower: "cifar100/sunflower"
    sweet_pepper: "cifar100/sweet_pepper"
    table: "cifar100/table"
    tank: "cifar100/tank"
    telephone: "cifar100/telephone"
    television: "cifar100/television"
    tiger: "cifar100/tiger"
    tractor: "cifar100/tractor"
    train: "cifar100/train"
    trout: "cifar100/trout"
    tulip: "cifar100/tulip"
    turtle: "cifar100/turtle"
    wardrobe: "cifar100/wardrobe"
    whale: "cifar100/whale"
    willow_tree: "cifar100/willow_tree"
    wolf: "cifar100/wolf"
    woman: "cifar100/woman"
    worm: "cifar100/worm"

domains:
  X: cifar100_t

streams:
  X: +x

guarantee_constraint_sampling: both

seed: [0, 12345, 67890, 88888, 66666, 11111]

```


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