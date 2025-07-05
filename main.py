import yaml
import tqdm.auto as tqdm
import multiprocessing

from utils import parse_config, seed
from dataset import SequenceGenerator, TaskGenerator
import os
import traceback

# Simple generator for ready-made datasets. It expects datasets to have been downloaded into a data/ subfolder.
# In case you want to generate new datasets, either write a yml specification and put it in the benchmark/specs folder,
# or import a generator and write your own generation loop.

def generate_task(filename, silent=False):
    try:
        config = parse_config("benchmark/specs/{}".format(filename))
        for s in config["seed"]:
            name = "{}/{}".format("".join(filename.split(".")[:-1]), s)
            os.makedirs("benchmark/tasks/{}".format(name), exist_ok=True)

            rng = seed(s)

            if config["mode"] == "sequential":
                generator = SequenceGenerator(config, rng, silent=silent)

                for split in config["splits"].keys():
                    dataset = generator.get_dataset(split, balance=True)
                    dataset.to_csv("benchmark/tasks/{}/sequence_{}.csv".format(name, split), index_label="sample_id")

            elif config["mode"] == "sequential_positive_only":
                generator = SequenceGenerator(config, rng, positive_only=True, silent=silent)

                for split in config["splits"].keys():
                    dataset = generator.get_dataset(split)
                    dataset.to_csv("benchmark/tasks/{}/sequence_pos_only_{}.csv".format(name, split))

            else:
                generator = TaskGenerator(config, rng, silent=silent)

                for split in config["splits"].keys():
                    dataset = generator.get_dataset(split)
                    dataset.to_csv("benchmark/tasks/{}/incremental_{}.csv".format(name, split), index_label="sample_id")

            str_automaton, dict_automaton, mappings = generator.dfa.export_automaton()

            with open("benchmark/tasks/{}/automaton.txt".format(name), "w") as file:
                file.write(str_automaton)
            with open("benchmark/tasks/{}/automaton.yml".format(name), "w") as file:
                yaml.dump(dict_automaton, file)
            with open("benchmark/tasks/{}/mappings.yml".format(name), "w") as file:
                yaml.dump(mappings, file)

            type_values = {k: sorted(([kk for kk in v.keys()] if isinstance(v,dict) else v)) for k, v in config["types"].items()}
            domain_values = {}
            for var, v in config["domains"].items():
                for split, var_type in v.items():
                    if split not in domain_values:
                        domain_values[split] = {}
                    if var_type in type_values:
                        domain_values[split][var] = type_values[var_type]
                    else:
                        domain_values[split][var] = None

            with open("benchmark/tasks/{}/domain_values.yml".format(name), "w") as file:
                yaml.dump(domain_values, file)
    except Exception: # Base class Exception is serializable, derived exceptions in imported libraries are not, so we "upcast" the exception and re-raise it.
        raise Exception("Error processing file {}\n{}".format(filename, traceback.format_exc()))



if __name__ == "__main__":
    assert os.path.exists("data"), "Dataset folder 'data' does not exist. Have you run download_images.py?"

    # Disable yaml aliases, this produces larger, but more human-readable, files.
    yaml.Dumper.ignore_aliases = lambda *args: True

    # Generate each task concurrently on a process pool.
    #tasks = sorted([os.path.join(dp, f).split("benchmark/specs")[1] for dp, dn, fn in os.walk("benchmark/specs") for f in fn if f.endswith(".yml")])
    #tasks = ["task_incremental/task2.yml"]
    #tasks = ["example_sec5.yml"]

    #tasks = sorted([os.path.join(dp, f).split("benchmark/specs/")[1] for dp, dn, fn in os.walk("benchmark/specs/safety_patterns") for f in fn if f.endswith(".yml")])
    #tasks = ["ansya/safety_001.yml", "ansya/safety_002.yml", "ansya/safety_003.yml"]
    tasks = ["mnist_add.yml"]

    with multiprocessing.Pool() as p:
        for _ in tqdm.tqdm(p.imap_unordered(generate_task, tasks), total=len(tasks)):
            pass
