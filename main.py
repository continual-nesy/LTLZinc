import yaml
import tqdm

from utils import parse_config, seed
from dataset import SequenceGenerator, TaskGenerator
import os

# Simple generator for ready-made datasets. It expects datasets to have been downloaded into a data/ subfolder.
# In case you want to generate new datasets, either write a yml specification and put it in the benchmark/specs folder,
# or import a generator and write your own generation loop.

if __name__ == "__main__":
    assert os.path.exists("data"), "Dataset folder 'data' does not exist. Have you run download_images.py?"

    # Disable yaml aliases, this produces larger files, but which are more human-readable.
    yaml.Dumper.ignore_aliases = lambda *args : True

    for file in (bar := tqdm.tqdm(sorted([f for f in os.listdir("benchmark/specs") if f.endswith(".yml")]), position=0, desc="dataset", leave=False)):
        bar.set_postfix_str(file)
        name = "".join(file.split(".")[:-1])
        os.makedirs("benchmark/tasks/{}".format(name), exist_ok=True)

        config = parse_config("benchmark/specs/{}".format(file))
        rng = seed(config["seed"])

        if config["mode"] == "sequential":
            generator = SequenceGenerator(config, rng)

            for split in config["splits"].keys():
                dataset = generator.get_dataset(split, balance=True)
                dataset.to_csv("benchmark/tasks/{}/sequence_{}.csv".format(name, split), index_label="sample_id")

        elif config["mode"] == "sequential_positive_only":
            generator = SequenceGenerator(config, rng, positive_only=True)

            for split in config["splits"].keys():
                dataset = generator.get_dataset(split)
                dataset.to_csv("benchmark/tasks/{}/sequence_pos_only_{}.csv".format(name, split))

        else:
            generator = TaskGenerator(config, rng)

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

        type_values = {k: sorted([kk for kk in v.keys()]) for k, v in config["types"].items()}
        domain_values = {}
        for var, v in config["domains"].items():
            for split, var_type in v.items():
                if split not in domain_values:
                    domain_values[split] = {}
                domain_values[split][var] = type_values[var_type]

        with open("benchmark/tasks/{}/domain_values.yml".format(name), "w") as file:
            yaml.dump(domain_values, file)