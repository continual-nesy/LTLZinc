import yaml
import os
import re
import numpy as np

#import random
#import sympy

import datetime

def seed(seed):
    """
    Seed the various random generators.

    Important: flloat library is not affected by this function, and it has non-deterministic behavior!
    :param seed: The seed to use.
    :return: A seeded numpy.random.Generator.
    """
    #os.environ["PYTHONHASHSEED"] = str(seed) # Doesn't fix the problem in flloat
    #sympy.core.random.seed(seed) # Doesn't fix the problem in flloat
    rng = np.random.default_rng(seed=seed)
    #random.seed(seed) # Doesn't fix the problem in flloat

    return rng

def parse_config(filename):
    """
    Parse the configuration file, check it for errors and initialize optional fields with default values.
    :param filename: The yaml configuration file.
    :return: A dictionary of parsed options.
    """
    with open(filename, "r") as file:
        config = yaml.safe_load(file)

    for k in ["mode", "length", "formula", "predicates", "splits", "types", "domains", "streams"]:
        assert k in config, "Missing mandatory key {}.".format(k)

    # Optional keys:
    if "seed" not in config:
        config["seed"] = [datetime.datetime.now().timestamp()]

    if "minizinc_prefix" not in config:
        config["minizinc_prefix"] = ""

    if "avoid_states" not in config:
        config["avoid_states"] = {"accepting_sinks": False, "rejecting_sinks": False, "self_loops": False}

    if "truncate_on_sink" not in config:
        config["truncate_on_sink"] = {"accepting": False, "rejecting": False}

    if "guarantee_constraint_sampling" not in config:
        config["guarantee_constraint_sampling"] = False

    if "guarantee_mode" not in config:
        config["guarantee_mode"] = "strict_f"

    if "incremental_guarantee_ratio" not in config:
        config["incremental_guarantee_ratio"] = 1.0

    assert isinstance(config["seed"], int) or isinstance(config["seed"], list), "Random seed must be integer or list, found {}.".format(type(config["seed"]))
    if isinstance(config["seed"], list):
        for s in config["seed"]:
            assert isinstance(s, int), "Every element of the seed list must be integer, found {} ({}).".format(s, type(s))
    else:
        config["seed"] = [config["seed"]]

    assert config["mode"] in ["sequential", "incremental", "sequential_positive_only"], \
        "Unknown benchmark mode. Expected one of ['sequential', 'sequential_positive_only', 'incremental'], found {}.".format(config["mode"])
    assert isinstance(config["length"], int) or len(config["length"]) == 2, \
        "Expected either a constant length or a 2-element list (min, max), found {}.".format(config["length"])
    assert config["mode"] != "incremental" or isinstance(config["length"], int), \
        "Building a benchmark in incremental mode requires a constant length, found {}.".format(config["length"])

    if isinstance(config["length"], int):
        config["length"] = [config["length"], config["length"]]

    assert config["length"][0] <= config["length"][1], \
        "The maximum length must be greater or equal than the minimum, found {}.".format(config["length"])
    assert config["length"][0] > 0, "Lengths must be strictly positive, found {}.".format(config["length"])

    assert isinstance(config["predicates"], dict), \
        "Entry 'predicates' must be a dictionary, found {}.".format(type(config["predicates"]))
    sanitized_predicates = {}
    for k, v in config["predicates"].items():
        safe_k = re.sub(r'\s', '', k)
        assert re.match(r'[a-z][0-9a-z]*\([A-Z]+(,[A-Z]+)*\)', safe_k), \
            "Predicate names must match [a-z][0-9a-z]*, terms must match [A-Z]+, the structure must conform p(A, B, ..., Z). Found {}.".format(k)

        terms = re.match(r'.+\(([A-Z,]+)\)', safe_k).groups()[0].split(",")
        for t in terms:
            assert t in v, "Expected term {} to appear in expression '{}'.".format(t, v)

        sanitized_predicates[safe_k] = {"constraint": v, "terms": terms}

    config["predicates"] = sanitized_predicates

    vars = set()
    extracted_preds = set(re.findall(r'[a-z][0-9a-z]*\([A-Z,\s]+\)', config["formula"]))

    # Add variables from temporal specification.
    for pred in extracted_preds:
        for t in pred.split("(")[1].split(")")[0].split(","):
            vars.add(t.strip())

    # Add variables from orphaned predicates (these won't require substitutions).
    temporal_preds = {p.split("(")[0] for p in extracted_preds}
    orphans = {p.split("(")[0] for p in config["predicates"].keys()}.difference(temporal_preds)
    for p in set(config["predicates"].keys()):
        if p.split("(")[0] in orphans:
            for t in p.split("(")[1].split(")")[0].split(","):
                vars.add(t.strip())


    assert isinstance(config["types"], dict), "Types should be a dict of dicts, found {}.".format(type(config["types"]))
    for k, v in config["types"].items():
        assert isinstance(v, dict) or isinstance(v, list), "Types should be a dict of dicts or a dict of lists, entry {} is {}.".format(k, type(v))
        is_enum = None
        is_int = None
        if isinstance(v, dict):
            for k2, v2 in v.items(): # If a data type is perceptual, we need to check whether its values are coherent and whether images exist.
                if is_enum is None:
                    is_enum = isinstance(k2, str)
                if is_int is None:
                    is_int = isinstance(k2, int)

                is_enum &= isinstance(k2, str)
                is_int &= isinstance(k2, int)

                for k3, v3 in config["splits"].items():
                    assert os.path.exists("{}/{}".format(v3["path"], v2)), \
                        "Dataset for (domain {}, value {}) not found at {}/{}.".format(k, k2, v3["path"], v2)
        else:
            for k2 in v: # If a data type is symbolic, we need to check whether its values are coherent.
                if is_enum is None:
                    is_enum = isinstance(k2, str)
                if is_int is None:
                    is_int = isinstance(k2, int)

                is_enum &= isinstance(k2, str)
                is_int &= isinstance(k2, int)

        assert is_enum or is_int, \
            "Values for type {} must be either all integers or all strings, found {}.".format(k, v.keys())


    assert isinstance(config["domains"], dict), \
        "Domains should be specified as a dict {{variable: type}}, found {}.".format(type(config["domains"]))
    tmp = {}
    for k, v in config["domains"].items():
        assert k in vars, "Variable {} does not appear in any constraint.".format(k)

        assert isinstance(v, str) or isinstance(v, dict), \
            "Domain specifications must be either type names (str) or dictionaries {{split_name: type_name}}, found {}.".format(type(v))
        if isinstance(v, str):
            assert v in config["types"].keys(), "Unknown type {}.".format(v)
            tmp[k] = {k2: v for k2 in config["splits"].keys()}
        else:
            if "default" in v.keys():
                assert v["default"] in config["types"].keys(), "Unknown type {}.".format(v["default"])
                tmp[k] = {k2: v["default"] for k2 in config["splits"].keys()}
            else:
                tmp[k] = {}
            for k2, v2 in v.items():
                if k2 != "default":
                    assert v2 in config["types"].keys(), "Unknown type {}.".format(v2)
                    tmp[k][k2] = v2

            dom_not_in_splits = set(tmp[k].keys()).difference(config["splits"].keys())
            splits_not_in_dom = set(config["splits"].keys()).difference(tmp[k].keys())
            assert len(dom_not_in_splits) == 0, \
                "Found unknown splits in domain specifications for variable {}: {}.".format(k, dom_not_in_splits)
            assert len(splits_not_in_dom) == 0, \
                "Found undefined domain specification for variable {} in splits: {}. Either add a 'default' field, or specify every split.".format(k, splits_not_in_dom)

    config["domains"] = tmp

    for v in vars:
        assert v in config["domains"].keys(), "Variable {} is not associated with any type.".format(v)

    assert isinstance(config["streams"], dict), \
        "Expected a dictionary of stream mappings, found {}.".format(type, config["streams"])
    assert len(set(config["streams"].keys()).symmetric_difference(config["domains"].keys())) == 0, \
        "Every variable must appear both on domains specifications and stream specifications. Found variables {} not appearing in both.".format(set(config["streams"].keys()).symmetric_difference(config["domains"].keys()))

    for k, v in config["streams"].items():
        assert isinstance(v, int) or isinstance(v, str), \
            "Expected stream for variable {} to be of type int or str, found {}.".format(k, type(v))

    for k, v in config["splits"].items():
        assert "path" in v.keys() and "samples" in v.keys(), \
            "Invalid specification for split {}. It must contain the following keys: 'path', 'samples', found: {}.".format(k, v)
        assert isinstance(v["samples"], int) and v["samples"] > 0, \
            "The number of samples for split {} must be > 0, found {}.".format(k, v["samples"])
        assert os.path.exists(v["path"]), "Data path for split {} not found at {}.".format(k, v["path"])

    assert {"accepting_sinks", "rejecting_sinks", "self_loops"} == set(config["avoid_states"].keys()), \
        "Avoidance policies must be defined for accepting sinks, rejecting sinks and self-loop states, found: {}.".format(config["avoid_states"].keys())
    tmp = {}
    for k, v in config["avoid_states"].items():
        assert isinstance(v, bool) or isinstance(v, dict), \
            "Avoidance policy for {} states must be either bool or dict, found {}.".format(type(k, v))
        if isinstance(v, dict):
            key = list(v.keys())[0]
            assert len(v.keys()) == 1 and key in ["linear", "exponential"], \
                "Avoidance policy for {} states must be a single mapping {{'linear': rate}} or {{'exponential': rate}}, found {}.".format(k, v.keys())
            val = float(v[key])
            assert val >= 0 and val <= 1, "Avoidance rate must be 0 <= rate <= 1, found {}.".format(val)
            tmp[k] = (key, val)
        else:
            tmp[k] = ("linear", 0.0 if v else 1.0)

    config["avoid_states"] = tmp

    assert config["guarantee_constraint_sampling"] in [False, True, "positive", "negative", "both"], \
        "Constraint sampling guarantees must be either bool or one of ['positive', 'negative', 'both'], found {}.".format(config["guarantee_constraint_sampling"])

    assert config["guarantee_mode"] in ["strict_f", "strict_g", "best_f", "best_g"], \
        "Constraint guarantee mode must be one of ['strict_f', 'strict_g', 'best_f', 'best_g'], found {}.".format(config["guarantee_mode"])

    assert (config["incremental_guarantee_ratio"] == "once" or isinstance(config["incremental_guarantee_ratio"], float) and
            config["incremental_guarantee_ratio"] > 0.0 and config["incremental_guarantee_ratio"] <= 1.0), \
        "Guarantee ratio for incremental mode must be a non-null probability (0 < p <= 1.0) or 'once', found {}.".format(config["incremental_guarantee_ratio"])

    if config["guarantee_constraint_sampling"] == "both":
        config["guarantee_constraint_sampling"] = True

    return config