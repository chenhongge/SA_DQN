import os
import json
import glob
import copy
import importlib
import torch
import numpy as np



def update_dict(d, u, show_warning = False):
    for k, v in u.items():
        if k not in d and show_warning:
            print("\033[91m Warning: key {} not found in config. Make sure to double check spelling and config option name. \033[0m".format(k))
        if isinstance(v, dict):
            d[k] = update_dict(d.get(k, {}), v, show_warning)
        else:
            d[k] = v
    return d

def load_config(args):
    print("loading config file: {}".format(args.config))
    with open("defaults.json") as f:
        config = json.load(f)
    with open(args.config) as f:
        update_dict(config, json.load(f))
    if args.overrides_dict:
        print("overriding parameters: \033[93mPlease check these parameters carefully.\033[0m")
        print("\033[93m" + str(args.overrides_dict) + "\033[0m")
        update_dict(config, args.overrides_dict, True)
    subset_models = []
    # remove ignored models

    if args.path_prefix:
        config["path_prefix"] = args.path_prefix
    return config

