import sys
import json
import glob
import copy

if len(sys.argv) < 2:
    print(f'Usage: {sys.argv[0]} config1.json [config2.json ...]')
    sys.exit(2)

def update_dict(d, u, show_warning = True, show_not_updated=True):
    updated = []
    for k, v in u.items():
        if k not in d and show_warning:
            print("\033[91m Warning: key {} not found in config. Make sure to double check spelling and config option name. \033[0m".format(k))
        if isinstance(v, dict):
            updated.append(k)
            d[k] = update_dict(d.get(k, {}), v, show_warning)
        else:
            updated.append(k)
            d[k] = v
    if show_not_updated:
        not_updated = set(d.keys()) - set(updated)
        if not_updated:
            print('Keys not updated:', not_updated)
    return d

def get_shared_config(config1, config2, verbose=True):
    shared_config = {}
    # assert set(config1.keys()) == set(config2.keys())
    for k in config1.keys():
        if isinstance(config1[k], dict):
            if k in config2.keys():
                shared_config[k] = get_shared_config(config1[k], config2[k])
        else:
            if k in config2.keys():
                if config1[k] == config2[k]:
                    shared_config[k] = config1[k]
                else:
                    if verbose:
                        print(f'mismatch for key {k}: {config1[k]} != {config2[k]}')
            else:
                if verbose:
                    print(f'mismatch: key {k} does not exist')
    return shared_config

def get_total_keys(config):
    count = 0
    for k in config.keys():
        if isinstance(config[k], dict):
            count += get_total_keys(config[k])
        else:
            count += 1
    return count

with open("../defaults.json") as f:
    common_config = json.load(f)

configs = []

for config_file in sys.argv[1:]:
    with open(config_file, "r") as f:
        print('Loading', config_file)
        configs.append(update_dict(copy.deepcopy(common_config), json.load(f)))

reduced_config = configs[0]

for f, c in zip(sys.argv[1:], configs):
    print(f'Processing {f}')
    get_shared_config(configs[0], c)
    reduced_config = get_shared_config(reduced_config, c, verbose=False)
    print(f'{get_total_keys(reduced_config)} keys left')

print(json.dumps(reduced_config, indent=2))
