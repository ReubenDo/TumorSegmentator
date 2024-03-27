# ADAPTED FROM tumorsegmentator: https://github.com/wasserth/tumorsegmentator/tree/master/tumorsegmentator

import os
import random
import json
import string
from pathlib import Path
import pkg_resources


def get_tumorseg_dir():
    tumorseg_dir = Path(__file__).parent.resolve()
    return tumorseg_dir


def get_weights_dir():
    tumorseg_dir = get_tumorseg_dir()
    config_dir = tumorseg_dir / "weights/results"
    return config_dir


def setup_nnunet():
    # check if environment variable tumorsegmentator_config is set
    config_dir = get_tumorseg_dir()
    weights_dir = config_dir / "weights/results"

    # This variables will only be active during the python script execution. Therefore
    # we do not have to unset them in the end.
    os.environ["nnUNet_raw"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_preprocessed"] = str(weights_dir)  # not needed, just needs to be an existing directory
    os.environ["nnUNet_results"] = str(weights_dir)


def setup_tumorseg(tumorseg_id=None):
    tumorseg_dir = get_tumorseg_dir()
    tumorseg_dir.mkdir(exist_ok=True)
    tumorseg_config_file = tumorseg_dir / "config.json"

    if tumorseg_config_file.exists():
        with open(tumorseg_config_file) as f:
            config = json.load(f)
    else:
        if tumorseg_id is None:
            tumorseg_id = "tumorseg_" + ''.join(random.Random().choices(string.ascii_uppercase + string.digits, k=8))
        config = {
            "tumorseg_id": tumorseg_id,
        }
        with open(tumorseg_config_file, "w") as f:
            json.dump(config, f, indent=4)

    return config


def increase_prediction_counter():
    tumorseg_dir = get_tumorseg_dir()
    tumorseg_config_file = tumorseg_dir / "config.json"
    if tumorseg_config_file.exists():
        with open(tumorseg_config_file) as f:
            config = json.load(f)
        config["prediction_counter"] += 1
        with open(tumorseg_config_file, "w") as f:
            json.dump(config, f, indent=4)
        return config


def get_version():
    try:
        return pkg_resources.get_distribution("tumorsegmentator").version
    except pkg_resources.DistributionNotFound:
        return "unknown"


def get_config_key(key_name):
    tumorseg_dir = get_tumorseg_dir()
    tumorseg_config_file = tumorseg_dir / "config.json"
    if tumorseg_config_file.exists():
        with open(tumorseg_config_file) as f:
            config = json.load(f)
        if key_name in config:
            return config[key_name]
    return None


def set_config_key(key_name, value):
    tumorseg_dir = get_tumorseg_dir()
    tumorseg_config_file = tumorseg_dir / "config.json"
    if tumorseg_config_file.exists():
        with open(tumorseg_config_file) as f:
            config = json.load(f)
        config[key_name] = value
        with open(tumorseg_config_file, "w") as f:
            json.dump(config, f, indent=4)
        return config
    else:
        print("WARNING: Could not set config key, because config file not found.")

