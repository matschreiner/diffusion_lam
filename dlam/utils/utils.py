import importlib
import os
import pickle as pkl

import yaml

from dlam.utils.attrdict import AttrDict


def load_yaml(path):
    with open(path, "r") as file:
        d = yaml.safe_load(file)
        return AttrDict(d)


def save(obj, name):
    name = "./" + name if not name.startswith("/") else name
    os.makedirs(os.path.dirname(name), exist_ok=True)
    with open(f"{name}", "wb") as f:
        pkl.dump(obj, f)


def load(name):
    with open(f"{name}", "rb") as f:
        return pkl.load(f)


def load_yaml(path):
    with open(path, "r") as file:
        return AttrDict(yaml.safe_load(file))
