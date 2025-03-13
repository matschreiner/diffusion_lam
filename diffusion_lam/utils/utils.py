import yaml

from diffusion_lam.utils.attrdict import AttrDict


def load_yaml_as_attrdict(path):
    with open(path, "r") as file:
        d = yaml.safe_load(file)
        return AttrDict(d)
