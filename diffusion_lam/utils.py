from types import SimpleNamespace

import yaml


def yaml_name_space(path):
    with open(path, "r") as file:
        d = yaml.safe_load(file)
        return AttrDict(d)


class AttrDict(dict):
    def __getattr__(self, key):
        if key in self:
            value = self[key]
            return AttrDict(value) if isinstance(value, dict) else value
        raise AttributeError(f"Key '{key}' not found")

    def __setattr__(self, key, value):
        self[key] = value

    def __delattr__(self, key):
        del self[key]
