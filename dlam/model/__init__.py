import importlib


def get_model_from_config(model_config):
    model_module = importlib.import_module(model_config.module)
    model = getattr(model_module, model_config.name)(**model_config.kwargs)
    return model
