import pytest
import torch

from dlam import utils


@pytest.fixture
def model():
    return torch.nn.Linear(1, 1)


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters())


@pytest.fixture
def batch():
    return utils.load("test/resources/batch.pkl")


@pytest.fixture
def test_input():
    return {"path": "test/resources/example_height.zarr", "feature": "altitude"}


@pytest.fixture
def data_config():
    return utils.load_yaml("test/resources/data_config.yaml")
