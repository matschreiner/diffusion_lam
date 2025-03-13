import pytest
import torch

from diffusion_lam import utils


@pytest.fixture
def model():
    return torch.nn.Linear(1, 1)


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters())


@pytest.fixture
def batch():
    return utils.load("test/resources/batch.pkl")
