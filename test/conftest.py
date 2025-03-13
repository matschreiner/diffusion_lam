import pytest
import torch


@pytest.fixture
def model():
    return torch.nn.Linear(1, 1)


@pytest.fixture
def optimizer(model):
    return torch.optim.Adam(model.parameters())
