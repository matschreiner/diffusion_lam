import pytest
import yaml

from dlam import utils
from dlam.data import WeatherDataset


@pytest.fixture
def data_config():
    return {
        "static": [{"path": "test/resources/example_single.zarr", "features": ["t2m"]}]
    }


def test_dataset_can_instantiate(data_config):
    WeatherDataset(data_config)


def test_dataset_islice(data_config):
    iselection = {"x": slice(0, 5), "y": slice(0, 7)}
    dataset = WeatherDataset(data_config, iselection=iselection)
    assert dataset[0].pos.shape == (35, 2)


def test_dataset_slice(test_input):
    selection = {"time": slice("1990-09-01", "1990-09-03"), "altitude": [30, 100]}
    dataset = WeatherDataset(inputs=test_input, selection=selection)
    assert dataset.data["u"].values.shape == (24, 2, 10, 10)


def test_get_item(data_config):
    dataset = WeatherDataset(**data_config)
    x = dataset.__getitem__(0)
    assert x.shape == (18, 10, 10)


def test_from_config(data_config):
    dataset = WeatherDataset(**data_config)
    batch = dataset[0]
    assert batch.state.shape == (25, 20)
    assert batch.static.shape == (25, 2)


def test_single_levels_data():
    inputs = [{"path": "test/resources/example_single.zarr"}]
    dataset = WeatherDataset(inputs=inputs)


def test_stack_features():
    inputs = [
        {"path": "test/resources/example_height.zarr", "feature": "altitude"},
        {"path": "test/resources/example_single.zarr"},
    ]
    dataset = WeatherDataset(inputs=inputs)


def test_stack_features_single():
    inputs = [
        {"path": "test/resources/example_single.zarr"},
    ]
    dataset = WeatherDataset(inputs=inputs)
