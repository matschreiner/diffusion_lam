import yaml

from dlam import utils
from dlam.data import WeatherDataset


def test_dataset_can_instantiate(zarr_test_path):
    WeatherDataset(inputs=zarr_test_path)


def test_dataset_islice(test_input):
    iselection = {"x": slice(0, 5), "y": slice(0, 5), "time": slice(0, 5)}
    dataset = WeatherDataset(inputs=test_input, iselection=iselection)
    assert dataset.data["u"].values.shape == (5, 9, 5, 5)


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
    assert dataset.data["u"].values.shape == (18, 9, 5, 5)


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
