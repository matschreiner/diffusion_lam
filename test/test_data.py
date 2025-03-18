import yaml

from dlam import utils
from dlam.data import WeatherDataset


def test_dataset_can_instantiate(zarr_test_path):
    WeatherDataset(path=zarr_test_path)


def test_dataset_islice(zarr_test_path):
    iselection = {"x": slice(0, 5), "y": slice(0, 5), "time": slice(0, 5)}
    dataset = WeatherDataset(path=zarr_test_path, iselection=iselection)
    assert dataset.data["u"].values.shape == (5, 9, 5, 5)


def test_dataset_slice(zarr_test_path):
    selection = {"time": slice("1990-09-01", "1990-09-03"), "altitude": [30, 100]}
    dataset = WeatherDataset(path=zarr_test_path, selection=selection)
    assert dataset.data["u"].values.shape == (24, 2, 10, 10)


def test_get_item(zarr_test_path):
    variables = ["u", "v"]
    dataset = WeatherDataset(zarr_test_path, variables=variables)
    x = dataset.__getitem__(0)
    assert x.shape == (18, 10, 10)


def test_from_config(data_config_path):
    config = utils.load_yaml(data_config_path)
    dataset = WeatherDataset(**config)
    assert dataset.data["u"].values.shape == (10, 9, 5, 5)
