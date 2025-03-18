from dlam.data import WeatherDataset


def test_dataset_can_instantiate():
    dataset = WeatherDataset(path="test/resources/example.zarr")
    __import__("pdb").set_trace()  # TODO delme
