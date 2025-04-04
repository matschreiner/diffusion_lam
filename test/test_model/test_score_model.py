from dlam.model.noise_model import NaiveModel


def test_model(batch):
    model = NaiveModel()
    out = model(batch)
