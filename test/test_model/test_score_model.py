from diffusion_lam.model.score_model import NaiveModel


def test_model(batch):
    model = NaiveModel()
    out = model(batch)
    __import__("pdb").set_trace()  # TODO delme
