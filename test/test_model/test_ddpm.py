from dlam.model.ddpm import DDPM


def test_ddpm_can_instantiate(model):
    DDPM(model)


def test_get_corrupted_has_correct_dimensions(batch):
    ddpm = DDPM(None)
    x = batch.target_states
    corrupted, epsilon, t_diff = ddpm.get_corrupted(batch.target_states)

    assert x.shape == corrupted.shape
    assert x.shape == epsilon.shape
    assert x.shape[0] == t_diff.shape[0]


def test_training_step(batch):
    score_model = lambda x, _: x
    ddpm = DDPM(score_model)
    ddpm.training_step(batch, None)
