from unittest.mock import MagicMock, patch

import torch

import diffusion_lam


def test_trainer_instantiates():
    diffusion_lam.trainer.Trainer()


def test_configure_optimizer_factory(model):
    trainer = diffusion_lam.trainer.Trainer(scheduler_config=GRAPHCAST_SCHEDULER_CONFIG)
    configure_optimizers = trainer.get_configure_optimizers_callback()
    [optimizer], [scheduler] = configure_optimizers(model)

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, diffusion_lam.scheduler.GraphcastScheduler)


def test_model_can_configure_optimizers(model):

    trainer = diffusion_lam.trainer.Trainer(scheduler_config=GRAPHCAST_SCHEDULER_CONFIG)

    with patch.object(diffusion_lam.trainer.Trainer.__bases__[0], "fit", MagicMock()):
        trainer.fit(model)

    [optimizer], [scheduler] = model.configure_optimizers()

    assert isinstance(optimizer, torch.optim.AdamW)
    assert isinstance(scheduler, diffusion_lam.scheduler.GraphcastScheduler)


def test_can_instantiate_torch_optimizer(model):
    optimizer_config = {
        "name": "Adam",
        "kwargs": {"lr": 0.01},
    }

    optimizer = diffusion_lam.trainer.get_optimizer(optimizer_config, model)
    assert isinstance(optimizer, torch.optim.Adam)
    assert optimizer.defaults["lr"] == 0.01


def test_can_instantiate_torch_scheduler(optimizer):
    scheduler_config = {
        "name": "StepLR",
        "kwargs": {"step_size": 10, "gamma": 0.1},
    }

    scheduler = diffusion_lam.trainer.get_scheduler(optimizer, scheduler_config)
    assert isinstance(scheduler, torch.optim.lr_scheduler.StepLR)
    assert scheduler.step_size == 10
    assert scheduler.gamma == 0.1


GRAPHCAST_SCHEDULER_CONFIG = {
    "name": "GraphcastScheduler",
    "kwargs": {
        "warmup_steps": 1000,
        "annealing_steps": 100000,
        "max_factor": 1.0,
        "min_factor": 0.001,
    },
}
