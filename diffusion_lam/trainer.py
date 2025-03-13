from types import MethodType

import pytorch_lightning as pl
import torch

import diffusion_lam


class Trainer(pl.Trainer):
    def __init__(self, scheduler_config=None, optimizer_config=None, *args, **kwargs):
        self.scheduler_config = scheduler_config
        self.optimizer_config = optimizer_config

        super().__init__(*args, **kwargs)

    def fit(
        self,
        model,
        *args,
        **kwargs,
    ):
        model.configure_optimizers = MethodType(
            self.get_configure_optimizers_callback(), model
        )
        super().fit(model, *args, **kwargs)

    def get_configure_optimizers_callback(self):

        def configure_optimizers(pl_module):
            if self.optimizer_config is None:
                optimizer = get_default_optimizer(pl_module)

            if self.scheduler_config is not None:
                optimizer = get_optimizer(self.optimizer_config, pl_module)
                scheduler = get_scheduler(optimizer, self.scheduler_config)
                return [optimizer], [scheduler]

            return optimizer

        return configure_optimizers


def get_optimizer(optimizer_config, pl_module):
    if optimizer_config is None:
        torch.optim.AdamW

    optimizer_name = optimizer_config["name"]
    optimizer_kwargs = optimizer_config.get("kwargs", {})
    optimizer_cls = getattr(torch.optim, optimizer_name, None)

    if optimizer_cls is None:
        raise ValueError(f"Unknown optimizer: {optimizer_config}")

    return optimizer_cls(pl_module.parameters(), **optimizer_kwargs)


def get_default_optimizer(pl_module):
    return torch.optim.AdamW(pl_module.parameters(), lr=0.001, betas=(0.9, 0.95))


def get_scheduler(optimizer, scheduler_config):
    scheduler_name = scheduler_config["name"]
    scheduler_cls = getattr(torch.optim.lr_scheduler, scheduler_name, None)
    if scheduler_cls is None:
        scheduler_cls = getattr(diffusion_lam.scheduler, scheduler_config["scheduler"])
    if scheduler_cls is None:
        raise ValueError(f"Unknown scheduler: {scheduler_name}")

    return scheduler_cls(optimizer, **scheduler_config["kwargs"])
