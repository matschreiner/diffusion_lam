from types import MethodType

import pytorch_lightning as pl
import torch

import dlam
from dlam import utils


class Trainer(pl.Trainer):
    def __init__(self, scheduler_config=None, optimizer_config=None, *args, **kwargs):
        self.scheduler_config = scheduler_config or {}
        self.optimizer_config = optimizer_config or {}

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
        def configure_optimizers(module):
            if not self.optimizer_config:
                self.optimizer_config = DEFAULT_OPTIMIZER_CONFIG

            params = module.parameters()
            opt = utils.get_component(self.optimizer_config, params=params)

            if self.scheduler_config:
                sched = utils.get_component(self.scheduler_config, optimizer=opt)
                return [opt], [sched]

            return opt

        return configure_optimizers


DEFAULT_OPTIMIZER_CONFIG = {
    "name": "Adam",
    "kwargs": {"lr": 0.001},
}
