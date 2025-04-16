import os
from datetime import timedelta
from pprint import pprint

import pytorch_lightning as pl
from torch.utils.data import DataLoader

import dlam
from dlam import utils
from dlam.trainer import Trainer


def main(config):
    pprint(config)
    dataset = utils.get_component(config.dataset)
    dataloader = DataLoader(dataset, **config.dataloader.get("kwargs", {}))

    ckpt_pth = f"{config.path}/checkpoints"
    import shutil

    shutil.rmtree(ckpt_pth, ignore_errors=True)
    trainer = Trainer(
        config.trainer.get("scheduler_config", {}),
        config.trainer.get("optimizer_config", {}),
        **config.trainer.get("kwargs", {}),
        accelerator=dlam.DEVICE,
        devices=1,
        callbacks=get_checkpoint_callbacks(dirpath=ckpt_pth),
        log_every_n_steps=1,
    )

    if "noise_based_model" in config:
        noise_model = utils.get_component(config.noise_model, domain=dataset.xy)
        model = utils.get_component(config.noise_based_model, noise_model=noise_model)

    elif "model" in config:
        model = utils.get_component(config.model, domain=dataset.xy)

    model.to(dlam.DEVICE)
    trainer.fit(model, dataloader)
    os.makedirs("results", exist_ok=True)


def get_checkpoint_callbacks(dirpath):
    loss_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="best",
        monitor="loss",
        mode="min",
        save_last=True,
    )

    timedelta_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        dirpath=dirpath,
        filename="timedelta",
        train_time_interval=timedelta(seconds=5),
    )

    return [
        loss_checkpoint_callback,
        timedelta_checkpoint_callback,
    ]


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.train_config)
    config.path = os.path.dirname(args.train_config)

    main(config)
