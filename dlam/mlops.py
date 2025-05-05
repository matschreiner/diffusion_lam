from datetime import timedelta

import pytorch_lightning as pl


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
