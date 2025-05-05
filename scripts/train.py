import os
from dataclasses import dataclass
from pprint import pprint

import dlam
from dlam import utils
from dlam.trainer import Trainer


def main(config):
    pprint(config)
    dataset = utils.get_component(config.dataset)
    dataloader = utils.get_component(config.dataloader, dataset=dataset)

    logger = utils.get_component(config.logger) if hasattr(config, "logger") else None
    trainer = Trainer(
        **config.trainer.get("kwargs", {}),
        accelerator=dlam.DEVICE,
        devices=1,
        callbacks=dlam.mlops.get_checkpoint_callbacks(
            dirpath=f"{config.path}/checkpoints"
        ),
        log_every_n_steps=1,
        logger=logger,
    )

    model = utils.get_component(config.model)

    create_graph = model.create_graph if hasattr(model, "create_graph") else None

    if hasattr(model, "create_graph"):
        dataset.add_graph(create_graph)

    model.to(dlam.DEVICE)
    trainer.fit(model, dataloader)
    os.makedirs("results", exist_ok=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.train_config)

    config.path = os.path.dirname(args.train_config)

    main(config)
