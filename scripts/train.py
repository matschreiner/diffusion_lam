import torch
import yaml

from diffusion_lam import utils
from diffusion_lam.trainer import Trainer


def main(config):
    trainer = Trainer(
        optimizer_config=config.training.optimizer,
        scheduler_config=config.training.scheduler,
    )

    model = torch.nn.Linear(1, 1)

    [optimizer], [scheduler] = trainer.get_configure_optimizers_callback()(model)

    pass


if __name__ == "__main__":
    import argparse

    import yaml

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = utils.yaml_name_space(args.train_config)

    main(config)
