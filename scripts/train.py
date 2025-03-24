import importlib

import matplotlib.pyplot as plt
import torch

from dlam import utils
from dlam.data import LaggedWeatherDataset
from dlam.model import get_model_from_config
from dlam.model.ddpm import DDPM
from dlam.trainer import Trainer
from dlam.utils import test_utils
from dlam.vis import vis2d

torch.manual_seed(0)


def main(config):
    trainer = Trainer(
        config.trainer.get("scheduler_config", {}),
        config.trainer.get("optimizer_config", {}),
        **config.trainer.get("kwargs", {})
    )

    data_config = utils.load_yaml(config.data.config_path)
    dataset = LaggedWeatherDataset(**data_config)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1, shuffle=True)

    model = get_model_from_config(config.model)

    batch = next(iter(dataloader))

    dataloader = test_utils.get_infinite_dataloader(batch)

    trainer.fit(model, dataloader)

    #  plts = 3
    #  pos = batch.cond.pos[0]
    #  _, ax = plt.subplots(plts, 2)
    #  for frame_idx in range(30, 30 + plts):
    #      out = model.last[0][0].T[frame_idx]
    #      gt = batch.target.state[0].T[frame_idx]
    #
    #      gtax = ax[frame_idx % 30][0]
    #      mdax = ax[frame_idx % 30][1]
    #      vis2d(pos, gt, ax=gtax)
    #      gtax.set_title("Ground Truth")
    #
    #      vis2d(pos, out, ax=mdax)
    #      mdax.set_title("Trained")

    plt.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.train_config)

    main(config)
