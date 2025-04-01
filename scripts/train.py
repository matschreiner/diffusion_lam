import os

import matplotlib.pyplot as plt
import pytorch_lightning as pl
import torch
from torch.utils.data import DataLoader

from dlam import utils
from dlam.model import get_model_from_config
from dlam.model.ddpm import DDPM
from dlam.trainer import Trainer
from dlam.vis import animate


def main(config):

    trainer = Trainer(
        config.trainer.get("scheduler_config", {}),
        config.trainer.get("optimizer_config", {}),
        **config.trainer.get("kwargs", {}),
    )

    dataset = utils.get_component_from_config(config.dataset)
    dataloader = DataLoader(dataset, **config.dataloader.get("kwargs", {}))

    score_model = utils.get_component_from_config(config.model)
    #  plt.scatter(*dataset.data.T, alpha=0.1, s=1)
    #  plt.show()

    example_batch = next(iter(dataloader))

    score_model = get_model_from_config(config.model)
    ddpm = DDPM(score_model)

    trainer.fit(ddpm, dataloader)

    utils.save(ddpm, "results/mine/model.pkl")

    out, intermediates = ddpm.sample(dataset.data, **config.get("sample", {}))

    fig, ax = plt.subplots()
    #  ax.scatter(*example_batch.T.numpy())
    #  movement = torch.stack(intermediates).diff(dim=0).norm(dim=-1).mean(-1)
    #  plt.plot(movement)
    #  plt.show()

    def fn(ax, data):
        ax.scatter(*example_batch.T.numpy())
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.scatter(*data.T.numpy(), alpha=0.1, s=1)
        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ani = animate.animate(
        intermediates,
        fn=fn,
        fig=fig,
        ax=ax,
    )

    os.makedirs("results/mine", exist_ok=True)
    ani.save("results/mine/render.mp4", fps=20)
    utils.save(
        intermediates,
        "results/mine/traj.pkl",
    )
    utils.save(
        out,
        "results/mine/sample.pkl",
    )
    plt.show()

    plt.scatter(*example_batch.T.numpy())
    plt.scatter(*out.T)
    plt.xlim(ax.get_xlim())
    plt.ylim(ax.get_ylim())
    plt.show()

    #  traj = torch.stack(intermediates).squeeze().numpy()
    #  out = out.squeeze().numpy()
    #  plt.hist(out, bins=100, density=True)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.train_config)

    main(config)
