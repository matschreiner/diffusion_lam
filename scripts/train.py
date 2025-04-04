import os
from pprint import pprint

import matplotlib.pyplot as plt
from pytorch_lightning.profilers import SimpleProfiler
from torch.utils.data import DataLoader

from dlam import utils
from dlam.trainer import Trainer
from dlam.vis import animate


def main(config):
    pprint(config)

    dataset = utils.get_component(config.dataset)
    dataloader = DataLoader(dataset, batch_size=2048)

    #  profiler_dir_path = "."
    #  profiler_filename = "profiler"
    #  trainer = Trainer(
    #      config.trainer.get("scheduler_config", {}),
    #      config.trainer.get("optimizer_config", {}),
    #      **config.trainer.get("kwargs", {}),
    #      profiler=SimpleProfiler(dirpath=profiler_dir_path, filename=profiler_filename),
    #  )
    #
    #
    #  noise_model = utils.get_component(config.noise_model)
    #  model = utils.get_component(config.score_based_model, noise_model=noise_model)
    #
    #  trainer.fit(model, dataloader)
    #  utils.save(model, "/tmp/model.pkl")
    #  print_profiler(profiler_dir_path, profiler_filename)

    model = utils.load("/tmp/model.pkl")

    evaluate_model(model, dataset)


def evaluate_model(model, dataset):
    batch = next(iter(DataLoader(dataset, batch_size=100_000)))
    sample, intermediate = model.sample(batch, steps=50)
    plot(sample, batch)

    mask0 = batch.cond.squeeze() == 0
    mask1 = batch.cond.squeeze() == 1

    def fn(ax, data):
        ax.scatter(*batch.target.T.numpy())
        xlim = ax.get_xlim()
        ylim = ax.get_ylim()

        ax.scatter(*data[mask0].T.numpy(), alpha=0.1, s=1)
        ax.scatter(*data[mask1].T.numpy(), alpha=0.1, s=1)

        ax.set_xlim(xlim)
        ax.set_ylim(ylim)

    ani = animate.animate(
        intermediate,
        fn,
    )
    ani.save("anim.mp4", fps=10)


def plot(sample, data):
    _, ax = plt.subplots(1, 3, figsize=(15, 5))

    mask0 = data.cond.squeeze() == 0
    mask1 = data.cond.squeeze() == 1

    ax[0].scatter(*data.target.T.numpy(), alpha=0.5)
    ax[0].scatter(*sample[mask0].T, s=1, alpha=0.1)
    ax[0].scatter(*sample[mask1].T, s=1, alpha=0.1)
    ax[0].set_title("sample")

    ax[1].hist(data.target[:, 0].numpy(), bins=100, alpha=0.5)
    ax[1].hist(sample[:, 0].numpy(), bins=100, histtype="step")
    ax[1].set_title("x")

    ax[2].hist(data.target[:, 1].numpy(), bins=100, alpha=0.5)
    ax[2].hist(sample[:, 1].numpy(), bins=100, histtype="step")
    ax[2].set_title("y")

    plt.show()


def print_profiler(dir_path, filename):
    fp = os.path.join(dir_path, "fit-" + filename) + ".txt"
    with open(fp, "rb") as f:
        print(f)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("train_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.train_config)

    main(config)
