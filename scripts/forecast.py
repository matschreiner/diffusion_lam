from pprint import pprint

import matplotlib.pyplot as plt

from dlam import utils
from dlam.model.graph_lam import GraphLAM
from dlam.vis.animate import animate


def main(config):
    model = GraphLAM.load_from_checkpoint(config.model.kwargs.checkpoint)
    model.eval()
    dataset = utils.get_component(config.dataset)
    dataloader = utils.get_component(config.dataloader, dataset=dataset)

    prev = None
    prevprev = None
    forecasts = []
    gts = []
    for batch in dataloader:
        if prev is not None:
            batch["cond"][:, 1] = prev
        if prevprev is not None:
            batch["cond"][:, 0] = prevprev

        out = model(batch)
        forecasts.append(out.reshape(batch["xy"].shape)[0][..., 0].detach().numpy())
        gt = (
            batch["target"][:, 0].reshape(batch["xy"].shape)[0][..., 0].detach().numpy()
        )
        gts.append(gt)

        if prev is None:
            prevprev = out
        prev = out

    fig, ax = plt.subplots(1, 2)
    ani = animate(
        data=list(zip(gts, forecasts)),
        fn=plot_fn,
        fig=fig,
        ax=ax,
        repeat=False,
        interval=1000,
    )

    plt.show()


def plot_fn(ax, data):
    for data, ax in zip(data, ax):
        ax.imshow(data)
        ax.axis("off")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("forecast_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.forecast_config)
    main(config)
