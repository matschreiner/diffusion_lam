import matplotlib.pyplot as plt
import torch

from dlam import utils
from dlam.data.weather_dataset import WeatherDataset
from dlam.vis.animate import animate


def main(config):
    example_forecast = utils.load(config.example_forecast)
    model = utils.load(config.model)

    batch = utils.AttrDict(
        {"cond": example_forecast[0].repeat(config.ensemble_members, 1, 1, 1)}
    )
    samples = ancestral_sample(model, batch, config.forecasting_steps)
    utils.save(samples, "samples.pkl")
    samples = utils.load("samples.pkl")

    data = list(zip(samples, example_forecast))
    fig, axs = plt.subplots(1, 3, figsize=(15, 5))

    _ = animate(data, plot_fn, fig, axs)
    plt.show()

    pass


def ancestral_sample(model, batch, forecasting_steps):
    samples = [batch.cond]

    for _ in range(forecasting_steps):
        sample, _ = model.sample(batch, steps=20)
        batch.cond = sample
        samples.append(sample)

    return samples


def plot_fn(ax, data):
    feature = 0
    sample = data[0]
    ground_truth = data[1]
    uncertainty = sample.std(axis=0)

    ax[0].imshow(sample[0][feature].cpu().numpy(), cmap="viridis", vmin=-2, vmax=2)
    ax[1].imshow(ground_truth[feature].cpu().numpy(), cmap="viridis", vmin=-2, vmax=2)
    ax[2].imshow(uncertainty[feature].cpu().numpy(), cmap="viridis", vmin=-2, vmax=2)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("eval_config", type=str)
    args = parser.parse_args()

    config = utils.load_yaml(args.eval_config)

    main(config)
