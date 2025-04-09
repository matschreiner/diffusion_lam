import matplotlib.animation as animation
import matplotlib.pyplot as plt
import torch

from dlam import utils
from dlam.data.weather_dataset import WeatherDataset
from dlam.vis import animate


def main():
    dataset = WeatherDataset("storage/mini.zarr/")
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    batch = next(iter(dataloader))

    #  model = utils.load("model800000.pkl")
    #  n_samples = 100
    #  samples = ancestral_sampling(model, batch, n_samples)
    #  utils.save(samples, "samples.pkl")

    samples = utils.load("samples.pkl")
    ground_truth = [dataset[i].cond.squeeze() for i in range(n_samples)]

    feature = 10
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    def plot_fn(ax, data):
        ax[0].imshow(data[0][feature].cpu().numpy(), cmap="viridis")
        ax[1].imshow(data[1][feature].cpu().numpy(), cmap="viridis")
        ax[0].set_title("t2m")

    ani = animate.animate(list(zip(samples, ground_truth)), plot_fn, fig=fig, ax=axs)
    plt.show()
    #  plt.show()

    #  plot(frame)

    #  ani = animate(data, interval=1000, repeat_delay=1000)


def ancestral_sampling(model, batch, n_samples):
    samples = [batch.cond.squeeze()]

    for i in range(n_samples):
        sample, intermediate = model.sample(batch, steps=20)
        batch.cond = sample

        plt.show()

        samples.append(sample.squeeze())

    return samples


#  def animate(data, **kwargs):
#      fig, axs = plt.subplots(1, 2, figsize=(10, 5))
#      feature = 3
#
#      def update_frame(i):
#          for ax in axs:
#              ax.clear()
#          axs[0].imshow(data[i][0][feature].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
#          axs[1].imshow(data[i][1][feature].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
#          axs[0].set_title(f"Frame: {i}")
#
#      ani = animation.FuncAnimation(fig, update_frame, frames=range(len(data)), **kwargs)
#      return ani


def plot(data, ax=None):
    if ax is None:
        _, ax = plt.subplots()

    ax.imshow(data.numpy(), cmap="viridis")
    ax.set_title("t2m")


if __name__ == "__main__":
    main()
