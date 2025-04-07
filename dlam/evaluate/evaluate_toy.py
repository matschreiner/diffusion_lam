import matplotlib.pyplot as plt
import torch

from dlam.vis import animate


def evaluate_model(model, dataset):
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=100_000)))
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
