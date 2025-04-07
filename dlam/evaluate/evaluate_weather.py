import matplotlib.pyplot as plt
import torch

from dlam.vis import animate


def ancestral_sampling(model, dataset):
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    sample, _ = model.sample(batch, steps=100)

    plt.imshow(sample, cmap="viridis")
    plt.show()


def evaluate_model(model, dataset):
    batch = next(iter(torch.utils.data.DataLoader(dataset, batch_size=1)))
    sample, intermediate = model.sample(batch, steps=50)
    #  sample.squeeze_().numpy()
    data = sample.squeeze_()[1]

    plt.imshow(data, cmap="viridis")
    plt.show()
