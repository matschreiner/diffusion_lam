import matplotlib.pyplot as plt
import torch
from sklearn.datasets import make_moons

from dlam import utils
from dlam.vis import animate


def main():
    model = utils.load("results/mine/model.pkl")
    example = torch.randn(100000, 2)

    ode_steps = 100
    out, inter = model.sample(example, ode_steps)
    fig, ax = plt.subplots()

    animation = animate.animate(
        inter,
        fn=lambda ax, data: ax.scatter(*data.T.numpy(), alpha=0.1, s=1),
        fig=fig,
        ax=ax,
        interval=5000 / ode_steps,
        repeat=False,
    )
    plt.show()

    plt.scatter(*out.T.numpy(), alpha=0.1, s=1)
    plt.show()


if __name__ == "__main__":
    main()
