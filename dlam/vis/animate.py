from collections.abc import Iterable

import matplotlib.animation as animation
import matplotlib.pyplot as plt


def animate(data, fn, fig=None, ax=None, **kwargs):

    if ax is None:
        fig, ax = plt.subplots()

    def update_frame(i):
        if isinstance(ax, Iterable):
            for a in ax:
                a.clear()
            ax[0].set_xlabel("frame: {}".format(i))
        else:
            ax.clear()
            ax.set_xlabel("frame: {}".format(i))

        fn(ax=ax, data=data[i])

    ani = animation.FuncAnimation(fig, update_frame, frames=range(len(data)), **kwargs)

    return ani
