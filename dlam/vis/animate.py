import matplotlib.animation as animation
import matplotlib.pyplot as plt


def animate(data, fn, fig=None, ax=None, **kwargs):

    if ax is None or fig is None:
        fig, ax = plt.subplots()

    def update_frame(i):
        ax.clear()
        ax.set_xlabel("frame: {}".format(i))
        fn(ax, data[i])

    ani = animation.FuncAnimation(fig, update_frame, frames=range(len(data)), **kwargs)

    return ani
