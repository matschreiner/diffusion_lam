import matplotlib.animation as animation
import matplotlib.pyplot as plt


def animate(data, fn, fig, ax, interval=10, repeat=False):
    def update_frame(i):
        ax.clear()
        ax.set_xlabel("frame: {}".format(i))
        fn(ax, data[i])

    ani = animation.FuncAnimation(
        fig,
        update_frame,
        frames=range(len(data)),
        interval=interval,
        repeat=repeat,
    )

    return ani
