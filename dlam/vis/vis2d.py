import matplotlib.pyplot as plt
import numpy as np
import torch
from scipy.interpolate import griddata


def vis2d(pos, vals, ax=None):
    if isinstance(pos, torch.Tensor):
        pos = pos.detach().numpy()
    if isinstance(vals, torch.Tensor):
        vals = vals.detach().numpy()

    if ax is None:
        fig, ax = plt.subplots()

    x = pos[:, 0]
    y = pos[:, 1]
    z = vals.ravel()
    xi = np.linspace(x.min(), x.max(), len(x))
    yi = np.linspace(y.min(), y.max(), len(y))
    xi, yi = np.meshgrid(xi, yi)
    zi = griddata((x, y), z, (xi, yi), method="linear")
    ax.imshow(zi, origin="lower", extent=[x.min(), x.max(), y.min(), y.max()])
