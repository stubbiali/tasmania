# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

fig, axes = plt.subplots(nrows=2, ncols=2)
for ax in axes.flat:
    im = ax.imshow(np.random.random((10, 10)), vmin=0, vmax=1)

fig.colorbar(im, orientation="vertical", ax=fig.get_axes())

plt.show()
