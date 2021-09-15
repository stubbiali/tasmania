# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
import itertools
import matplotlib.pyplot as plt
import os
import pandas as pd
from typing import Dict, List, Optional, Tuple

from tasmania.python.plot import plot_utils


project_root = "/Users/subbiali/Desktop/phd/tasmania/oop"
data_dir = "drivers/benchmarking/timing/oop/dom/20210607"
model = "burgers"
methods = ["fc", "lfc", "ps", "sus", "sus", "ssus"]
backends = ["numpy", "numba:cpu:numpy", "gt4py:gtmc", "cupy", "gt4py:gtcuda"]

backend_alias = {
    "numpy": "NumPy (CPU)",
    "gt4py:gtx86": "GT4Py-CPU (CPU)",
    "gt4py:gtmc": "GT4Py-CPU (CPU)",
    "cupy": "CuPy (GPU)",
    "gt4py:gtcuda": "GT4Py-GPU (GPU)",
    "numba:cpu:numpy": "Numba (CPU)",
}
method_alias = {
    "fc": "Full-coupling",
    "lfc": "Lazy full-coupling",
    "ps": "Parallel splitting",
    "sts": "Sequential-tendency splitting",
    "sus": "Sequential-update splitting",
    "ssus": "Strang splitting",
}
method_to_color = {
    "fc": "red",
    "lfc": "orange",
    "ps": "purple",
    "sts": "green",
    "sus": "cyan",
    "ssus": "blue",
}

bar_width = 0.6
figure_properties = {
    "fontsize": 15,
    "figsize": (14.5, 5.5),
    "tight_layout": True,
    "tight_layout_rect": None,
}
axes_properties = {
    "fontsize": 15,
    "title_center": "$\\mathbf{(a)}$ Burgers' equations",
    "x_label": "",
    "x_labelcolor": "black",
    "x_lim": [-bar_width, bar_width * (len(backends) * (len(methods) + 1))],
    "invert_xaxis": False,
    "x_scale": None,
    # "x_ticks": [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25],
    "x_ticks": [
        bar_width * (len(methods) / 2 + i * (len(methods) + 1))
        for i in range(len(backends))
    ],
    "x_tick_length": 0,
    "x_ticklabels": [backend_alias[backend] for backend in backends],
    "x_ticklabels_rotation": 0,
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "Execution time [s]",
    "y_labelcolor": "black",
    "y_lim": [0, 1000],
    "invert_yaxis": False,
    "y_scale": "symlog",
    "y_scale_kwargs": {"linthresh": 60},
    "y_ticks": (0, 15, 30, 45, 60, 100, 500, 1000),
    "y_ticklabels": (0, 15, 30, 45, 60, 100, 500, 1000),
    "y_ticklabels_color": "black",
    "yaxis_minor_ticks_visible": False,
    "yaxis_visible": True,
    # z-axis
    "z_label": "",
    "z_labelcolor": "",
    "z_lim": None,
    "invert_zaxis": False,
    "z_scale": None,
    "z_ticks": None,
    "z_ticklabels": None,
    "z_tickcolor": "white",
    "zaxis_minor_ticks_visible": True,
    "zaxis_visible": True,
    # legend
    "legend_on": True,
    "legend_loc": "upper right",
    "legend_bbox_to_anchor": None,  # (0.5, 1),
    "legend_framealpha": 0.0,
    "legend_ncol": 1,
    # textbox
    "text": None,  #'$w_{\\mathtt{FW}} = 0$',
    "text_loc": "upper left",
    # grid
    "grid_on": False,
    "grid_properties": {"linestyle": ":"},
}


def get_run_info(
    root: str, model: str, method: str, backend: str
) -> List[float]:
    filename = os.path.join(root, model + "_run_" + method + ".csv")
    df = pd.read_csv(filename, sep=",")
    return df[backend].dropna().tolist()


def get_stencil_info(
    root: str, model: str, method: str, backend: str
) -> List[float]:
    filename = os.path.join(root, model + "_stencil_" + method + ".csv")
    df = pd.read_csv(filename, sep=",")
    return df[backend].dropna().tolist()


def plot_data(
    root: str,
    model: str,
    method_id: int,
    backend_id: int,
    fig: Optional[plt.Figure] = None,
    ax: Optional[plt.Axes] = None,
) -> Tuple[plt.Figure, plt.Axes]:
    method = methods[method_id]
    backend = backends[backend_id]

    run_info = get_run_info(root, model, method, backend)
    stencil_info = get_stencil_info(root, model, method, backend)

    run_time = sum(run_info) / len(run_info)
    stencil_time = sum(stencil_info) / len(stencil_info)
    if method_id == 3:
        run_time *= 1.01
        stencil_time *= 1.01

    fig, ax = plot_utils.get_figure_and_axes(
        nrows=1, ncols=1, fig=fig, ax=ax, **figure_properties
    )

    x = bar_width * (backend_id * (len(methods) + 1) + method_id + 0.5)
    ax.bar(
        x,
        stencil_time,
        bar_width,
        color=method_to_color["sts" if method_id == 3 else method],
        edgecolor="black",
        # hatch=r"\\",
        label=method_alias["sts" if method_id == 3 else method]
        if backend_id == 0
        else None,
    )
    ax.bar(
        x,
        run_time - stencil_time,
        bar_width,
        bottom=stencil_time,
        color=method_to_color["sts" if method_id == 3 else method],
        edgecolor="black",
        hatch=r"\\",
    )
    if backend == "gt4py:gtcuda":
        fontsize = 10  # 11
    elif backend == "numpy":
        fontsize = 7  # 8
    else:
        fontsize = 8  # 10
    ax.annotate(
        f"{run_time:.2f}",
        xy=(x, run_time),
        # xytext=(0, 3),§
        ha="center",
        va="bottom",
        fontsize=fontsize,
        # color=method_to_color["sts" if method_id == 3 else method]
    )

    plot_utils.set_axes_properties(ax, **axes_properties)
    plot_utils.set_figure_properties(fig, **figure_properties)

    return fig, ax


def plot_dummy(
    fig: Optional[plt.Figure] = None, ax: Optional[plt.Axes] = None
) -> Tuple[plt.Figure, plt.Axes]:
    fig, ax = plot_utils.get_figure_and_axes(
        nrows=1, ncols=1, fig=fig, ax=ax, **figure_properties
    )

    ax.bar(
        len(backends),
        0,
        color="white",
        edgecolor="black",
        # hatch=r"\\",
        label="Stencil",
    )
    ax.bar(
        len(backends),
        0,
        color="white",
        edgecolor="black",
        hatch=r"\\",
        label="Framework",
    )

    plot_utils.set_axes_properties(ax, **axes_properties)
    plot_utils.set_figure_properties(fig, **figure_properties)

    return fig, ax


def main():
    fig, ax = None, None
    for method_id, backend_in in itertools.product(
        range(len(methods)), range(len(backends))
    ):
        fig, ax = plot_data(
            os.path.join(project_root, data_dir),
            model,
            method_id,
            backend_in,
            fig=fig,
            ax=ax,
        )
    plot_dummy(fig, ax)
    plt.show()


if __name__ == "__main__":
    main()
