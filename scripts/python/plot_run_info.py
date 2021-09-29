# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
from typing import Dict, List

from tasmania.python.plot import plot_utils
from tasmania.python.utils.backend import is_gt


backend_alias = {
    "numpy": "NumPy (CPU)",
    # "gt4py:gtx86": "GT4Py CPU",
    "gt4py:gtmc": "GT4Py:gtmc (CPU)",
    "cupy": "CuPy (GPU)",
    "gt4py:gtcuda": "GT4Py:gtcuda (GPU)",
}

figure_properties = {
    "fontsize": 15,
    "figsize": (9.5, 5),
    "tight_layout": True,
    "tight_layout_rect": None,
}

axes_properties = {
    "fontsize": 15,
    "x_label": "",
    "x_labelcolor": "black",
    "x_lim": [-0.5, 6.5],
    "invert_xaxis": False,
    "x_scale": None,
    # "x_ticks": [0.25, 1.25, 2.25, 3.25, 4.25, 5.25, 6.25],
    "x_ticks": [0, 2, 4, 6],
    "x_ticklabels": list(backend_alias.values()),
    "x_ticklabels_rotation": 0,
    "x_tickcolor": "black",
    "xaxis_minor_ticks_visible": False,
    "xaxis_visible": True,
    # y-axis
    "y_label": "Run time [s]",
    "y_labelcolor": "black",
    "y_lim": [0, 700],
    "invert_yaxis": False,
    "y_scale": None,
    "y_ticks": range(0, 601, 100),
    "y_ticklabels": None,
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
    "legend_on": False,
    "legend_loc": "upper center",
    "legend_bbox_to_anchor": None,  # (0.5, 1),
    "legend_framealpha": 1.0,
    "legend_ncol": 3,
    # textbox
    "text": None,  #'$w_{\\mathtt{FW}} = 0$',
    "text_loc": "upper left",
    # grid
    "grid_on": False,
    "grid_properties": {"linestyle": ":"},
}


def get_run_info(root: str, model: str, method: str) -> Dict[str, List[float]]:
    filename = os.path.join(root, model + "_run_" + method + ".csv")
    df = pd.read_csv(filename, sep=",")
    data = {backend: df[backend].dropna().tolist() for backend in df.columns}
    return data


def get_cpp_time(root: str, model: str, method: str, backend: str) -> float:
    filename = os.path.join(
        root, model + "_exec_" + method + "_" + backend + ".csv"
    )
    df = pd.read_csv(filename, sep=",")
    return df["total_run_cpp_time"].sum()


def get_gt4py_run_time(
    root: str, model: str, method: str, backend: str
) -> float:
    filename = os.path.join(
        root, model + "_exec_" + method + "_" + backend + ".csv"
    )
    df = pd.read_csv(filename, sep=",")
    return df["total_run_time"].sum()


def get_call_time(root: str, model: str, method: str, backend: str) -> float:
    filename = os.path.join(
        root, model + "_exec_" + method + "_" + backend + ".csv"
    )
    df = pd.read_csv(filename, sep=",")
    return df["total_call_time"].sum()


def plot_data(root: str, model: str, method: str):
    run_info = get_run_info(root, model, method)

    backends = backend_alias.keys()  # run_info.keys()
    run_time = [
        sum(run_info[backend]) / len(run_info[backend]) for backend in backends
    ]
    # run_time = [
    #     run_info[backend][-1] for backend in backends
    # ]
    cpp_time = [
        get_gt4py_run_time(root, model, method, backend)
        if is_gt(backend)
        else 0.0
        for backend in backends
    ]
    python_time = [run - cpp for run, cpp in zip(run_time, cpp_time)]
    taz_time = [
        run - get_call_time(root, model, method, backend)
        if is_gt(backend)
        else run
        for run, backend in zip(run_time, backends)
    ]
    gt4py_time = [python - taz for python, taz in zip(python_time, taz_time)]
    gt4py_call_overhead = [
        run - taz - cpp for run, taz, cpp in zip(run_time, taz_time, cpp_time)
    ]

    fig, ax = plot_utils.get_figure_and_axes(
        nrows=1, ncols=1, **figure_properties
    )

    width = 0.6
    # colors = ["red", "orange", "blue", "green"]
    colors = ["white"] * 4

    x = range(len(backends))
    for i in x:
        ax.bar(
            i,
            cpp_time[i],
            width,
            color=colors[i],
            edgecolor="black",
            # hatch=r"\\",
        )
        ax.bar(
            i,
            gt4py_time[i],
            width,
            bottom=cpp_time[i],
            color=colors[i],
            edgecolor="black",
            hatch=r"\\",
        )
        ax.bar(
            i,
            taz_time[i],
            width,
            bottom=cpp_time[i] + gt4py_time[i],
            color=colors[i],
            edgecolor="black",
            hatch=r"oo",
        )
        ax.annotate(
            f"{run_time[i]:.3f}",
            xy=(i, run_time[i]),
            # xytext=(0, 3),§
            ha="center",
            va="bottom",
        )

    ax.bar(
        len(backends),
        0,
        color="white",
        edgecolor="black",
        # hatch=r"\\",
        label="C++",
    )
    ax.bar(
        len(backends),
        0,
        color="white",
        edgecolor="black",
        hatch=r"\\",
        label="GT4Py bindings",
    )
    ax.bar(
        len(backends),
        0,
        color="white",
        edgecolor="black",
        hatch="oo",
        label="Tasmania",
    )

    plot_utils.set_axes_properties(ax, **axes_properties)
    plot_utils.set_figure_properties(fig, **figure_properties)

    plt.show()


def plot_data_strip(root: str, model: str, method: str):
    run_info = get_run_info(root, model, method)

    backends = backend_alias.keys()  # run_info.keys()
    run_time = [
        sum(run_info[backend]) / len(run_info[backend]) for backend in backends
    ]

    fig, ax = plot_utils.get_figure_and_axes(
        nrows=1, ncols=1, **figure_properties
    )

    width = 0.6
    # colors = ["red", "orange", "blue", "green"]
    colors = ["grey"] * 4

    x = axes_properties["x_ticks"]
    for i in range(len(x)):
        ax.bar(
            x[i],
            run_time[i],
            width,
            color=colors[i],
            edgecolor="black",
            # hatch=r"\\",
        )
        ax.annotate(
            f"{run_time[i]:.3f}",
            xy=(x[i], run_time[i]),
            # xytext=(0, 3),§
            ha="center",
            va="bottom",
        )

    plot_utils.set_axes_properties(ax, **axes_properties)
    plot_utils.set_figure_properties(fig, **figure_properties)

    plt.show()


if __name__ == "__main__":
    project_root = "/Users/subbiali/Desktop/phd/tasmania/oop"
    data_dir = "drivers/benchmarking/timing/oop/dom/20210607"
    plot_data_strip(
        os.path.join(project_root, data_dir), "isentropic_moist", "fc"
    )
