# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, TYPE_CHECKING, Tuple

from tasmania.python.plot.drawer import Drawer
from tasmania.python.plot.plot_utils import make_lineplot
from tasmania.python.plot.retrievers import DataRetriever
from tasmania.python.plot.utils import to_units
from tasmania.python.utils import typingx

if TYPE_CHECKING:
    from tasmania.python.domain.grid import Grid


class LineProfile(Drawer):
    """
    Drawer which plots the profile of a given quantity along a line
    perpendicular to a coordinate plane.
    If the line is horizontal (respectively, vertical), the spatial
    coordinate is on the plot x-axis (resp., y-axis) and the quantity
    is on the plot y-axis (resp., x-axis).
    """

    def __init__(
        self,
        grid: "Grid",
        field_name: str,
        field_units: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        z: Optional[int] = None,
        axis_name: Optional[str] = None,
        axis_units: Optional[str] = None,
        axis_x: Optional[int] = None,
        axis_y: Optional[int] = None,
        axis_z: Optional[int] = None,
        properties: Optional[typingx.options_dict_t] = None,
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.Grid
            The underlying grid.
        field_name : str
            The state quantity to visualize.
        field_units : str
            The units for the quantity to visualize.
        x : `int`, optional
            Index along the first dimension of the field array identifying the
            line to visualize. Not to be specified if both `y` and `z` are given.
        y : `int`, optional
            Index along the first dimension of the field array identifying the
            line to visualize. Not to be specified if both `x` and `z` are given.
        z : `int`, optional
            Index along the first dimension of the field array identifying the
            line to visualize. Not to be specified if both `x` and `y` are given.
        axis_name : `str`, optional
            The name of the spatial axis. Options are:

                * 'x' (default and only effective if `x` is not given);
                * 'y' (default and only effective if `y` is not given);
                * 'z' (default and only effective if `z` is not given);
                * 'height' (only effective if `z` is not given);
                * 'height_on_interface_levels' (only effective if `z` is not given);
                * 'air_pressure' (only effective if `z` is not given).
                * 'air_pressure_on_interface_levels' (only effective if `z` is not given).

        axis_units : `str`, optional
            Units for the spatial axis. If not specified, the native units of
            the axis are used.
        axis_x : `int`, optional
            Index along the first dimension of the axis array identifying the line
            to visualize. Defaults to `x`. Only effective if `axis_name` is 'height' or
            'air_pressure'. Not to be specified if both `axis_y` and `axis_z` are given.
        axis_y : `int`, optional
            Index along the first dimension of the axis array identifying the line
            to visualize. Defaults to `y`. Only effective if :`axis_name` is 'height' or
            'air_pressure'. Not to be specified if both `axis_y` and `axis_z` are given.
        axis_z : `int`, optional
            Index along the first dimension of the axis array identifying the line
            to visualize. Defaults to `z`. Only effective if :`axis_name` is 'height' or
            'air_pressure'. Not to be specified if both `axis_y` and `axis_z` are given.
        properties : `dict`, optional
            Dictionary whose keys are strings denoting plot-specific settings
            and whose values specify values for those settings.
            See :func:`tasmania.python.plot.plot_utils.make_lineplot`.
        """
        super().__init__(properties)

        flag_x = 0 if x is None else 1
        flag_y = 0 if y is None else 1
        flag_z = 0 if z is None else 1
        if flag_x + flag_y + flag_z != 2:
            raise ValueError(
                "A line is uniquely identified by two indices, but here "
                "x is{}given, y is{}given and z is{}given.".format(
                    " " if flag_x else " not ",
                    " " if flag_y else " not ",
                    " " if flag_z else " not ",
                )
            )

        slice_x = slice(x, x + 1 if x != -1 else None, None) if flag_x else None
        slice_y = slice(y, y + 1 if y != -1 else None, None) if flag_y else None
        slice_z = slice(z, z + 1 if z != -1 else None, None) if flag_z else None

        retriever = DataRetriever(grid, field_name, field_units, slice_x, slice_y, slice_z)

        if not flag_x:
            self._slave = lambda state, ax: make_xplot(
                grid, axis_units, retriever, state, ax, **self.properties
            )
        elif not flag_y:
            self._slave = lambda state, ax: make_yplot(
                grid, axis_units, retriever, state, ax, **self.properties
            )
        else:
            aname = "z" if axis_name is None else axis_name
            if aname != "z":
                ax = axis_x if axis_x is not None else x
                ay = axis_y if axis_y is not None else y
                aslice_x = slice(ax, ax + 1 if ax != -1 else None, None)
                aslice_y = slice(ay, ay + 1 if ay != -1 else None, None)
                axis_retriever = DataRetriever(grid, aname, axis_units, aslice_x, aslice_y)
                self._slave = lambda state, ax: make_hplot(
                    axis_retriever, retriever, state, ax, **self.properties
                )
            else:
                self._slave = lambda state, ax: make_zplot(
                    grid, axis_units, retriever, state, ax, **self.properties
                )

    def __call__(
        self,
        state: typingx.DataArrayDict,
        fig: Optional[plt.Figure] = None,
        ax: Optional[plt.Axes] = None,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Call operator generating the plot.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The model state from which retrieving the data used to draw the plot.
        fig : matplotlib.figure.Figure
            The figure encapsulating the plot.
        ax : matplotlib.axes.Axes
            The axes encapsulating the plot

        Returns
        -------
        x : numpy.ndarray
            1-D array gathering the x-coordinates of the plotted points.
        y : numpy.ndarray
            1-D array gathering the y-coordinates of the plotted points.
        """
        return self._slave(state, ax)


def make_xplot(
    grid: "Grid",
    axis_units: str,
    field_retriever: DataRetriever,
    state: typingx.DataArrayDict,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.squeeze(field_retriever(state))
    x = (
        to_units(grid.x, axis_units).values
        if y.shape[0] == grid.nx
        else to_units(grid.x_at_u_locations, axis_units).values
    )

    if ax is not None:
        make_lineplot(x, y, ax, **kwargs)

    return x, y


def make_yplot(
    grid: "Grid",
    axis_units: str,
    field_retriever: DataRetriever,
    state: typingx.DataArrayDict,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    y = np.squeeze(field_retriever(state))
    x = (
        to_units(grid.y, axis_units).values
        if y.shape[0] == grid.ny
        else to_units(grid.y_at_v_locations, axis_units).values
    )

    if ax is not None:
        make_lineplot(x, y, ax, **kwargs)

    return x, y


def make_zplot(
    grid: "Grid",
    axis_units: str,
    field_retriever: DataRetriever,
    state: typingx.DataArrayDict,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    x = np.squeeze(field_retriever(state))
    y = (
        to_units(grid.z, axis_units).values
        if x.shape[0] == grid.nz
        else to_units(grid.z_on_interface_levels, axis_units).values
    )

    if ax is not None:
        make_lineplot(x, y, ax, **kwargs)

    return x, y


def make_hplot(
    axis_retriever: DataRetriever,
    field_retriever: DataRetriever,
    state: typingx.DataArrayDict,
    ax: Optional[plt.Axes] = None,
    **kwargs,
) -> Tuple[np.ndarray, np.ndarray]:
    xv = np.squeeze(field_retriever(state))
    yv = np.squeeze(axis_retriever(state))

    x = 0.5 * (xv[:-1] + xv[1:]) if xv.shape[0] > yv.shape[0] else xv
    y = 0.5 * (yv[:-1] + yv[1:]) if xv.shape[0] < yv.shape[0] else yv

    if ax is not None:
        make_lineplot(x, y, ax, **kwargs)

    return x, y
