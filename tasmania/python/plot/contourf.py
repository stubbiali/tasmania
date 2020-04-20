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
from matplotlib import pyplot as plt
import numpy as np
from typing import Optional, TYPE_CHECKING

from tasmania.python.plot.drawer import Drawer
from tasmania.python.plot.plot_utils import make_contourf
from tasmania.python.plot.retrievers import DataRetriever
from tasmania.python.plot.utils import to_units
from tasmania.python.utils import taz_types

if TYPE_CHECKING:
    from tasmania.python.domain.grid import Grid


class Contourf(Drawer):
    """
    Drawer which generates a contour-filled plot of a state quantity
    at a cross-section parallel to one coordinate plane.
    """

    def __init__(
        self,
        grid: "Grid",
        field_name: str,
        field_units: str,
        x: Optional[int] = None,
        y: Optional[int] = None,
        z: Optional[int] = None,
        xaxis_name: Optional[str] = None,
        xaxis_units: Optional[str] = None,
        xaxis_y: Optional[int] = None,
        xaxis_z: Optional[int] = None,
        yaxis_name: Optional[str] = None,
        yaxis_units: Optional[str] = None,
        yaxis_x: Optional[int] = None,
        yaxis_z: Optional[int] = None,
        zaxis_name: Optional[str] = None,
        zaxis_units: Optional[str] = None,
        zaxis_x: Optional[int] = None,
        zaxis_y: Optional[int] = None,
        properties: Optional[taz_types.options_dict_t] = None,
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
            Index along the first dimension of the field array identifying
            the cross-section to visualize. To be specified only if both
            `y` and `z` are not given.
        y : `int`, optional
            Index along the second dimension of the field array identifying
            the cross-section to visualize. To be specified only if both
            `x` and `z` are not given.
        z : `int`, optional
            Index along the third dimension of the field array identifying
            the cross-section to visualize. To be specified only if both
            `x` and `y` are not given.
        xaxis_name : `str`, optional
            If either `y` or `z` is given, the name of the grid	axis to
            place on the plot x-axis. Options are:

                * 'x' (default).

        xaxis_units : `str`, optional
            If either `y` or `z` is given, units for the `xaxis_name` axis.
            If not specified, the native units of the grid axis are used.
        xaxis_y : `int`, optional
            Index along the second dimension of the `xaxis_name` grid axis
            array identifying the cross-section to visualize. Defaults to `y`.
            Only effective if `xaxis_name` is not 'x' and `y` is given.
        xaxis_z : `int`, optional
            Index along the third dimension of the `xaxis_name` grid axis
            array identifying the cross-section to visualize. Defaults to `z`.
            Only effective if `xaxis_name` is not 'x' and `z` is given.
        yaxis_name : `str`, optional
            The name of the grid axis to place either on the plot x-axis
            if `x` is given, or on the plot y-axis if `z` is given. Options are:

                * 'y' (default).

        yaxis_units : `str`, optional
            If either `x` or `z` is given, units for the `yaxis_name` grid axis.
            If not specified, the native units of the grid axis are used.
        yaxis_x : `int`, optional
            Index along the first dimension of the `yaxis_name` grid axis
            array identifying the cross-section to visualize. Defaults to `x`.
            Only effective if `yaxis_name` is not 'y' and `x` is given.
        yaxis_z : `int`, optional
            Index along the third dimension of the `yaxis_name` grid axis
            array identifying the cross-section to visualize. Defaults to `z`.
            Only effective if `yaxis_name` is not 'y' and `z` is given.
        zaxis_name : `str`, optional
            If either `x` or `y` is given, the name of the grid axis to
            place on the plot y-axis. Options are:

                * 'z' (default);
                * 'height';
                * 'height_on_interface_levels';
                * 'air_pressure';
                * 'air_pressure_on_interface_levels'.

        zaxis_units : `str`, optional
            If either `x` or `y` is given, units for the `zaxis_name` grid axis.
            If not specified, the native units of the grid axis are used.
        zaxis_x : `int`, optional
            Index along the first dimension of the `zaxis_name` grid axis
            array identifying the cross-section to visualize. Defaults to `x`.
            Only effective if `zaxis_name` is not 'z' and `x` is given.
        zaxis_y : `int`, optional
            Index along the second dimension of the `zaxis_name` grid axis
            array identifying the cross-section to visualize. Defaults to `y`.
            Only effective if `zaxis_name` is not 'z' and `y` is given.
        properties : `dict`, optional
            Dictionary whose keys are strings denoting plot-specific
            settings, and whose values specify values for those settings.
            See :func:`~tasmania.python.plot.plot_utils.make_contourf`.
        """
        super().__init__(properties)

        flag_x = 0 if x is None else 1
        flag_y = 0 if y is None else 1
        flag_z = 0 if z is None else 1
        if flag_x + flag_y + flag_z != 1:
            raise ValueError(
                "A plane is uniquely identified by one index, but here "
                "x is{}given, y is{}given and z is{}given.".format(
                    " " if flag_x else " not ",
                    " " if flag_y else " not ",
                    " " if flag_z else " not ",
                )
            )

        slice_x = slice(x, x + 1 if x != -1 else None, None) if flag_x else None
        slice_y = slice(y, y + 1 if y != -1 else None, None) if flag_y else None
        slice_z = slice(z, z + 1 if z != -1 else None, None) if flag_z else None

        retriever = DataRetriever(
            grid, field_name, field_units, slice_x, slice_y, slice_z
        )

        if flag_z:
            self._slave = lambda state, fig, ax: make_contourf_xy(
                grid,
                xaxis_units,
                yaxis_units,
                retriever,
                state,
                fig,
                ax,
                **self.properties
            )
        else:
            if zaxis_name != "z":
                zax = zaxis_x if zaxis_x is not None else x
                zay = zaxis_y if zaxis_y is not None else y
                zaslice_x = (
                    None
                    if zax is None
                    else slice(zax, zax + 1 if zax != -1 else None, None)
                )
                zaslice_y = (
                    None
                    if zay is None
                    else slice(zay, zay + 1 if zay != -1 else None, None)
                )
                zaxis_retriever = DataRetriever(
                    grid, zaxis_name, zaxis_units, zaslice_x, zaslice_y
                )

                if flag_x:
                    self._slave = lambda state, fig, ax: make_contourf_yh(
                        grid,
                        yaxis_units,
                        zaxis_retriever,
                        retriever,
                        state,
                        fig,
                        ax,
                        **self.properties
                    )
                else:
                    self._slave = lambda state, fig, ax: make_contourf_xh(
                        grid,
                        xaxis_units,
                        zaxis_retriever,
                        retriever,
                        state,
                        fig,
                        ax,
                        **self.properties
                    )
            else:
                if flag_x:
                    self._slave = lambda state, fig, ax: make_contourf_yz(
                        grid,
                        yaxis_units,
                        zaxis_units,
                        retriever,
                        state,
                        fig,
                        ax,
                        **self.properties
                    )
                else:
                    self._slave = lambda state, fig, ax: make_contourf_xz(
                        grid,
                        xaxis_units,
                        zaxis_units,
                        retriever,
                        state,
                        fig,
                        ax,
                        **self.properties
                    )

    def __call__(
        self, state: taz_types.dataarray_dict_t, fig: plt.Figure, ax: plt.Axes
    ) -> None:
        """
        Call operator generating the contourf plot.
        """
        self._slave(state, fig, ax)


def make_contourf_xy(
    grid: "Grid",
    xaxis_units: str,
    yaxis_units: str,
    field_retriever: DataRetriever,
    state: taz_types.dataarray_dict_t,
    fig: plt.Figure,
    ax: plt.Axes,
    **kwargs
) -> None:
    field = np.squeeze(field_retriever(state))

    xv = (
        to_units(grid.x, xaxis_units).values
        if field.shape[0] == grid.nx
        else to_units(grid.x_at_u_locations, xaxis_units).values
    )
    yv = (
        to_units(grid.y, yaxis_units).values
        if field.shape[1] == grid.ny
        else to_units(grid.y_at_v_locations, yaxis_units).values
    )
    x = np.repeat(xv[:, np.newaxis], yv.shape[0], axis=1)
    y = np.repeat(yv[np.newaxis, :], xv.shape[0], axis=0)

    make_contourf(x, y, field, fig, ax, **kwargs)


def make_contourf_xz(
    grid: "Grid",
    xaxis_units: str,
    zaxis_units: str,
    field_retriever: DataRetriever,
    state: taz_types.dataarray_dict_t,
    fig: plt.Figure,
    ax: plt.Axes,
    **kwargs
) -> None:
    field = np.squeeze(field_retriever(state))

    xv = (
        to_units(grid.x, xaxis_units).values
        if field.shape[0] == grid.nx
        else to_units(grid.x_at_u_locations, xaxis_units).values
    )
    zv = (
        to_units(grid.z, zaxis_units).values
        if field.shape[1] == grid.nz
        else to_units(grid.z_on_interface_levels, zaxis_units).values
    )
    x = np.repeat(xv[:, np.newaxis], zv.shape[0], axis=1)
    z = np.repeat(zv[np.newaxis, :], xv.shape[0], axis=0)

    make_contourf(x, z, field, fig, ax, **kwargs)


def make_contourf_xh(
    grid: "Grid",
    xaxis_units: str,
    zaxis_retriever: DataRetriever,
    field_retriever: DataRetriever,
    state: taz_types.dataarray_dict_t,
    fig: plt.Figure,
    ax: plt.Axes,
    **kwargs
) -> None:
    field = np.squeeze(field_retriever(state))

    zv = np.squeeze(zaxis_retriever(state))
    field = (
        0.5 * (field[:-1, :] + field[1:, :]) if field.shape[0] > zv.shape[0] else field
    )
    zv = 0.5 * (zv[:-1, :] + zv[1:, :]) if zv.shape[0] > field.shape[0] else zv
    field = (
        0.5 * (field[:, :-1] + field[:, 1:]) if field.shape[1] > zv.shape[1] else field
    )
    z = 0.5 * (zv[:, :-1] + zv[:, 1:]) if zv.shape[1] > field.shape[1] else zv

    xv = (
        to_units(grid.x, xaxis_units).values
        if field.shape[0] == grid.nx
        else to_units(grid.x_at_u_locations, xaxis_units).values
    )
    x = np.repeat(xv[:, np.newaxis], z.shape[1], axis=1)

    make_contourf(x, z, field, fig, ax, **kwargs)


def make_contourf_yz(
    grid: "Grid",
    yaxis_units: str,
    zaxis_units: str,
    field_retriever: DataRetriever,
    state: taz_types.dataarray_dict_t,
    fig: plt.Figure,
    ax: plt.Axes,
    **kwargs
) -> None:
    field = np.squeeze(field_retriever(state))

    yv = (
        to_units(grid.y, yaxis_units).values
        if field.shape[0] == grid.ny
        else to_units(grid.y_at_v_locations, yaxis_units).values
    )
    zv = (
        to_units(grid.z, zaxis_units).values
        if field.shape[1] == grid.nz
        else to_units(grid.z_on_interface_levels, zaxis_units).values
    )
    y = np.repeat(yv[:, np.newaxis], zv.shape[0], axis=1)
    z = np.repeat(zv[np.newaxis, :], yv.shape[0], axis=0)

    make_contourf(y, z, field, fig, ax, **kwargs)


def make_contourf_yh(
    grid: "Grid",
    yaxis_units: str,
    zaxis_retriever: DataRetriever,
    field_retriever: DataRetriever,
    state: taz_types.dataarray_dict_t,
    fig: plt.Figure,
    ax: plt.Axes,
    **kwargs
) -> None:
    field = np.squeeze(field_retriever(state))

    zv = np.squeeze(zaxis_retriever(state))
    field = (
        0.5 * (field[:-1, :] + field[1:, :]) if field.shape[0] > zv.shape[0] else field
    )
    zv = 0.5 * (zv[:-1, :] + zv[1:, :]) if zv.shape[0] > field.shape[0] else zv
    field = (
        0.5 * (field[:, :-1] + field[:, 1:]) if field.shape[1] > zv.shape[1] else field
    )
    z = 0.5 * (zv[:, :-1] + zv[:, 1:]) if zv.shape[1] > field.shape[1] else zv

    yv = (
        to_units(grid.y, yaxis_units).values
        if field.shape[0] == grid.ny
        else to_units(grid.y_at_v_locations, yaxis_units).values
    )
    y = np.repeat(yv[:, np.newaxis], z.shape[1], axis=1)

    make_contourf(y, z, field, fig, ax, **kwargs)
