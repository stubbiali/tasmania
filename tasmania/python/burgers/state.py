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
import numpy as np
import pint

from tasmania.python.utils.storage_utils import get_dataarray_3d, zeros


class ZhaoSolutionFactory:
    """ Factory of valid velocity fields for the Zhao test case. """

    def __init__(self, initial_time, eps):
        """
        Parameters
        ----------
        initial_time : datetime.datetime
            The starting time of the simulation.
        eps : sympl.DataArray
            1-item :class:`sympl.DataArray` representing the diffusivity.
            The units should be compatible with 'm s^-2'.
        """
        self._itime = initial_time
        self._eps = eps.to_units("m^2 s^-1").values.item()

        self._ureg = pint.UnitRegistry()

    def __call__(
        self,
        time,
        grid,
        slice_x=None,
        slice_y=None,
        field_name="x_velocity",
        field_units=None,
    ):
        """
        Parameters
        ----------
        time : datetime.datetime
            The time instant when the solution should be computed.
        grid : tasmania.Grid
            The underlying grid.
        slice_x : `slice`, optional
            The portion of the grid along the x-axis where the solution should be computed.
            If not specified, the solution is calculated over all grid points in x-direction.
        slice_y : `slice`, optional
            The portion of the grid along the y-axis where the solution should be computed.
            If not specified, the solution is calculated over all grid points in y-direction.
        field_name : `str`, optional
            The field to calculate. Either:

                * 'x_velocity' (default);
                * 'y_velocity'.

        field_units : `str`, optional
            The field units, which should be compatible with m s^-1.

        Return
        ------
        numpy.ndarray :
            The computed model variable.
        """
        eps = self._eps
        ureg = self._ureg

        slice_x = slice(0, grid.nx) if slice_x is None else slice_x
        slice_y = slice(0, grid.ny) if slice_y is None else slice_y

        mi = slice_x.stop - slice_x.start
        mj = slice_y.stop - slice_y.start

        x = grid.x.to_units("m").values[slice_x]
        x = np.tile(x[:, np.newaxis, np.newaxis], (1, mj, grid.nz))
        y = grid.y.to_units("m").values[slice_y]
        y = np.tile(y[np.newaxis, :, np.newaxis], (mi, 1, grid.nz))

        t = (time - self._itime).total_seconds()

        if field_units is None or field_units == "m s^-1":
            factor = 1.0
        else:
            factor = (1.0 * ureg.meter / ureg.second).to(field_units).magnitude

        if field_name == "x_velocity":
            tmp = (
                -2.0
                * eps
                * 2.0
                * np.pi
                * np.exp(-5.0 * np.pi ** 2 * eps * t)
                * np.cos(2.0 * np.pi * x)
                * np.sin(np.pi * y)
                / (
                    2.0
                    + np.exp(-5.0 * np.pi ** 2 * eps * t)
                    * np.sin(2.0 * np.pi * x)
                    * np.sin(np.pi * y)
                )
            )
        elif field_name == "y_velocity":
            tmp = (
                -2.0
                * eps
                * np.pi
                * np.exp(-5.0 * np.pi ** 2 * eps * t)
                * np.sin(2.0 * np.pi * x)
                * np.cos(np.pi * y)
                / (
                    2.0
                    + np.exp(-5.0 * np.pi ** 2 * eps * t)
                    * np.sin(2.0 * np.pi * x)
                    * np.sin(np.pi * y)
                )
            )
        else:
            raise ValueError()

        return factor * tmp


class ZhaoStateFactory:
    """ Factory of valid states for the Zhao test case. """

    def __init__(self, initial_time, eps, *, backend, dtype, default_origin):
        """
        Parameters
        ----------
        initial_time : datetime
            The initial time of the simulation.
        eps : sympl.DataArray
            1-item :class:`sympl.DataArray` representing the diffusivity.
            The units should be compatible with 'm s^-2'.
        backend : str
            The GT4Py backend.
        dtype : numpy.dtype
            Data type of the storages.
        default_origin : tuple
            Storage default origin.
        """
        self._solution_factory = ZhaoSolutionFactory(initial_time, eps)
        self._backend = backend
        self._dtype = dtype
        self._default_origin = default_origin

    def __call__(self, time, grid):
        """
        Parameters
        ----------
        grid : tasmania.Grid
            The underlying grid.
        time : datetime.datetime
            The temporal instant.

        Return
        ------
        dict :
            The computed model state dictionary.
        """
        nx, ny = grid.nx, grid.ny
        backend = self._backend
        dtype = self._dtype
        default_origin = self._default_origin

        u = zeros((nx, ny, 1), backend, dtype, default_origin=default_origin)
        u[...] = self._solution_factory(time, grid, field_name="x_velocity")
        u_da = get_dataarray_3d(u, grid, "m s^-1", "x_velocity", set_coordinates=False)
        u_da.attrs["backend"] = backend
        u_da.attrs["default_origin"] = default_origin

        v = zeros((nx, ny, 1), backend, dtype, default_origin=default_origin)
        v[...] = self._solution_factory(time, grid, field_name="y_velocity")
        v_da = get_dataarray_3d(v, grid, "m s^-1", "y_velocity", set_coordinates=False)
        v_da.attrs["backend"] = backend
        v_da.attrs["default_origin"] = default_origin

        return {"time": time, "x_velocity": u_da, "y_velocity": v_da}
