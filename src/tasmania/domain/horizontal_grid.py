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

from __future__ import annotations
import numpy as np
from typing import TYPE_CHECKING

from sympl import DataArray

from tasmania.framework.options import StorageOptions

if TYPE_CHECKING:
    from typing import Optional

    from tasmania.domain.horizontal_boundary import HorizontalBoundary


class HorizontalGrid:
    """A two-dimensional rectilinear grid.

    The grid is embedded in a reference system whose coordinates are, in the
    order, :math:`x` and :math:`y`. No assumption is made on the nature of the
    coordinates. For instance, :math:`x` may be the longitude (in which case
    :math:`x \equiv \lambda`) and :math:`y` may be the latitude (in which case
    :math:`y \equiv \phi`).
    """

    def __init__(
        self,
        x: DataArray,
        y: DataArray,
        x_at_u_locations: Optional[DataArray] = None,
        y_at_v_locations: Optional[DataArray] = None,
        *,
        storage_options: Optional[StorageOptions] = None,
    ) -> None:
        """
        Parameters
        ----------
        x : sympl.DataArray
            1-D :class:`~sympl.DataArray` collecting the coordinates of the mass
            grid points along the first horizontal dimension.
        y : sympl.DataArray
            1-D :class:`~sympl.DataArray` collecting the coordinates of the mass
            grid points along the second horizontal dimension.
        x_at_u_locations : `sympl.DataArray`, optional
            1-D :class:`~sympl.DataArray` collecting the coordinates of the
            staggered grid points along the first horizontal dimension. If not
            given, these are retrieved from `x`.
        y_at_v_locations : `sympl.DataArray`, optional
            1-D :class:`~sympl.DataArray` collecting the coordinates of the
            staggered grid points along the second horizontal dimension. If not
            given, these are retrieved from `y`.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        # storage properties
        so = storage_options or StorageOptions()
        dtype = so.dtype

        # x-coordinates of the mass points
        self._x = x

        # number of mass points along the x-axis
        nx = x.values.shape[0]
        self._nx = nx

        # x-spacing
        dx_v = 1.0 if nx == 1 else (x.values[-1] - x.values[0]) / (nx - 1)
        dx_v = dx_v if dx_v != 0.0 else 1.0
        self._dx = DataArray(dx_v, name="dx", attrs={"units": x.attrs["units"]})

        # x-coordinates of the x-staggered points
        if x_at_u_locations is not None:
            self._xu = x_at_u_locations
        else:
            xu_v = np.linspace(
                x.values[0] - 0.5 * dx_v,
                x.values[-1] + 0.5 * dx_v,
                nx + 1,
                dtype=dtype,
            )
            self._xu = DataArray(
                xu_v,
                coords=[xu_v],
                dims=(x.dims[0] + "_at_u_locations"),
                name=x.dims[0] + "_at_u_locations",
                attrs={"units": x.attrs["units"]},
            )

        # y-coordinates of the mass points
        self._y = y

        # number of mass points along the y-axis
        ny = self._y.values.shape[0]
        self._ny = ny

        # y-spacing
        dy_v = 1.0 if ny == 1 else (y.values[-1] - y.values[0]) / (ny - 1)
        dy_v = dy_v if dy_v != 0.0 else 1.0
        self._dy = DataArray(dy_v, name="dy", attrs={"units": y.attrs["units"]})

        # y-coordinates of the y-staggered points
        if y_at_v_locations is not None:
            self._yv = y_at_v_locations
        else:
            yv_v = np.linspace(
                y.values[0] - 0.5 * dy_v,
                y.values[-1] + 0.5 * dy_v,
                ny + 1,
                dtype=dtype,
            )
            self._yv = DataArray(
                yv_v,
                coords=[yv_v],
                dims=(y.dims[0] + "_at_v_locations"),
                name=y.dims[0] + "_at_v_locations",
                attrs={"units": y.attrs["units"]},
            )

    @property
    def dx(self) -> DataArray:
        """
        1-item :class:`~sympl.DataArray` representing the grid spacing
        along the first horizontal dimension. In case of a non-uniform
        spacing, this should be intended as the average grid spacing.
        """
        return self._dx

    @property
    def dy(self) -> DataArray:
        """
        1-item :class:`~sympl.DataArray` representing the grid spacing
        along the second horizontal dimension. In case of a non-uniform
        spacing, this should be intended as the average grid spacing.
        """
        return self._dy

    @property
    def nx(self) -> int:
        """Number of mass grid points along the first horizontal dimension."""
        return self._nx

    @property
    def ny(self) -> int:
        """Number of mass grid points along the second horizontal dimension."""
        return self._ny

    @property
    def x(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the coordinates of the mass
        grid points along the first horizontal dimension.
        """
        return self._x

    @property
    def x_at_u_locations(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the coordinates of the
        staggered grid points along the first horizontal dimension.
        """
        return self._xu

    @property
    def y(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the coordinates of the mass
        grid points along the second horizontal dimension.
        """
        return self._y

    @property
    def y_at_v_locations(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the coordinates of the
        staggered grid points along the second horizontal dimension.
        """
        return self._yv


class PhysicalHorizontalGrid(HorizontalGrid):
    """A two-dimensional regular grid covering a physical domain."""

    def __init__(
        self,
        domain_x: DataArray,
        nx: int,
        domain_y: DataArray,
        ny: int,
        *,
        storage_options: Optional[StorageOptions] = None,
    ) -> None:
        """
        Parameters
        ----------
        domain_x : sympl.DataArray
            2-items :class:`~sympl.DataArray` storing the end-points, dimension
            and units of the interval which the physical domain includes along
            the first horizontal dimension.
        nx : int
            Number of mass points along the first horizontal dimension.
        domain_y : sympl.DataArray
            2-items :class:`~sympl.DataArray` storing the end-points, dimension
            and units of the interval which the physical domain includes along
            the second horizontal dimension.
        ny : int
            Number of mass points along the second horizontal dimension.
        storage_options : `StorageOptions`, optional
            Storage-related options.

        Note
        ----
        Axes labels should use the `CF Conventions <http://cfconventions.org>`_.
        """
        # storage properties
        so = storage_options or StorageOptions()
        dtype = so.dtype

        # extract x-axis properties
        values_x = domain_x.values
        dims_x = domain_x.dims
        units_x = domain_x.attrs["units"]

        # x-coordinates of the mass points
        x_v = (
            np.array([0.5 * (values_x[0] + values_x[1])], dtype=dtype)
            if nx == 1
            else np.linspace(values_x[0], values_x[1], nx, dtype=dtype)
        )
        x = DataArray(
            x_v,
            coords=[x_v],
            dims=dims_x,
            name=dims_x[0],
            attrs={"units": units_x},
        )

        # extract y-axis properties
        values_y = domain_y.values
        dims_y = domain_y.dims
        units_y = domain_y.attrs["units"]

        # y-coordinates of the mass points
        y_v = (
            np.array([0.5 * (values_y[0] + values_y[1])], dtype=dtype)
            if ny == 1
            else np.linspace(values_y[0], values_y[1], ny, dtype=dtype)
        )
        y = DataArray(
            y_v,
            coords=[y_v],
            dims=dims_y,
            name=dims_y[0],
            attrs={"units": units_y},
        )

        # call parent's constructor
        super().__init__(x, y)


class NumericalHorizontalGrid(HorizontalGrid):
    """A two-dimensional regular grid covering a numerical domain."""

    def __init__(self, boundary: HorizontalBoundary) -> None:
        """
        Parameters
        ----------
        boundary : tasmania.HorizontalBoundary
            The object handling the horizontal boundary conditions.
        """
        # the associated physical grid
        phys_grid = boundary.physical_grid

        # x-coordinates of the mass points
        dims = "c_" + phys_grid.x.dims[0]
        x = boundary.get_numerical_xaxis(dims=dims)

        # x-coordinates of the x-staggered points
        dims = "c_" + phys_grid.x_at_u_locations.dims[0]
        xu = boundary.get_numerical_xaxis_staggered(dims=dims)

        # y-coordinates of the mass points
        dims = "c_" + phys_grid.y.dims[0]
        y = boundary.get_numerical_yaxis(dims=dims)

        # y-coordinates of the y-staggered points
        dims = "c_" + phys_grid.y_at_v_locations.dims[0]
        yv = boundary.get_numerical_yaxis_staggered(dims=dims)

        # call parent's constructor
        super().__init__(x, y, xu, yv)

        # coherency checks
        assert self.nx == boundary.ni
        assert self.ny == boundary.nj
