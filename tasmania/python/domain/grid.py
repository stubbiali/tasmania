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
import math
import numpy as np
from sympl import DataArray
from typing import Any, Dict, Optional, TYPE_CHECKING

from tasmania.python.domain.horizontal_grid import (
    PhysicalHorizontalGrid,
    NumericalHorizontalGrid,
)
from tasmania.python.domain.topography import PhysicalTopography, NumericalTopography
from tasmania.python.utils import taz_types
from tasmania.python.utils.utils import smaller_than as lt, smaller_or_equal_than as le

if TYPE_CHECKING:
    from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
    from tasmania.python.domain.topography import Topography
    from tasmania.python.domain.horizontal_grid import HorizontalGrid


class Grid:
    """ Three-dimensional rectilinear grid.

    The grid is embedded in a reference system whose coordinates are:

        * two horizontal coordinates :math:`x` and :math:`y`;
        * a vertical (terrain-following) coordinate :math:`z`.

    The vertical coordinate :math:`z` may be formulated to define a hybrid
    terrain-following coordinate system with terrain-following coordinate
    lines between the surface and :math:`z = z_F`, where :math:`z`-coordinate
    lines change back to flat horizontal lines.

    No assumption is made on the actual nature of :math:`z`, which may be either
    pressure-based or height-based.

    This class stores:

        * the :class:`~tasmania.HorizontalGrid` covering the horizontal domain;
        * the vertical discretization;
        * the underlying :class:`~tasmania.Topography`.

    Note
    ----
    For the sake of compliance with the `COSMO model <http://cosmo-model.org>`_,
    the vertical grid points are ordered from the top of the domain to the surface.
    """

    def __init__(
        self,
        grid_xy: "HorizontalGrid",
        z: DataArray,
        z_on_interface_levels: DataArray,
        z_interface: DataArray,
        topography: "Topography",
    ) -> None:
        """
        Parameters
        ----------
        grid_xy : tasmania.HorizontalGrid
            The horizontal grid.
        z : sympl.DataArray
            1-D numerical collecting the vertical coordinates
            of the vertical main levels.
        z_on_interface_levels : sympl.DataArray
            1-D :class:`~sympl.DataArray` collecting the vertical coordinates
            of the vertical interface levels.
        z_interface : sympl.DataArray
            1-item :class:`~sympl.DataArray` representing the interface
            altitude :math:`z_F`.
        topography : tasmania.Topography
            The underlying topography.
        """
        self._grid_xy = grid_xy
        self._z = z
        self._zhl = z_on_interface_levels
        self._zi = z_interface
        self._topo = topography

        self._nz = z.values.shape[0]
        dz_v = math.fabs(self._zhl.values[0] - self._zhl.values[-1]) / self._nz
        dz_v = 1.0 if dz_v == 0.0 else dz_v
        self._dz = DataArray(dz_v, name="dz", attrs={"units": z.attrs["units"]})

    @property
    def grid_xy(self) -> "HorizontalGrid":
        """
        The underlying :class:`~tasmania.HorizontalGrid`.
        """
        return self._grid_xy

    @property
    def x(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the coordinates of the mass
        grid points along the first horizontal dimension.
        """
        return self._grid_xy.x

    @property
    def x_at_u_locations(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the coordinates of the
        staggered grid points along the first horizontal dimension.
        """
        return self._grid_xy.x_at_u_locations

    @property
    def nx(self) -> int:
        """
        Number of mass grid points featured by the grid along
        the first horizontal dimension.
        """
        return self._grid_xy.nx

    @property
    def dx(self) -> DataArray:
        """
        1-item :class:`~sympl.DataArray` representing the grid spacing
        along the first horizontal dimension.
        """
        return self._grid_xy.dx

    @property
    def y(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the coordinates of the mass
        grid points along the second horizontal dimension.
        """
        return self._grid_xy.y

    @property
    def y_at_v_locations(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the coordinates of the
        staggered grid points along the second horizontal dimension.
        """
        return self._grid_xy.y_at_v_locations

    @property
    def ny(self) -> int:
        """
        Number of mass grid points featured by the grid along
        the second horizontal dimension.
        """
        return self._grid_xy.ny

    @property
    def dy(self) -> DataArray:
        """
        1-item :class:`~sympl.DataArray` representing the grid spacing
        along the second horizontal dimension.
        """
        return self._grid_xy.dy

    @property
    def z(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the vertical coordinates of the
        vertical main levels.
        """
        return self._z

    @property
    def z_on_interface_levels(self) -> DataArray:
        """
        1-D :class:`~sympl.DataArray` collecting the vertical coordinates of the
        vertical interface levels.
        """
        return self._zhl

    @property
    def nz(self) -> int:
        """
        Number of vertical main levels.
        """
        return self._nz

    @property
    def dz(self) -> DataArray:
        """
        1-item :class:`~sympl.DataArray` representing the vertical
        grid spacing.
        """
        return self._dz

    @property
    def z_interface(self) -> DataArray:
        """
        1-item :class:`~sympl.DataArray` representing the interface
        altitude where the terrain-following coordinate surfaces
        flat black to horizontal lines.
        """
        return self._zi

    @property
    def topography(self) -> "Topography":
        """
        The :class:`~tasmania.Topography` defined over the underlying
        :class:`~tasmania.HorizontalGrid`.
        """
        return self._topo

    def update_topography(self, time: taz_types.datetime_t) -> None:
        """ Update the underlying (time-dependent) :class:`~tasmania.Topography`.

        Parameters
        ----------
        time : datetime.timedelta
            The elapsed simulation time.
        """
        self._topo.update(time)


class PhysicalGrid(Grid):
    """ Three-dimensional rectilinear grid covering a *physical* domain. """

    def __init__(
        self,
        domain_x: DataArray,
        nx: int,
        domain_y: DataArray,
        ny: int,
        domain_z: DataArray,
        nz: int,
        z_interface: Optional[DataArray] = None,
        topography_type: str = "flat",
        topography_kwargs: Dict[str, Any] = None,
        dtype: taz_types.dtype_t = np.float64,
    ) -> None:
        """
        Parameters
        ----------
        domain_x : sympl.DataArray
            2-items :class:`~sympl.DataArray` storing the end-points, dimension
            and units of the interval which the physical domain includes along
            the first horizontal dimension.
        nx : int
            Number of mass points featured by the *physical* grid
            along the first horizontal dimension.
        domain_y : sympl.DataArray
            2-items :class:`~sympl.DataArray` storing the end-points, dimension
            and units of the interval which the physical domain includes along
            the second horizontal dimension.
        ny : int
            Number of mass points featured by the *physical* grid
            along the second horizontal dimension.
        domain_z : sympl.DataArray
            2-items :class:`~sympl.DataArray` storing the end-points, dimension
            and units of the interval which the domain includes along the
            :math:`z`-axis. The interval should be specified in the form
            :math:`(z_{top}, ~ z_{surface})`.
        nz : int
            Number of vertical main levels.
        z_interface : `sympl.DataArray`, optional
            Interface value :math:`z_F`. If not specified, it is assumed that
            :math:`z_F = z_T`, with :math:`z_T` the value of :math:`z` at the
            top of the domain. In other words, the coordinate system is supposed
            fully terrain-following.
        topography_type : `str`, optional
            The topography type. Defaults to 'flat'.
                 See :class:`~tasmania.PhysicalTopography` for all available options.
        topography_kwargs : `dict`, optional
            Keyword arguments to be forwarded to the constructor of
            :class:`~tasmania.PhysicalTopography`.
        dtype : `data-type`, optional
            The data-type of the storages.

        Raises
        ------
        ValueError :
            If ``interface`` lays outside the domain.
        """
        # xy-grid
        grid_xy = PhysicalHorizontalGrid(domain_x, nx, domain_y, ny, dtype=dtype)

        # extract z-axis properties
        values_z = domain_z.values
        dims_z = domain_z.dims
        dims_zhl = domain_z.dims[0] + "_on_interface_levels"
        units_z = domain_z.attrs["units"]

        # z-coordinates of the half-levels
        zhl_v = np.linspace(values_z[0], values_z[1], nz + 1, dtype=dtype)
        zhl = DataArray(
            zhl_v,
            coords=[zhl_v],
            dims=dims_zhl,
            name="z_on_interface_levels",
            attrs={"units": units_z},
        )

        # z-coordinates of the main-levels
        z_v = 0.5 * (zhl_v[:-1] + zhl_v[1:])
        z = DataArray(z_v, coords=[z_v], dims=dims_z, name="z", attrs={"units": units_z})

        # z-interface
        if z_interface is None:
            zi = DataArray(values_z[0], attrs={"units": units_z})
        else:
            zi = z_interface.to_units(units_z)

        # checks
        zi_v = zi.values.item()
        if lt(values_z[0], values_z[1]):
            if not (le(values_z[0], zi_v) and le(zi_v, values_z[1])):
                raise ValueError(
                    "z_interface should be in the range ({}, {}).".format(
                        values_z[0], values_z[1]
                    )
                )
        else:
            if not (le(values_z[1], zi_v) and le(zi_v, values_z[0])):
                raise ValueError(
                    "z_interface should be in the range ({}, {}).".format(
                        values_z[1], values_z[0]
                    )
                )

        # underlying topography
        kwargs = (
            {}
            if (topography_kwargs is None or not isinstance(topography_kwargs, dict))
            else topography_kwargs
        )
        topo = PhysicalTopography.factory(topography_type, grid_xy, **kwargs)

        # call parent's constructor
        super().__init__(grid_xy, z, zhl, zi, topo)


class NumericalGrid(Grid):
    """ Three-dimensional rectilinear grid covering a *numerical* domain. """

    def __init__(self, phys_grid: PhysicalGrid, boundary: "HorizontalBoundary") -> None:
        """
        Parameters
        ----------
        phys_grid : tasmania.PhysicalGrid
            The associated physical grid.
        boundary : tasmania.HorizontalBoundary
            The object handling the lateral boundary conditions.
        """
        # the horizontal grid
        phys_grid_xy = phys_grid.grid_xy
        grid_xy = NumericalHorizontalGrid(phys_grid_xy, boundary)

        # the vertical discretization
        z = phys_grid.z
        zhl = phys_grid.z_on_interface_levels
        zi = phys_grid.z_interface

        # the underlying topography
        phys_topo = phys_grid.topography
        topo = NumericalTopography(grid_xy, phys_topo, boundary)

        # call parent's constructor
        super().__init__(grid_xy, z, zhl, zi, topo)
