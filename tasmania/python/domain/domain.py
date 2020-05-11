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
import functools
import numpy as np
from sympl import DataArray
from typing import Any, Dict, Optional

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.grid import PhysicalGrid, NumericalGrid
from tasmania.python.utils import taz_types


class Domain:
    """ Discrete spatial domain.

    This class instantiates, maintains and exposes:

    * the :class:`~tasmania.PhysicalGrid` covering the physical domain;
    * the associated :class:`~tasmania.NumericalGrid`;
    * the proper :class:`~tasmania.HorizontalBoundary` handling the \
        lateral boundary conditions.
    """

    def __init__(
        self,
        domain_x: DataArray,
        nx: int,
        domain_y: DataArray,
        ny: int,
        domain_z: DataArray,
        nz: int,
        z_interface: Optional[DataArray] = None,
        horizontal_boundary_type: str = "periodic",
        nb: int = 3,
        horizontal_boundary_kwargs: Optional[Dict[str, Any]] = None,
        topography_type: str = "flat",
        topography_kwargs: Optional[Dict[str, Any]] = None,
        gt_powered: bool = True,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        storage_shape: Optional[taz_types.triplet_int_t] = None,
        managed_memory: bool = False
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
        horizontal_boundary_type : `str`, optional
            The type of lateral boundary conditions. Defaults to ``'periodic'``.
            See :class:`tasmania.HorizontalBoundary` for all available options.
        nb : `int`, optional
            Number of boundary layers. Defaults to 3.
        horizontal_boundary_kwargs : `dict`, optional
            Keyword arguments to be broadcast to
            :meth:`tasmania.HorizontalBoundary.factory`.
        topography_type : `str`, optional
            Topography type. Defaults to ``'flat'``.
            See :class:`tasmania.Topography` for available options.
        topography_kwargs : `dict`, optional
            Keyword arguments to be forwarded to the constructor of
            :class:`tasmania.Topography`.
        gt_powered : bool
            ``True`` to harness GT4Py, ``False`` for a vanilla Numpy implementation.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            The data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple[int]`, optional
            The default origin of the storages.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class instantiation,
            ``False`` to rely on the caching mechanism implemented by GT4Py.
        storage_shape : `tuple[int]`, optional
            The shape of the storages allocated within this class.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory, ``False`` otherwise.
        """
        # the physical grid
        topo_kwargs = (
            {}
            if (topography_kwargs is None or not isinstance(topography_kwargs, dict))
            else topography_kwargs
        )
        self._pgrid = PhysicalGrid(
            domain_x,
            nx,
            domain_y,
            ny,
            domain_z,
            nz,
            z_interface=z_interface,
            topography_type=topography_type,
            topography_kwargs=topo_kwargs,
            dtype=dtype,
        )

        # the object handling the horizontal boundary conditions
        hb_kwargs = (
            {}
            if (
                horizontal_boundary_kwargs is None
                or not isinstance(horizontal_boundary_kwargs, dict)
            )
            else horizontal_boundary_kwargs
        )
        self._hb = HorizontalBoundary.factory(
            horizontal_boundary_type,
            nx,
            ny,
            nb,
            gt_powered=gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            exec_info=exec_info,
            default_origin=default_origin,
            rebuild=rebuild,
            storage_shape=storage_shape,
            managed_memory=managed_memory,
            **hb_kwargs
        )

        # the numerical grid
        self._cgrid = NumericalGrid(self._pgrid, self._hb)

    @property
    def physical_grid(self) -> PhysicalGrid:
        """ The :class:`~tasmania.PhysicalGrid`. """
        return self._pgrid

    @property
    def numerical_grid(self) -> NumericalGrid:
        """ The :class:`~tasmania.NumericalGrid`. """
        return self._cgrid

    @property
    def horizontal_boundary(self) -> HorizontalBoundary:
        """
        Instance of :class:`~tasmania.HorizontalBoundary` handling the boundary
        conditions.

        This object is enriched with the following new methods:

        * ``dmn_enforce_field``,
        * ``dmn_enforce_raw``,
        * ``dmn_enforce``,
        * ``dmn_set_outermost_layers_x``, and
        * ``dmn_set_outermost_layers_y``.
        """
        hb = self._hb

        hb.dmn_enforce_field = functools.partial(
            hb.enforce_field, grid=self.numerical_grid
        )
        hb.dmn_enforce_raw = functools.partial(hb.enforce_raw, grid=self.numerical_grid)
        hb.dmn_enforce = functools.partial(hb.enforce, grid=self.numerical_grid)
        hb.dmn_set_outermost_layers_x = functools.partial(
            hb.set_outermost_layers_x, grid=self.numerical_grid
        )
        hb.dmn_set_outermost_layers_y = functools.partial(
            hb.set_outermost_layers_y, grid=self.numerical_grid
        )

        return self._hb

    def update_topography(self, time: taz_types.datetime_t) -> None:
        """ Update the (time-dependent) :class:`~tasmania.Topography`.

        Parameters
        ----------
        time : datetime.timedelta
            The elapsed simulation time.
        """
        self._pgrid.update_topography(time)
        self._cgrid.update_topography(time)
