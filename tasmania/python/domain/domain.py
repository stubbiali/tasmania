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
from sympl import DataArray
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING

from tasmania.python.domain.grid import PhysicalGrid
from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.utils import typing as ty

if TYPE_CHECKING:
    from tasmania.python.domain.grid import NumericalGrid
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class Domain:
    """ Discrete spatial domain.

    This class instantiates, maintains and exposes:

    * the :class:`~tasmania.PhysicalGrid` covering the physical domain;
    * the associated :class:`~tasmania.NumericalGrid`;
    * the proper :class:`~tasmania.HorizontalBoundary` handling the \
        lateral boundary conditions.
    """

    def __init__(
        self: "Domain",
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
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None
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
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        # the physical grid
        topo_kwargs = (
            {}
            if (
                topography_kwargs is None
                or not isinstance(topography_kwargs, dict)
            )
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
            storage_options=storage_options,
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
            self._pgrid,
            nb,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
            **hb_kwargs
        )

    @property
    def horizontal_boundary(self: "Domain") -> HorizontalBoundary:
        """
        Instance of :class:`~tasmania.HorizontalBoundary` handling the
        boundary conditions.
        """
        return self._hb

    @property
    def numerical_grid(self: "Domain") -> "NumericalGrid":
        """The :class:`~tasmania.NumericalGrid`."""
        return self._hb.numerical_grid

    @property
    def physical_grid(self: "Domain") -> PhysicalGrid:
        """The :class:`~tasmania.PhysicalGrid`."""
        return self._pgrid

    def update_topography(self: "Domain", time: ty.datetime_t) -> None:
        """Update the (time-dependent) :class:`~tasmania.Topography`.

        Parameters
        ----------
        time : datetime.timedelta
            The elapsed simulation time.
        """
        self._pgrid.update_topography(time)
        self._hb.numerical_grid.update_topography(time)
