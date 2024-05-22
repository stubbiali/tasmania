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

from datetime import timedelta
import numpy as np

import sympl

from tasmania.domain.grid import Grid
from tasmania.utils.constants import get_physical_constants
from tasmania.utils.utils import (
    equal_to as eq,
    smaller_than as lt,
    smaller_or_equal_than as le,
    greater_than as gt,
)


# Default values for the physical constants used in the module
_d_physical_constants = {
    "air_pressure_at_sea_level": sympl.DataArray(1e5, attrs={"units": "Pa"}),
    "air_temperature_at_sea_level": sympl.DataArray(288.15, attrs={"units": "K"}),
    "beta": sympl.DataArray(42.0, attrs={"units": "K Pa^-1"}),
    "gas_constant_of_dry_air": sympl.DataArray(287.05, attrs={"units": "J K^-1 kg^-1"}),
    "gravitational_acceleration": sympl.DataArray(9.80665, attrs={"units": "m s^-2"}),
}


class GalChen3d(Grid):
    """
    This class inherits :class:`~tasmania.domain.grid_xyz.GridXYZ` to represent
    a rectangular and regular computational grid embedded in a three-dimensional
    terrain-following reference system, whose coordinates are:

        * the first horizontal coordinate :math:`x`, e.g., the longitude;
        * the second horizontal coordinate :math:`y`, e.g., the latitude;
        * the Gal-Chen terrain-following coordinate :math:`\mu`.

    The vertical coordinate :math:`\mu` may be formulated to define a hybrid
    terrain-following coordinate system with terrain-following coordinate lines
    between the surface terrain-height and :math:`\mu = \mu_F`, where
    :math:`\mu`-coordinate lines change back to flat horizontal lines.

    Attributes
    ----------
    height : sympl.DataArray
        3-D :class:`~sympl.DataArray` representing the geometric height
        of the main levels (in [m]).
    height_on_interface_levels : sympl.DataArray
        3-D :class:`~sympl.DataArray` representing the geometric height
        of the half levels (in [m]).
    height_interface : sympl.DataArray
        Geometric height corresponding to :math:`\mu = \mu_F` (in [m]).
    reference_pressure : sympl.DataArray
        3-D :class:`~sympl.DataArray` representing the reference pressure
        at the main levels (in [m]).
    reference_pressure_on_interface_levels : sympl.DataArray
        3-D :class:`~sympl.DataArray` representing the reference pressure
        at the half levels (in [m]).
    """

    def __init__(
        self,
        domain_x,
        nx,
        domain_y,
        ny,
        domain_z,
        nz,
        z_interface=None,
        topo_type="flat_terrain",
        topo_time=timedelta(),
        topo_kwargs=None,
        physical_constants=None,
        dtype=np.float64,
    ):
        """
        Constructor.

        Parameters
        ----------
        domain_x : sympl.DataArray
            2-items :class:`~sympl.DataArray` storing the end-points of the interval
            which the domain includes along the :math:`x`-axis, as well as the axis
            dimension and units.
        nx : int
            Number of mass points in the :math:`x`-direction.
        domain_y : sympl.DataArray
            2-items :class:`~sympl.DataArray` storing the end-points of the interval
            which the domain includes along the :math:`y`-axis, as well as the axis
            dimension and units.
        ny : int
            Number of mass points in the :math:`y`-direction.
        domain_z : sympl.DataArray
            2-items :class:`~sympl.DataArray` storing the end-points of the interval
            which the domain includes along the :math:`z`-axis, as well as the axis
            dimension and units. The interval should be specified in the form
            :math:`(z_{top}, ~ z_{surface})`.
        nz : int
            Number of vertical main levels.
        z_interface : `sympl.DataArray`, optional
            Interface value :math:`z_F`. If not specified, it is assumed that
            :math:`z_F = z_T`, with :math:`z_T` the value of :math:`z` at the
            top of the domain. In other words, the coordinate system is supposed
            fully terrain-following.
        topo_type : `str`, optional
            Topography type. Defaults to 'flat_terrain'.
            See :class:`~tasmania.domain.topography.Topography1d` for further details.
        topo_time : `timedelta`, optional
            :class:`datetime.timedelta` representing the simulation time after
            which the topography should stop increasing. Default is 0, corresponding
            to a time-invariant terrain surface-height. See
            :mod:`~tasmania.domain.topography.Topography1d` for further details.
        topo_kwargs : `dict`, optional
            Keyword arguments to be forwarded to the constructor of
            :class:`~tasmania.domain.topography.Topography1d`.
        physical_constants : `dict[str, sympl.DataArray]`, optional
            Dictionary whose keys are strings indicating physical constants used
            within this object, and whose values are :class:`~sympl.DataArray`\s
            storing the values and units of those constants. The constants might be:

                * 'air_pressure_at_sea_level', in units compatible with [Pa];
                * 'air_temperature_at_sea_level', in units compatible with [K];
                * 'beta' (the rate of increase in reference temperature with the \
                    logarithm of reference pressure), in units compatible with \
                    ([K ~ Pa^-1]);
                * 'gas_constant_of_dry_air', in units compatible with \
                    ([J K^-1 kg:math:`^{-1}`]);
                * 'gravitational acceleration', in units compatible with [m s^-2].

            Please refer to
            :func:`tasmania.utils.data_utils.get_physical_constants` and
            :obj:`tasmania.domain.gal_chen._d_physical_constants`
            for the default values.
        dtype : `obj`, optional
            Instance of :class:`data-type` specifying the data type for
            any :class:`gt4py.storage.storage.Storage` used within this class.
            Defaults to :obj:`numpy.float64`.

        Raises
        ------
        ValueError :
            If the vertical coordinate either assumes negative values, or
            does not vanish at the terrain surface.
        ValueError :
            If `z_interface` is outside the domain.
        """
        # Ensure th vertical axis is expressed in meters
        domain_z_conv = sympl.DataArray(
            domain_z.to_units("m").values,
            dims="atmosphere_hybrid_height_coordinate",
            attrs={"units": "m"},
        )

        # Preliminary checks
        if not (eq(domain_z_conv.values[1], 0.0) and gt(domain_z_conv.values[0], 0.0)):
            raise ValueError(
                "Gal-Chen vertical coordinate should be positive "
                "and vanish at the terrain surface."
            )

        # Call parent's constructor
        super().__init__(
            domain_x,
            nx,
            domain_y,
            ny,
            domain_z_conv,
            nz,
            z_interface=z_interface,
            topo_type=topo_type,
            topo_time=topo_time,
            topo_kwargs=topo_kwargs,
            dtype=dtype,
        )

        # Interface height
        self.height_interface = self.z_interface.to_units("m")

        # Keep track of the physical constants to use
        self._physical_constants = get_physical_constants(_d_physical_constants, physical_constants)

        # Compute geometric height and reference pressure
        self._update_metric_terms()

    def update_topography(self, time):
        """
        Update the (time-dependent) topography, then re-compute the metric.

        Parameters
        ----------
        time : datetime.timedelta
            :class:`datetime.timedelta` representing the elapsed simulation time.
        """
        super().update_topography(time)
        self._update_metric_terms()

    def _update_metric_terms(self):
        """
        Compute the metric terms, i.e., the geometric height and the
        reference pressure, at both half and main levels. In doing this,
        a logarithmic vertical profile of reference pressure is assumed.
        This method should be called every time the topography is updated or changed.
        """
        # Shortcuts
        p_sl = self._physical_constants["air_pressure_at_sea_level"]
        T_sl = self._physical_constants["air_temperature_at_sea_level"]
        beta = self._physical_constants["beta"]
        Rd = self._physical_constants["gas_constant_of_dry_air"]
        g = self._physical_constants["gravitational_acceleration"]
        hs = np.repeat(self.topography_height[:, :, np.newaxis], self.nz + 1, axis=2)
        zv = np.reshape(
            self.z_on_interface_levels.values[:, np.newaxis, np.newaxis],
            (1, 1, self.nz + 1),
        )
        zf = self.z_interface.values.item()

        # Geometric height at the interface levels
        a = np.tile(zv, (self.nx, self.ny, 1))
        b = (zf - zv) / zf * (np.logical_and(le(0.0, zv), lt(zv, zf)))
        b = np.tile(b, (self.nx, self.ny, 1))
        z_hl = a + b * hs
        self.height_on_interface_levels = sympl.DataArray(
            z_hl,
            coords=[
                self.x.values,
                self.y.values,
                self.z_on_interface_levels.values,
            ],
            dims=[
                self.x.dims[0],
                self.y.dims[0],
                self.z_on_interface_levels.dims[0],
            ],
            name="height_on_interface_levels",
            attrs={"units": "m"},
        )

        # Reference pressure at the interface levels
        if eq(beta, 0.0):
            p0_hl = p_sl * np.exp(-g * z_hl / (Rd * T_sl))
        else:
            p0_hl = p_sl * np.exp(
                -T_sl / beta * (1.0 - np.sqrt(1.0 - 2.0 * beta * g * z_hl / (Rd * T_sl**2)))
            )
        self.reference_pressure_on_interface_levels = sympl.DataArray(
            p0_hl,
            coords=[
                self.x.values,
                self.y.values,
                self.z_on_interface_levels.values,
            ],
            dims=[
                self.x.dims[0],
                self.y.dims[0],
                self.z_on_interface_levels.dims[0],
            ],
            name="reference_pressure_on_interface_levels",
            attrs={"units": "Pa"},
        )

        # Reference pressure at the main levels
        self.reference_pressure = sympl.DataArray(
            0.5 * (p0_hl[:, :, :-1] + p0_hl[:, :, 1:]),
            coords=[self.x.values, self.y.values, self.z.values],
            dims=[self.x.dims[0], self.y.dims[0], self.z.dims[0]],
            name="reference_pressure",
            attrs={"units": "Pa"},
        )

        # Geometric height at the main levels
        self.height = sympl.DataArray(
            0.5 * (z_hl[:, :, :-1] + z_hl[:, :, 1:]),
            coords=[self.x.values, self.y.values, self.z.values],
            dims=[self.x.dims[0], self.y.dims[0], self.z.dims[0]],
            name="height",
            attrs={"units": "m"},
        )
