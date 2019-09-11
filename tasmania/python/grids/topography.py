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
"""
This module contains:
	Topography
	PhysicalTopography
	NumericalTopography
"""
from copy import deepcopy
import numpy as np
from pandas import Timedelta
from sympl import DataArray

from tasmania.python.utils.data_utils import make_dataarray_2d
from tasmania.python.utils.utils import smaller_than as lt

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float32


class Topography:
    """
    Class which represents a possibly time-dependent topography. Although
    clearly not physical, a terrain surface (slowly) growing in the early
    stages of a simulation may help to retain numerical stability, as it
    prevents steep gradients in the first few iterations.

    Letting :math:`h_s = h_s(x,y)` be a two-dimensional topography,
    with :math:`x \in [a_x,b_x]` and :math:`y \in [a_y,b_y]`, the user may
    choose among:

        * a flat terrain, i.e., :math:`h_s(x,y) \equiv 0`;
        * a Gaussian shaped-mountain, i.e.

            .. math::
                h_s(x,y) = h_{max} \exp{\left[ - \left( \\frac{x - c_x}{\sigma_x}
                \\right)^2 - \left( \\frac{y - c_y}{\sigma_y} \\right)^2 \\right]} ;

        * a modified Gaussian-shaped mountain proposed by Schaer and Durran (1997),

            .. math::
                h_s(x,y) = \\frac{h_{max}}{\left[ 1 + \left( \\frac{x - c_x}{\sigma_x}
                \\right)^2 + \left( \\frac{y - c_y}{\sigma_y} \\right)^2 \\right]^{3/2}}.

    Further, user-defined profiles are supported as well, provided that they
    admit an analytical expression. This is passed to the class as a string,
    which is then parsed in C++ via `Cython <http://cython.org>`_
    (see :class:`~tasmania.grids.parser.parser_2d`). Therefore, the string
    must be fully C++-compliant.

    Reference
    ---------
	Schaer, C., and D. R. Durran. (1997). Vortex formation and vortex shedding \
	in continuously stratified flows past isolated topography. \
	*Journal of Atmospheric Sciences*, *54*:534-554.
	"""

    def __init__(self, topography_type, steady_profile, profile=None, **kwargs):
        """
		Parameters
		----------
		topography_type : str
			Topography type. Either:

				* 'flat_terrain' (default);
				* 'gaussian';
				* 'schaer';
				* 'user_defined'.
		steady_profile : sympl.DataArray
			2-D :class:`sympl.DataArray` representing the steady-state height profile.
		profile : sympl.DataArray
			2-D :class:`sympl.DataArray` representing the current height profile.

		Keyword arguments
		-----------------
		time : timedelta
			The elapsed simulation time after which the topography should stop increasing.
			If not specified, a time-invariant terrain surface-height is assumed.
		max_height : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the maximum mountain height.
			Effective when :data:`topography_type` is either 'gaussian' or 'schaer'.
		center_x : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the :math:`x`-coordinate
			of the mountain center. By default, the mountain center is placed in
			the center of the domain. Effective when :data:`topography_type` is either
			'gaussian' or 'schaer'.
		center_y : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the :math:`y`-coordinate
			of the mountain center. By default, the mountain center is placed in
			the center of the domain. Effective when :data:`topography_type` is either
			'gaussian' or 'schaer'.
		width_x : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the mountain half-width
			in the :math:`x`-direction. Defaults to 1, in the same units of the
			:data:`x`-axis. Effective when :data:`topography_type` is either 'gaussian'
			or 'schaer'.
		width_y : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the mountain half-width
			in the :math:`y`-direction. Defaults to 1, in the same units of the
			:data:`y`-axis. Effective when :data:`topography_type` is either 'gaussian'
			or 'schaer'.
		expression : str
			Terrain profile expression in the independent variables :math:`x` and
			:math:`y`. Must be fully C++-compliant. Effective only when
			:data:`topography_type` is 'user_defined'.
		smooth : bool
			:obj:`True` to smooth the topography out, :obj:`False` otherwise.
			Defaults to :obj:`False`.

		Raises
		------
		ValueError :
			If the argument :obj:`topography_type` is neither 'flat_terrain',
			'gaussian', 'schaer', nor 'user_defined'.
		"""
        if topography_type not in [
            "flat_terrain",
            "gaussian",
            "schaer",
            "user_defined",
        ]:
            raise ValueError(
                "Unknown topography type. Supported types are: "
                "''flat_terrain'', ''gaussian'', ''schaer'', ''user_defined''."
            )

        self._type = topography_type
        self._steady_profile = steady_profile
        self._kwargs = deepcopy(kwargs)

        time = kwargs.get("time", Timedelta(seconds=0))
        self._fact = float(time.total_seconds() == 0.0)
        self._kwargs["time"] = time

        self._profile = profile if profile is not None else deepcopy(steady_profile)
        self._profile.values[...] = self._fact * steady_profile.values[...]

    @property
    def type(self):
        """
		Returns
		-------
		str :
			The topography type.
		"""
        return self._type

    @property
    def profile(self):
        """
		Returns
		-------
		sympl.DataArray :
			2-D :class:`sympl.DataArray` representing the current
			topography profile.
		"""
        return self._profile

    @property
    def steady_profile(self):
        """
		Returns
		-------
		sympl.DataArray :
			2-D :class:`sympl.DataArray` representing the steady-state
			topography profile.
		"""
        return self._steady_profile

    @property
    def kwargs(self):
        """
		Returns
		-------
		dict :
			The keyword arguments passed to the constructor.
		"""
        return self._kwargs

    def update(self, time):
        """
		Update topography at current simulation time.

		Parameters
		----------
		time : timedelta
			The elapsed simulation time.
		"""
        if lt(self._fact, 1.0):
            self._fact = min(time / self._kwargs["time"], 1.0)
            self._profile.values[...] = self._fact * self._steady_profile[...]


class PhysicalTopography(Topography):
    """
    Class which represents a possibly time-dependent topography.
	"""

    def __init__(self, grid, topography_type, **kwargs):
        """
		Parameters
		----------
		grid : `tasmania.HorizontalGrid`
			The underlying :class:`tasmania.HorizontalGrid`.
		topography_type : str
			Topography type. Either:
			
				* 'flat_terrain' (default);
				* 'gaussian';
				* 'schaer'; 
				* 'user_defined'.

		Keyword arguments
		-----------------
		time : timedelta
			The elapsed simulation time after which the topography should stop increasing.
			If not specified, a time-invariant terrain surface-height is assumed.
		max_height : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the maximum mountain height.
			Effective when :data:`topography_type` is either 'gaussian' or 'schaer'.
		center_x : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the :math:`x`-coordinate
			of the mountain center. By default, the mountain center is placed in
			the center of the domain. Effective when :data:`topography_type` is either
			'gaussian' or 'schaer'.
		center_y : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the :math:`y`-coordinate
			of the mountain center. By default, the mountain center is placed in
			the center of the domain. Effective when :data:`topography_type` is either
			'gaussian' or 'schaer'.
		width_x : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the mountain half-width
			in the :math:`x`-direction. Defaults to 1, in the same units of the
			:data:`x`-axis. Effective when :data:`topography_type` is either 'gaussian'
			or 'schaer'.
		width_y : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the mountain half-width
			in the :math:`y`-direction. Defaults to 1, in the same units of the
			:data:`y`-axis. Effective when :data:`topography_type` is either 'gaussian'
			or 'schaer'.
		expression : str
			Terrain profile expression in the independent variables :math:`x` and
			:math:`y`. Must be fully C++-compliant. Effective only when
			:data:`topography_type` is 'user_defined'.
		smooth : bool
			:obj:`True` to smooth the topography out, :obj:`False` otherwise.
			Defaults to :obj:`False`.

		Raises
		------
		ValueError :
			If the argument :obj:`topography_type` is neither 'flat_terrain',
			'gaussian', 'schaer', nor 'user_defined'.
		ImportError :
			If :class:`tasmania.cpp.parser.parser_2d.Parser2d` cannot be
			imported (likely because it has not been compiled).
		"""
        if topography_type not in [
            "flat_terrain",
            "gaussian",
            "schaer",
            "user_defined",
        ]:
            raise ValueError(
                "Unknown topography type. Supported types are: "
                "''flat_terrain'', ''gaussian'', ''schaer'', ''user_defined''."
            )

        topo_kwargs = deepcopy(kwargs)

        x, y = grid.x, grid.y
        xv, yv = grid.x.values, grid.y.values

        dtype = xv.dtype

        if topography_type == "flat_terrain":
            topo_steady = np.zeros((grid.nx, grid.ny), dtype=dtype)
        elif topography_type == "gaussian":
            max_height_ = kwargs.get(
                "max_height", DataArray(500.0, attrs={"units": "m"})
            )
            max_height = max_height_.to_units("m").values.item()

            width_x_ = kwargs.get(
                "width_x", DataArray(1.0, attrs={"units": x.attrs["units"]})
            )
            width_x = width_x_.to_units(x.attrs["units"]).values.item()

            width_y_ = kwargs.get(
                "width_y", DataArray(1.0, attrs={"units": y.attrs["units"]})
            )
            width_y = width_y_.to_units(y.attrs["units"]).values.item()

            cx = 0.5 * (xv[0] + xv[-1])
            center_x = (
                cx
                if kwargs.get("center_x") is None
                else kwargs["center_x"].to_units(x.attrs["units"]).values.item()
            )

            cy = 0.5 * (yv[0] + yv[-1])
            center_y = (
                cy
                if kwargs.get("center_y") is None
                else kwargs["center_y"].to_units(y.attrs["units"]).values.item()
            )

            topo_kwargs["max_height"] = DataArray(max_height, attrs={"units": "m"})
            topo_kwargs["width_x"] = DataArray(
                width_x, attrs={"units": x.attrs["units"]}
            )
            topo_kwargs["width_y"] = DataArray(
                width_y, attrs={"units": y.attrs["units"]}
            )
            topo_kwargs["center_x"] = DataArray(
                center_x, attrs={"units": x.attrs["units"]}
            )
            topo_kwargs["center_y"] = DataArray(
                center_y, attrs={"units": y.attrs["units"]}
            )

            xv_, yv_ = np.meshgrid(xv, yv, indexing="ij")
            topo_steady = max_height * np.exp(
                -((xv_ - center_x) / width_x) ** 2 - ((yv_ - center_y) / width_y) ** 2
            )
        elif topography_type == "schaer":
            max_height_ = kwargs.get(
                "max_height", DataArray(500.0, attrs={"units": "m"})
            )
            max_height = max_height_.to_units("m").values.item()

            width_x_ = kwargs.get(
                "width_x", DataArray(1.0, attrs={"units": x.attrs["units"]})
            )
            width_x = width_x_.to_units(x.attrs["units"]).values.item()

            width_y_ = kwargs.get(
                "width_y", DataArray(1.0, attrs={"units": y.attrs["units"]})
            )
            width_y = width_y_.to_units(y.attrs["units"]).values.item()

            cx = 0.5 * (xv[0] + xv[-1])
            center_x = (
                cx
                if kwargs.get("center_x") is None
                else kwargs["center_x"].to_units(x.attrs["units"]).values.item()
            )

            cy = 0.5 * (yv[0] + yv[-1])
            center_y = (
                cy
                if kwargs.get("center_y") is None
                else kwargs["center_y"].to_units(y.attrs["units"]).values.item()
            )

            topo_kwargs["max_height"] = DataArray(max_height, attrs={"units": "m"})
            topo_kwargs["width_x"] = DataArray(
                width_x, attrs={"units": x.attrs["units"]}
            )
            topo_kwargs["width_y"] = DataArray(
                width_y, attrs={"units": y.attrs["units"]}
            )
            topo_kwargs["center_x"] = DataArray(
                center_x, attrs={"units": x.attrs["units"]}
            )
            topo_kwargs["center_y"] = DataArray(
                center_y, attrs={"units": y.attrs["units"]}
            )

            xv_, yv_ = np.meshgrid(xv, yv, indexing="ij")
            topo_steady = max_height / (
                (
                    1
                    + ((xv_ - center_x) / width_x) ** 2
                    + ((yv_ - center_y) / width_y) ** 2
                )
                ** 1.5
            )
        else:
            expression = (
                "x + y" if kwargs.get("expression") is None else kwargs["expression"]
            )

            topo_kwargs["expression"] = expression

            # import the parser
            try:
                from tasmania.cpp.parser.parser_2d import Parser2d
            except ImportError:
                print("Hint: did you compile the parser?")
                raise

            # parse
            parser = Parser2d(
                expression.encode("UTF-8"),
                grid.x.to_units("m").values,
                grid.y.to_units("m").values,
            )
            topo_steady = parser.evaluate()

        # smooth the topography out
        topo_kwargs["smooth"] = kwargs.get("smooth", False)
        if topo_kwargs["smooth"]:
            topo_steady[1:-1, 1:-1] += 0.125 * (
                topo_steady[:-2, 1:-1]
                + topo_steady[2:, 1:-1]
                + topo_steady[1:-1, :-2]
                + topo_steady[1:-1, 2:]
                - 4.0 * topo_steady[1:-1, 1:-1]
            )

        topo_steady = DataArray(
            topo_steady,
            coords=[xv, yv],
            dims=[x.dims[0], y.dims[0]],
            attrs={"units": "m"},
        )

        super().__init__(topography_type, topo_steady, **topo_kwargs)


class NumericalTopography(Topography):
    """
	Class which represents a possibly time-dependent topography defined
	over a *numerical* grid.
	"""

    def __init__(self, grid, phys_topography, boundary):
        """
		Parameters
		----------
		grid : tasmania.NumericalHorizontalGrid
			The underlying numerical grid.
		phys_topography : tasmania.Topography
			The topography defined over the associated *physical* grid.
		boundary : tasmania.HorizontalBoundary
			The :class:`tasmania.HorizontalBoundary` handling the horizontal
			boundary conditions.
		"""
        topo_type = phys_topography.type
        topo_kwargs = phys_topography.kwargs

        ptopo = phys_topography.profile.values
        ptopo_steady = phys_topography.steady_profile.values
        units = phys_topography.profile.attrs["units"]

        ctopo = make_dataarray_2d(boundary.get_numerical_field(ptopo), grid, units)
        ctopo_steady = make_dataarray_2d(
            boundary.get_numerical_field(ptopo_steady), grid, units
        )

        super().__init__(topo_type, ctopo_steady, ctopo, **topo_kwargs)
