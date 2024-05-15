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
import numpy as np
from sympl import DataArray

from tasmania.python.domain.topography import PhysicalTopography
from tasmania.python.framework.register import register


@register(name="schaer")
class Schaer(PhysicalTopography):
    """A modified Gaussian mountain proposed by Schaer and Durran (1997).

    Let :math:`h_s = h_s(x,y)` be the topography. Then

    .. math::
        h_s(x,y) = \\frac{h_{max}}{\\left[ 1 + \\left( \\frac{x - c_x}{\\sigma_x}
        \\right)^2 + \\left( \\frac{y - c_y}{\\sigma_y} \\right)^2 \\right]^{3/2}}.

    Reference
    ---------
    Schaer, C., and D. R. Durran. (1997). Vortex formation and vortex shedding \
    in continuously stratified flows past isolated topography. \
    *Journal of Atmospheric Sciences*, *54*:534-554.
    """

    def __init__(
        self,
        grid,
        time,
        smooth,
        *,
        max_height=None,
        center_x=None,
        center_y=None,
        width_x=None,
        width_y=None,
        **kwargs
    ) -> None:
        """
        Parameters
        ----------
        grid : tasmania.PhysicalHorizontalGrid
            The underlying :class:`tasmania.PhysicalHorizontalGrid`.
        time : datetime.timedelta
            The elapsed simulation time after which the topography should stop
            increasing. If not specified, a time-invariant terrain surface-height
            is assumed.
        smooth : bool
            ``True`` to smooth the topography out, ``False`` otherwise.
        max_height : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing the maximum mountain
            height :math:`h_{max}`.
        center_x : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing the :math:`x`-coordinate
            of the mountain center :math:`c_x`. By default, the mountain center
            coincides with the center of the domain.
        center_y : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing the :math:`y`-coordinate
            of the mountain center :math:`c_y`. By default, the mountain center
            coincides with the center of the domain.
        width_x : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing the mountain half-width
            in the :math:`x`-direction :math:`\sigma_x`. Defaults to 1, in the
            same units of the `x`-axis.
        width_y : `sympl.DataArray`, optional
            1-item :class:`~sympl.DataArray` representing the mountain half-width
            in the :math:`y`-direction :math:`\sigma_y`. Defaults to 1, in the
            same units of the `y`-axis.
        **kwargs :
            Catch-all unused keyword arguments.
        """
        super().__init__(
            grid,
            time,
            smooth,
            max_height=max_height,
            center_x=center_x,
            center_y=center_y,
            width_x=width_x,
            width_y=width_y,
        )

    def compute_steady_profile(self, grid, **kwargs):
        x, y = grid.x, grid.y
        xv, yv = grid.x.values, grid.y.values
        dtype = grid.x.dtype

        max_height = kwargs.get("max_height", None)
        max_height = max_height or DataArray(500.0, attrs={"units": "m"})
        hmax = max_height.to_units("m").values.item()

        width_x = kwargs.get("width_x", None)
        width_x = width_x or DataArray(1.0, attrs={"units": x.attrs["units"]})
        wx = width_x.to_units(x.attrs["units"]).values.item()

        width_y = kwargs.get("width_y", None)
        width_y = width_y or DataArray(1.0, attrs={"units": y.attrs["units"]})
        wy = width_y.to_units(y.attrs["units"]).values.item()

        cx = 0.5 * (xv[0] + xv[-1])
        cx = (
            cx
            if kwargs.get("center_x", None) is None
            else kwargs["center_x"].to_units(x.attrs["units"]).values.item()
        )

        cy = 0.5 * (yv[0] + yv[-1])
        cy = (
            cy
            if kwargs.get("center_y", None) is None
            else kwargs["center_y"].to_units(y.attrs["units"]).values.item()
        )

        xx, yy = np.meshgrid(xv, yv, indexing="ij")
        topo_steady = np.zeros_like(xx, dtype=dtype)
        topo_steady[...] = hmax / (
            (1 + ((xx - cx) / wx) ** 2 + ((yy - cy) / wy) ** 2) ** 1.5
        )

        return topo_steady
