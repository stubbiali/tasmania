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

from tasmania.python.domain.topography import PhysicalTopography, registry


@registry(name="user_defined")
class UserDefined(PhysicalTopography):
    """ User-defined terrain profile.

    The analytical expression of the profile is passed to the class as a string,
    which is then parsed in C++ via `Cython <http://cython.org>`_ . Therefore
    the string must be fully C++-compliant.
    """

    def __init__(self, grid, time, smooth, *, expression=None) -> None:
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
        expression : `str`, optional
            Analytical expression of the terrain profile in the independent
            variables :math:`x` and :math:`y`. Must be fully C++-compliant.
        """
        super().__init__(grid, time, smooth, expression=expression)

    def compute_steady_profile(self, grid, **kwargs):
        expression = (
            "x + y" if kwargs.get("expression") is None else kwargs["expression"]
        )

        # import the parser
        try:
            from tasmania.cpp.parser.parser_2d import Parser2d
        except ImportError:
            print("Hint: did you compile the parser?")
            raise

        # parse
        parser = Parser2d(expression.encode("UTF-8"), grid.x.values, grid.y.values)
        topo_steady = np.zeros((grid.nx, grid.ny), dtype=grid.x.dtype)
        topo_steady[...] = parser.evaluate()

        return topo_steady
