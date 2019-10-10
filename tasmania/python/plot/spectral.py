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
	CDF(Drawer)
"""
import numpy as np

from tasmania.python.plot.drawer import Drawer
from tasmania.python.plot.retrievers import DataRetriever
from tasmania.python.plot.plot_utils import make_cdf


class CDF(Drawer):
    """
	Drawer which computes and visualizes a cumulative distribution function (CDF)
	of a state quantity based on its grid point values at multiple time steps.
	"""

    def __init__(
        self, grid, field_name, field_units, x=None, y=None, z=None, properties=None
    ):
        """
		Parameters
		----------
		grid : tasmania.Grid
			The underlying grid.
		field_name : str
			The state quantity for which the CDF should be computed and visualized.
		field_units : str
			The units for the quantity to visualize.
		x : `slice`, optional
			The slice of indices to be selected along the first array
			dimension. If not given, all indices along the first array
			dimension will be considered.
		y : `slice`, optional
			The slice of indices to be selected along the second array
			dimension. If not given, all indices along the second array
			dimension will be considered.
		z : `slice`, optional
			The slice of indices to be selected along the third array
			dimension. If not given, all indices along the third array
			dimension will be considered.
		properties : `dict`, optional
			Dictionary whose keys are strings denoting plot-specific
			settings, and whose values specify values for those settings.
			See :func:`tasmania.python.plot.utils.make_cdf`.
		"""
        super().__init__(properties)
        self._retriever = DataRetriever(grid, field_name, field_units, x, y, z)
        self._data = None

    def reset(self):
        self._data = None

    def __call__(self, state, fig=None, ax=None):
        """
		Call operator computing and visualizing the CDF.
		"""
        if self._data is None:
            self._data = self._retriever(state)
        else:
            self._data = np.concatenate((self._data, self._retriever(state)), axis=2)

        if ax is not None:
            make_cdf(self._data, ax, **self.properties)
