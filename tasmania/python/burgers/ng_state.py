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

import gridtools as gt
from tasmania.python.burgers.state import ZhaoSolutionFactory
from tasmania.python.utils.data_utils import make_dataarray_3d
from tasmania.python.utils.storage_utils import get_storage_descriptor


class NGZhaoStateFactory:
    """
	A class generating valid states for the Zhao test case.
	"""

    def __init__(self, initial_time, eps):
        """
		Parameters
		----------
		initial_time : datetime
			The initial time of the simulation.
		eps : sympl.DataArray
			1-item :class:`sympl.DataArray` representing the diffusivity.
			The units should be compatible with 'm s^-2'.
		"""
        self._solution_factory = ZhaoSolutionFactory(initial_time, eps)

    def __call__(self, time, grid, *, backend, dtype, halo):
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
        storage_shape = (grid.nx, grid.ny, grid.nz)
        descriptor = get_storage_descriptor(storage_shape, dtype, halo=halo)

        u_st = gt.storage.from_array(
            self._solution_factory(time, grid, field_name="x_velocity"),
            descriptor=descriptor,
            backend=backend,
        )
        u = make_dataarray_3d(u_st.data, grid, "m s^-1", "x_velocity")
        u.attrs["gt_storage"] = u_st

        v_st = gt.storage.from_array(
            self._solution_factory(time, grid, field_name="y_velocity"),
            descriptor=descriptor,
            backend=backend,
        )
        v = make_dataarray_3d(v_st.data, grid, "m s^-1", "y_velocity")
        v.attrs["gt_storage"] = v_st

        return {"time": time, "x_velocity": u, "y_velocity": v}
