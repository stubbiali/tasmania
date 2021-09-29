# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from sympl import DataArray

from tests import utilities


if __name__ == "__main__":
    for _ in range(100):
        out = utilities.st_floats().example()
        assert not np.isnan(out)
        assert not np.isinf(out)

        utilities.st_interval(axis_name="x").example()
        utilities.st_interval(axis_name="y").example()
        utilities.st_interval(axis_name="z").example()

        utilities.st_length(axis_name="x").example()
        utilities.st_length(axis_name="y").example()
        utilities.st_length(axis_name="z").example()

        x = DataArray([0, 100], attrs={"units": "m"})
        time = timedelta(seconds=1800)
        utilities.st_topography_kwargs(x).example()
        utilities.st_topography_kwargs(x, topo_time=time).example()

        y = DataArray([0, 100], attrs={"units": "m"})
        utilities.st_topography_kwargs(x, y).example()
        utilities.st_topography_kwargs(x, y, topo_time=time).example()

        domain = utilities.st_domain()
        g = domain.physical_grid
        utilities.st_isentropic_state(g)
        utilities.st_isentropic_state(g, moist=True)
        utilities.st_isentropic_state(g, moist=True, precipitation=True)
