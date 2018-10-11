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
from datetime import timedelta
import gridtools as gt
import numpy as np
import tasmania as taz


grid, states = taz.load_netcdf_dataset('../tests/baseline_datasets/isentropic_dry.nc')
state = states[0]

pt = state['air_pressure_on_interface_levels'][0, 0, 0]
dv = taz.IsentropicDiagnostics(grid, moist_on=False, pt=pt, 
							   backend=gt.mode.NUMPY, dtype=np.float32)

pg = taz.NonconservativeIsentropicPressureGradient(grid, order=2, horizontal_boundary_type='relaxed',
							   					   backend=gt.mode.NUMPY, dtype=np.float32)

dt = timedelta(seconds=24)
grid.update_topography(dt)

diagnostics = dv(state)
state.update(diagnostics)

tendencies, _ = pg(state)
u_tnd = tendencies['x_velocity'].values

print('End.')
