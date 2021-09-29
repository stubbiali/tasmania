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
import numpy as np
import tasmania as taz


fname1 = "../../data/test/burgers_fc_gt4py:gtx86_protocol.nc"
fname2 = "../../data/test/burgers_fc_gt4py:gtx86.nc"


def main():
    _, _, states1 = taz.load_netcdf_dataset(fname1)
    _, _, states2 = taz.load_netcdf_dataset(fname2)

    state1 = states1[-1]
    state2 = states2[-1]
    shared_keys = [key for key in state1 if key in state2 and key != "time"]
    for key in shared_keys:
        field1 = state1[key].data
        field2 = state2[key].data
        np.testing.assert_allclose(field1, field2)


if __name__ == "__main__":
    main()
