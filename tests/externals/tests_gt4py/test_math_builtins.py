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

from gt4py import gtscript

from tasmania.python.framework.allocators import as_storage, zeros


def exponential_defs(
    in_phi: gtscript.Field[float], out_phi: gtscript.Field[float]
):
    with computation(PARALLEL), interval(...):
        out_phi = exp(in_phi)


if __name__ == "__main__":
    exponential = gtscript.stencil("numpy", exponential_defs)

    in_phi_np = np.random.rand(10, 10, 10)
    in_phi = as_storage("numpy", data=in_phi_np)
    out_phi = zeros("numpy", shape=(10, 10, 10))

    out_phi_np = np.exp(in_phi_np)
    exponential(in_phi, out_phi, origin=(0, 0, 0), domain=(10, 10, 10))

    assert np.all(np.equal(out_phi, out_phi_np))
