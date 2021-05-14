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
from typing import TYPE_CHECKING

from gt4py import gtscript

from tasmania.python.framework.stencil import stencil_subroutine

if TYPE_CHECKING:
    import numpy as np

    from tasmania.python.utils.typingx import GTField


@stencil_subroutine.register(backend=("numpy", "cupy"), stencil="set_output")
def set_output_numpy(
    lhs: "np.ndarray", rhs: "np.ndarray", overwrite: bool
) -> None:
    lhs[...] = rhs if overwrite else lhs + rhs


@stencil_subroutine.register(backend="gt4py*", stencil="set_output")
@gtscript.function
def set_output_gt4py(
    lhs: "GTField", rhs: "GTField", overwrite: bool
) -> "GTField":
    return rhs if overwrite else lhs + rhs