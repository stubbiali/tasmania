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
# >>> cupy
try:
    import cupy
except (ImportError, ModuleNotFoundError):
    cupy = None


# >>> dawn4py
try:
    import dawn4py
except (ImportError, ModuleNotFoundError):
    dawn4py = None

# >>> gt4py
try:
    import gt4py
except (ImportError, ModuleNotFoundError):
    gt4py = None

# >>> numba
try:
    import numba
except (ImportError, ModuleNotFoundError):
    numba = None

# >>> taichi
# try:
#     from contextlib import redirect_stdout
#
#     with open("/dev/null", "w") as f:
#         with redirect_stdout(f):
#             import taichi
# except (ImportError, ModuleNotFoundError):
#     taichi = None
taichi = None

# >>> extra
GPU_AVAILABLE = cupy is not None
