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
try:
    import tasmania

    print("\n\u2705 HOORAY! tasmania has been installed successfully.\n")
except ImportError:
    print("\n\u274C Sorry! tasmania is not available.\n")

try:
    import gt4py

    print("\u2705 HOORAY! gt4py has been installed successfully.\n")
except ImportError:
    print("\u274C Sorry! gt4py is not available.\n")

try:
    import dawn4py

    print("\u2705 HOORAY! dawn4py has been installed successfully.\n")
except ImportError:
    print("\u274C Sorry! dawn4py is not available.\n")

try:
    import cupy

    print("\u2705 HOORAY! cupy has been installed successfully.\n")
except ImportError:
    print("\u274C Sorry! cupy is not available.\n")
