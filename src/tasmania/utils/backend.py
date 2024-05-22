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
def is_gt(backend: str):
    return len(backend) > 6 and backend[:6] == "gt4py:"


def get_gt_backend(backend):
    assert is_gt(backend), f"{backend} is not a GT4Py backend."
    return backend[6:]


def is_ti(backend):
    return len(backend) > 7 and backend[:7] == "taichi:"


def get_ti_arch(backend):
    assert is_ti(backend), f"{backend} is not a Taichi backend."
    return backend[7:]
