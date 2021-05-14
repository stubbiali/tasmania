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
from typing import Sequence


class ConcurrentCouplingStaticOperator:
    @staticmethod
    def get_overwrite_tendencies(parent):
        components = getattr(parent, "component_list", [])
        tendencies = set()
        out = []
        for component in components:
            if hasattr(component, "tendency_properties"):
                ot = {}
                for name in component.input_tendency_properties:
                    ot[name] = name not in tendencies
                    tendencies.add(name)


def merge_dims(dim1: Sequence[str], dim2: Sequence[str]) -> Sequence[str]:
    if "*" not in dim1 and "*" not in dim2:
        return dim1
    elif "*" in dim1 and "*" not in dim2:
        return dim2
    elif "*" not in dim1 and "*" in dim2:
        return dim1
    else:
        return tuple(set(dim1).union(set(dim2)))