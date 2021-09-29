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
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary


class ZhaoHorizontalBoundary(HorizontalBoundary):
    def __init__(self, grid, nb, init_time, solution_factory):
        super().__init__(grid, nb, init_time)
        self._solution_factory = solution_factory

    @property
    def mi(self):
        return self._grid.nx

    @property
    def mj(self):
        return self._grid.ny

    def from_physical_to_computational_domain(self, phi):
        return phi

    def from_computational_to_physical_domain(
        self, phi_, out_dims=None, change_sign=None
    ):
        return phi_

    def enforce(self, phi_new, phi_now, field_name, time):
        nb = self.nb
        g = self._grid

        t = time - self._init_time

        phi_new[:nb, :, :] = self._solution_factory(
            g, t, slice_x=slice(0, nb), slice_y=None, field_name=field_name
        )
        phi_new[-nb:, :, :] = self._solution_factory(
            g,
            t,
            slice_x=slice(g.nx - nb, g.nx),
            slice_y=None,
            field_name=field_name,
        )
        phi_new[:, :nb, :] = self._solution_factory(
            g, t, slice_x=None, slice_y=slice(0, nb), field_name=field_name
        )
        phi_new[:, -nb:, :] = self._solution_factory(
            g,
            t,
            slice_x=None,
            slice_y=slice(g.ny - nb, g.ny),
            field_name=field_name,
        )

    def set_outermost_layers_x(self, phi_new, phi_now):
        pass

    def set_outermost_layers_y(self, phi_new, phi_now):
        pass

    def get_computational_grid(self):
        return self._grid
