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
import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.subclasses.horizontal_boundaries.utils import (
    repeat_axis,
    shrink_axis,
)
from tasmania.python.utils.framework_utils import register


class Identity(HorizontalBoundary):
    """ *Identity* boundary conditions. """

    def __init__(self, nx, ny, nb, backend, dtype):
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed ny/2."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(nx, ny, nb, backend=backend, dtype=dtype)

    @property
    def ni(self):
        return self.nx

    @property
    def nj(self):
        return self.ny

    def get_numerical_xaxis(self, paxis, dims=None):
        return paxis

    def get_numerical_yaxis(self, paxis, dims=None):
        return paxis

    def get_numerical_field(self, field, field_name=None):
        return np.asarray(field)

    def get_physical_xaxis(self, caxis, dims=None):
        return caxis

    def get_physical_yaxis(self, caxis, dims=None):
        return caxis

    def get_physical_field(self, field, field_name=None):
        return np.asarray(field)

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass


class Identity1DX(HorizontalBoundary):
    """ *Identity* boundary conditions for a physical grid with ``ny=1``. """

    def __init__(self, nx, ny, nb, backend, dtype):
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert (
            ny == 1
        ), "Number of grid points along second dimension must be 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed nx/2."

        super().__init__(nx, ny, nb, backend=backend, dtype=dtype)

    @property
    def ni(self):
        return self.nx

    @property
    def nj(self):
        return 2 * self.nb + 1

    def get_numerical_xaxis(self, paxis, dims=None):
        return paxis

    def get_numerical_yaxis(self, paxis, dims=None):
        return repeat_axis(paxis, self.nb, dims)

    def get_numerical_field(self, field, field_name=None):
        try:
            li, lj, lk = field.shape
            cfield = np.zeros((li, lj + 2 * self.nb, lk), dtype=self._dtype)
        except ValueError:
            li, lj = field.shape
            cfield = np.zeros((li, lj + 2 * self.nb), dtype=self._dtype)

        cfield[:, : self.nb + 1] = field[:, :1]
        cfield[:, self.nb + 1 :] = field[:, -1:]

        return cfield

    def get_physical_xaxis(self, caxis, dims=None):
        return caxis

    def get_physical_yaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_field(self, field, field_name=None):
        return np.asarray(field[:, self.nb : -self.nb])

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        ny, nb = self.ny, self.nb
        field_name = field_name or ""
        my = (
            2
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else 1
        )
        field[:, :nb] = field[:, nb : nb + 1]
        field[:, my + nb : my + 2 * nb] = (
            field[:, nb : nb + 1] if my == 1 else field[:, nb + 1 : nb + 2]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass


class Identity1DY(HorizontalBoundary):
    """ *Identity* boundary conditions for a physical grid with ``nx=1``. """

    def __init__(self, nx, ny, nb, backend, dtype):
        assert (
            nx == 1
        ), "Number of grid points along first dimension must be 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(nx, ny, nb, backend=backend, dtype=dtype)

    @property
    def ni(self):
        return 2 * self.nb + 1

    @property
    def nj(self):
        return self.ny

    def get_numerical_xaxis(self, paxis, dims=None):
        return repeat_axis(paxis, self.nb, dims)

    def get_numerical_yaxis(self, paxis, dims=None):
        return paxis

    def get_numerical_field(self, field, field_name=None):
        try:
            li, lj, lk = field.shape
            cfield = np.zeros((li + 2 * self.nb, lj, lk), dtype=self._dtype)
        except ValueError:
            li, lj = field.shape
            cfield = np.zeros((li + 2 * self.nb, lj), dtype=self._dtype)

        cfield[: self.nb + 1, :] = field[:1, :]
        cfield[self.nb + 1 :, :] = field[-1:, :]

        return cfield

    def get_physical_xaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_yaxis(self, caxis, dims=None):
        return caxis

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, :]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        nx, nb = self.nx, self.nb
        field_name = field_name or ""
        mx = (
            2
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else 1
        )
        field[:nb, :] = field[nb : nb + 1, :]
        field[mx + nb : mx + 2 * nb, :] = (
            field[nb : nb + 1, :] if mx == 1 else field[nb + 1 : nb + 2, :]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass


@register(name="identity", registry_class=HorizontalBoundary)
def dispatch(
    nx,
    ny,
    nb,
    backend="numpy",
    backend_opts=None,
    dtype=np.float64,
    build_info=None,
    exec_info=None,
    default_origin=None,
    rebuild=False,
    storage_shape=None,
    managed_memory=False,
):
    """ Dispatch based on the grid size. """
    if nx == 1:
        return Identity1DY(1, ny, nb, backend, dtype)
    elif ny == 1:
        return Identity1DX(nx, 1, nb, backend, dtype)
    else:
        return Identity(nx, ny, nb, backend, dtype)
