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
from sympl import DataArray

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.horizontal_boundaries.utils import repeat_axis, shrink_axis
from tasmania.python.utils.framework_utils import register


class Periodic(HorizontalBoundary):
    """ Periodic boundary conditions. """

    def __init__(self, nx, ny, nb, gt_powered, backend, dtype):
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed ny/2."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(nx, ny, nb, gt_powered, backend=backend, dtype=dtype)

    @property
    def ni(self):
        return self.nx + 2 * self.nb

    @property
    def nj(self):
        return self.ny + 2 * self.nb

    def get_numerical_xaxis(self, paxis, dims=None):
        nb = self.nb
        pvalues = paxis.values
        cdims = dims if dims is not None else paxis.dims[0]
        mi, dtype = pvalues.shape[0], pvalues.dtype

        cvalues = np.zeros(mi + 2 * nb, dtype=dtype)
        cvalues[nb:-nb] = pvalues[...]
        cvalues[:nb] = np.array(
            [pvalues[0] - i * (pvalues[1] - pvalues[0]) for i in range(nb, 0, -1)],
            dtype=dtype,
        )
        cvalues[-nb:] = np.array(
            [pvalues[-1] + (i + 1) * (pvalues[1] - pvalues[0]) for i in range(nb)],
            dtype=dtype,
        )

        return DataArray(
            cvalues,
            coords=[cvalues],
            dims=cdims,
            name=paxis.name,
            attrs={"units": paxis.attrs["units"]},
        )

    def get_numerical_yaxis(self, paxis, dims=None):
        return self.get_numerical_xaxis(paxis, dims)

    def get_numerical_field(self, field, field_name=None):
        nx, ny, nb = self.nx, self.ny, self.nb
        dtype = field.dtype
        field_name = field_name or ""
        mx = (
            nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else nx
        )
        mi = mx + 2 * nb
        my = (
            ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else ny
        )

        try:
            li, lj, lk = field.shape
            cfield = np.zeros((li + 2 * nb, lj + 2 * nb, lk), dtype=dtype)
        except ValueError:
            li, lj = field.shape
            cfield = np.zeros((li + 2 * nb, lj + 2 * nb), dtype=dtype)

        cfield[nb : mx + nb, nb : my + nb] = np.asarray(field[:mx, :my])
        cfield[:nb, nb : my + nb] = cfield[nx - 1 : nx - 1 + nb, nb : my + nb]
        cfield[mx + nb : mx + 2 * nb, nb : my + nb] = (
            cfield[nb + 1 : 2 * nb + 1, nb : my + nb]
            if mx == nx
            else cfield[nb + 2 : 2 * nb + 2, nb : my + nb]
        )
        cfield[:mi, :nb] = cfield[:mi, ny - 1 : ny - 1 + nb]
        cfield[:mi, my + nb : my + 2 * nb] = (
            cfield[:mi, nb + 1 : 2 * nb + 1]
            if my == ny
            else cfield[:mi, nb + 2 : 2 * nb + 2]
        )

        return cfield

    def get_physical_xaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_yaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        nx, ny, nb = self.nx, self.ny, self.nb
        field_name = field_name or ""
        mx = (
            nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else nx
        )
        mi = mx + 2 * nb
        my = (
            ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else ny
        )

        field[:nb, nb : my + nb] = field[nx - 1 : nx - 1 + nb, nb : my + nb]
        field[mx + nb : mx + 2 * nb, nb : my + nb] = (
            field[nb + 1 : 2 * nb + 1, nb : my + nb]
            if mx == nx
            else field[nb + 2 : 2 * nb + 2, nb : my + nb]
        )
        field[:mi, :nb] = field[:mi, ny - 1 : ny - 1 + nb]
        field[:mi, my + nb : my + 2 * nb] = (
            field[:mi, nb + 1 : 2 * nb + 1]
            if my == ny
            else field[:mi, nb + 2 : 2 * nb + 2]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field[0, :] = field[-2, :]
        field[-1, :] = field[1, :]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field[:, 0] = field[:, -2]
        field[:, -1] = field[:, 1]


class Periodic1DX(HorizontalBoundary):
    """ Periodic boundary conditions for a physical grid with ``ny=1``. """

    def __init__(self, nx, ny, nb, gt_powered, backend, dtype):
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert ny == 1, "Number of grid points along second dimension must be 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed nx/2."

        super().__init__(nx, ny, nb, gt_powered, backend=backend, dtype=dtype)

    @property
    def ni(self):
        return self.nx + 2 * self.nb

    @property
    def nj(self):
        return 2 * self.nb + 1

    def get_numerical_xaxis(self, paxis, dims=None):
        nb = self.nb
        pvalues = paxis.values
        cdims = dims if dims is not None else paxis.dims[0]
        mi, dtype = pvalues.shape[0], pvalues.dtype

        cvalues = np.zeros(mi + 2 * nb, dtype=dtype)
        cvalues[nb:-nb] = pvalues[...]
        cvalues[:nb] = np.array(
            [pvalues[0] - i * (pvalues[1] - pvalues[0]) for i in range(nb, 0, -1)],
            dtype=dtype,
        )
        cvalues[-nb:] = np.array(
            [pvalues[-1] + (i + 1) * (pvalues[1] - pvalues[0]) for i in range(nb)],
            dtype=dtype,
        )

        return DataArray(
            cvalues,
            coords=[cvalues],
            dims=cdims,
            name=paxis.name,
            attrs={"units": paxis.attrs["units"]},
        )

    def get_numerical_yaxis(self, paxis, dims=None):
        return repeat_axis(paxis, self.nb, dims)

    def get_numerical_field(self, field, field_name=None):
        nx, nb = self.nx, self.nb
        dtype = field.dtype
        field_name = field_name or ""
        mx = (
            nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else nx
        )
        mi = mx + 2 * nb
        my = (
            2 if "at_v_locations" in field_name or "at_uv_locations" in field_name else 1
        )

        try:
            li, lj, lk = field.shape
            cfield = np.zeros((li + 2 * nb, lj + 2 * nb, lk), dtype=dtype)
        except ValueError:
            li, lj = field.shape
            cfield = np.zeros((li + 2 * nb, lj + 2 * nb), dtype=dtype)

        cfield[nb : mx + nb, nb : my + nb] = np.asarray(field[:mx, :my])
        cfield[:nb, nb : my + nb] = cfield[nx - 1 : nx - 1 + nb, nb : my + nb]
        cfield[mx + nb : mx + 2 * nb, nb : my + nb] = (
            cfield[nb + 1 : 2 * nb + 1, nb : my + nb]
            if mx == nx
            else cfield[nb + 2 : 2 * nb + 2, nb : my + nb]
        )
        cfield[:mi, :nb] = cfield[:mi, nb : nb + 1]
        cfield[:mi, my + nb : my + 2 * nb] = (
            cfield[:mi, nb : nb + 1] if my == 1 else cfield[:mi, nb + 1 : nb + 2]
        )

        return cfield

    def get_physical_xaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_yaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        nx, ny, nb = self.nx, self.ny, self.nb
        field_name = field_name or ""
        mx = (
            nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else nx
        )
        mi = mx + 2 * nb
        my = (
            2 if "at_v_locations" in field_name or "at_uv_locations" in field_name else 1
        )

        field[:nb, nb : my + nb] = field[nx - 1 : nx - 1 + nb, nb : my + nb]
        field[mx + nb : mx + 2 * nb, nb : my + nb] = (
            field[nb + 1 : 2 * nb + 1, nb : my + nb]
            if mx == nx
            else field[nb + 2 : 2 * nb + 2, nb : my + nb]
        )
        field[:mi, :nb] = field[:mi, nb : nb + 1]
        field[:mi, my + nb : my + 2 * nb] = (
            field[:mi, nb : nb + 1] if my == 1 else field[:mi, nb + 1 : nb + 2]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field[0, :] = field[-2, :]
        field[-1, :] = field[1, :]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field[:, 0] = field[:, -2]
        field[:, -1] = field[:, 1]


class Periodic1DY(HorizontalBoundary):
    """ Periodic boundary conditions for a physical grid with ``ny=1``. """

    def __init__(self, nx, ny, nb, gt_powered, backend, dtype):
        assert nx == 1, "Number of grid points along first dimension must be 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(nx, ny, nb, gt_powered, backend=backend, dtype=dtype)

    @property
    def ni(self):
        return 2 * self.nb + 1

    @property
    def nj(self):
        return self.ny + 2 * self.nb

    def get_numerical_xaxis(self, paxis, dims=None):
        return repeat_axis(paxis, self.nb, dims)

    def get_numerical_yaxis(self, paxis, dims=None):
        nb = self.nb
        pvalues = paxis.values
        cdims = dims if dims is not None else paxis.dims[0]
        mi, dtype = pvalues.shape[0], pvalues.dtype

        cvalues = np.zeros(mi + 2 * nb, dtype=dtype)
        cvalues[nb:-nb] = pvalues[...]
        cvalues[:nb] = np.array(
            [pvalues[0] - i * (pvalues[1] - pvalues[0]) for i in range(nb, 0, -1)],
            dtype=dtype,
        )
        cvalues[-nb:] = np.array(
            [pvalues[-1] + (i + 1) * (pvalues[1] - pvalues[0]) for i in range(nb)],
            dtype=dtype,
        )

        return DataArray(
            cvalues,
            coords=[cvalues],
            dims=cdims,
            name=paxis.name,
            attrs={"units": paxis.attrs["units"]},
        )

    def get_numerical_field(self, field, field_name=None):
        ny, nb = self.ny, self.nb
        dtype = field.dtype
        field_name = field_name or ""
        mx = (
            2 if "at_u_locations" in field_name or "at_uv_locations" in field_name else 1
        )
        my = (
            ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else ny
        )
        mj = my + 2 * nb

        try:
            li, lj, lk = field.shape
            cfield = np.zeros((li + 2 * nb, lj + 2 * nb, lk), dtype=dtype)
        except ValueError:
            li, lj = field.shape
            cfield = np.zeros((li + 2 * nb, lj + 2 * nb), dtype=dtype)

        cfield[nb : mx + nb, nb : my + nb] = np.asarray(field[:mx, :my])
        cfield[nb : mx + nb, :nb] = cfield[nb : mx + nb, ny - 1 : ny - 1 + nb]
        cfield[nb : mx + nb, my + nb : my + 2 * nb] = (
            cfield[nb : mx + nb, nb + 1 : 2 * nb + 1]
            if my == ny
            else cfield[nb : mx + nb, nb + 2 : 2 * nb + 2]
        )
        cfield[:nb, :mj] = cfield[nb : nb + 1, :mj]
        cfield[mx + nb : mx + 2 * nb, :mj] = (
            cfield[nb : nb + 1, :mj] if mx == 1 else cfield[nb + 1 : nb + 2, :mj]
        )

        return cfield

    def get_physical_xaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_yaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        ny, nb = self.ny, self.nb
        field_name = field_name or ""
        mx = (
            2 if "at_u_locations" in field_name or "at_uv_locations" in field_name else 1
        )
        my = (
            ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else ny
        )
        mj = my + 2 * nb

        field[nb : mx + nb, :nb] = field[nb : mx + nb, ny - 1 : ny - 1 + nb]
        field[nb : mx + nb, my + nb : my + 2 * nb] = (
            field[nb : mx + nb, nb + 1 : 2 * nb + 1]
            if my == ny
            else field[nb : mx + nb, nb + 2 : 2 * nb + 2]
        )
        field[:nb, :mj] = field[nb : nb + 1, :mj]
        field[mx + nb : mx + 2 * nb, :mj] = (
            field[nb : nb + 1, :mj] if mx == 1 else field[nb + 1 : nb + 2, :mj]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field[0, :] = field[-2, :]
        field[-1, :] = field[1, :]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field[:, 0] = field[:, -2]
        field[:, -1] = field[:, 1]


@register(name="periodic", registry_class=HorizontalBoundary)
def dispatch(
    nx,
    ny,
    nb,
    gt_powered,
    backend="numpy",
    backend_opts=None,
    build_info=None,
    dtype=np.float64,
    exec_info=None,
    default_origin=None,
    rebuild=False,
    storage_shape=None,
    managed_memory=False,
):
    """ Dispatch based on the grid size. """
    if nx == 1:
        return Periodic1DY(1, ny, nb, gt_powered, backend, dtype)
    elif ny == 1:
        return Periodic1DX(nx, 1, nb, gt_powered, backend, dtype)
    else:
        return Periodic(nx, ny, nb, gt_powered, backend, dtype)
