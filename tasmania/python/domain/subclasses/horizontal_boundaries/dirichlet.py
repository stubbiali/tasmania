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
import inspect
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
from tasmania.python.utils.storage_utils import get_asarray_function


def placeholder(time, grid, slice_x, slice_y, field_name, field_units):
    pass


class Dirichlet(HorizontalBoundary):
    """ Dirichlet boundary conditions. """

    def __init__(self, nx, ny, nb, backend, dtype, core=placeholder):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the physical grid
            along the first dimension.
        ny : int
            Number of points featured by the physical grid
            along the second dimension.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        dtype : data-type
            The data type of the storages.
        core : `callable`, optional
            Callable object providing the boundary layers values.
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed ny/2."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        signature = inspect.signature(core)
        error_msg = (
            "The signature of the core function should be "
            "core(time, grid, slice_x=None, slice_y=None, field_name=None, field_units=None)"
        )
        assert tuple(signature.parameters.keys())[0] == "time", error_msg
        assert tuple(signature.parameters.keys())[1] == "grid", error_msg
        assert "slice_x" in signature.parameters, error_msg
        assert "slice_y" in signature.parameters, error_msg
        assert "field_name" in signature.parameters, error_msg
        assert "field_units" in signature.parameters, error_msg

        super().__init__(nx, ny, nb, backend=backend, dtype=dtype)

        self._kwargs["core"] = core

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
        nb, core = self.nb, self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ny
        )

        asarray = get_asarray_function(self._backend or "numpy")

        field[:nb, :mj] = asarray(
            core(
                time, grid, slice(0, nb), slice(0, mj), field_name, field_units
            )
        )
        field[mi - nb : mi, :mj] = asarray(
            core(
                time,
                grid,
                slice(mi - nb, mi),
                slice(0, mj),
                field_name,
                field_units,
            )
        )
        field[nb : mi - nb, :nb] = asarray(
            core(
                time,
                grid,
                slice(nb, mi - nb),
                slice(0, nb),
                field_name,
                field_units,
            )
        )
        field[nb : mi - nb, mj - nb : mj] = asarray(
            core(
                time,
                grid,
                slice(nb, mi - nb),
                slice(mj - nb, mj),
                field_name,
                field_units,
            )
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ny
        )

        asarray = get_asarray_function(self._backend or "numpy")

        field[:1, :mj] = asarray(
            core(
                time, grid, slice(0, 1), slice(0, mj), field_name, field_units
            )
        )
        field[mi - 1 : mi, :mj] = asarray(
            core(
                time,
                grid,
                slice(mi - 1, mi),
                slice(0, mj),
                field_name,
                field_units,
            )
        )

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ny
        )

        asarray = get_asarray_function(self._backend or "numpy")

        field[:mi, :1] = asarray(
            core(
                time, grid, slice(0, mi), slice(0, 1), field_name, field_units
            )
        )
        field[:mi, mj - 1 : mj] = asarray(
            core(
                time,
                grid,
                slice(0, mi),
                slice(mj - 1, mj),
                field_name,
                field_units,
            )
        )


class Dirichlet1DX(HorizontalBoundary):
    """ Dirichlet boundary conditions for a physical grid with ``ny=1``. """

    def __init__(self, nx, ny, nb, backend, dtype, core=placeholder):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the physical grid
            along the first dimension.
        ny : int
            Number of points featured by the physical grid
            along the second dimension. It must be 1.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        dtype : data-type
            The data type of the storages.
        core : `callable`, optional
            Callable object providing the boundary layers values.
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert (
            ny == 1
        ), "Number of grid points along second dimension must be 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed nx/2."

        signature = inspect.signature(core)
        error_msg = (
            "The signature of the core function should be "
            "core(time, grid, slice_x=None, slice_y=None, field_name=None, field_units=None)"
        )
        assert tuple(signature.parameters.keys())[0] == "time", error_msg
        assert tuple(signature.parameters.keys())[1] == "grid", error_msg
        assert "slice_x" in signature.parameters, error_msg
        assert "slice_y" in signature.parameters, error_msg
        assert "field_name" in signature.parameters, error_msg
        assert "field_units" in signature.parameters, error_msg

        super().__init__(nx, ny, nb, backend=backend, dtype=dtype)

        self._kwargs["core"] = core

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
        nb, core = self.nb, self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )

        asarray = get_asarray_function(self._backend or "numpy")

        field[:nb, nb : mj - nb] = asarray(
            core(
                time,
                grid,
                slice(0, nb),
                slice(nb, mj - nb),
                field_name,
                field_units,
            )
        )
        field[mi - nb : mi, nb : mj - nb] = asarray(
            core(
                time,
                grid,
                slice(mi - nb, mi),
                slice(nb, mj - nb),
                field_name,
                field_units,
            )
        )

        field[:mi, :nb] = field[:mi, nb : nb + 1]
        field[:mi, mj - nb : mj] = field[:mi, mj - nb - 1 : mj - nb]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )

        asarray = get_asarray_function(self._backend or "numpy")

        field[:1, :mj] = asarray(
            core(
                time, grid, slice(0, 1), slice(0, mj), field_name, field_units
            )
        )
        field[mi - 1 : mi, :mj] = asarray(
            core(
                time,
                grid,
                slice(mi - 1, mi),
                slice(0, mj),
                field_name,
                field_units,
            )
        )

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )

        asarray = get_asarray_function(self._backend or "numpy")

        field[:mi, :1] = asarray(
            core(
                time, grid, slice(0, mi), slice(0, 1), field_name, field_units
            )
        )
        field[:mi, mj - 1 : mj] = asarray(
            core(
                time,
                grid,
                slice(0, mi),
                slice(mj - 1, mj),
                field_name,
                field_units,
            )
        )


class Dirichlet1DY(HorizontalBoundary):
    """ Dirichlet boundary conditions for a physical grid with ``nx=1``. """

    def __init__(self, nx, ny, nb, backend, dtype, core=placeholder):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the physical grid
            along the first dimension. It must be 1.
        ny : int
            Number of points featured by the physical grid
            along the second dimension.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        dtype : data-type
            The data type of the storages.
        core : `callable`, optional
            Callable object providing the boundary layers values.
        """
        assert (
            nx == 1
        ), "Number of grid points along first dimension must be 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        signature = inspect.signature(core)
        error_msg = (
            "The signature of the core function should be "
            "core(time, grid, slice_x=None, slice_y=None, field_name=None, field_units=None)"
        )
        assert tuple(signature.parameters.keys())[0] == "time", error_msg
        assert tuple(signature.parameters.keys())[1] == "grid", error_msg
        assert "slice_x" in signature.parameters, error_msg
        assert "slice_y" in signature.parameters, error_msg
        assert "field_name" in signature.parameters, error_msg
        assert "field_units" in signature.parameters, error_msg

        super().__init__(nx, ny, nb, backend=backend, dtype=dtype)

        self._kwargs["core"] = core

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
        return np.asarray(field[self.nb : -self.nb, :])

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        nb, core = self.nb, self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ny
        )

        asarray = get_asarray_function(self._backend or "numpy")

        field[nb : mi - nb, :nb] = asarray(
            core(
                time,
                grid,
                slice(nb, mi - nb),
                slice(0, nb),
                field_name,
                field_units,
            )
        )
        field[nb : mi - nb, mj - nb : mj] = asarray(
            core(
                time,
                grid,
                slice(nb, mi - nb),
                slice(mj - nb, mj),
                field_name,
                field_units,
            )
        )

        field[:nb, :mj] = field[nb : nb + 1, :mj]
        field[mi - nb : mi, :mj] = field[mi - nb - 1 : mi - nb, :mj]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ny
        )

        asarray = get_asarray_function(self._backend or "numpy")

        field[:1, :mj] = asarray(
            core(
                time, grid, slice(0, 1), slice(0, mj), field_name, field_units
            )
        )
        field[mi - 1 : mi, :mj] = asarray(
            core(
                time,
                grid,
                slice(mi - 1, mi),
                slice(0, mj),
                field_name,
                field_units,
            )
        )

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ny
        )

        asarray = get_asarray_function(self._backend or "numpy")

        field[:mi, :1] = asarray(
            core(
                time, grid, slice(0, mi), slice(0, 1), field_name, field_units
            )
        )
        field[:mi, mj - 1 : mj] = asarray(
            core(
                time,
                grid,
                slice(0, mi),
                slice(mj - 1, mj),
                field_name,
                field_units,
            )
        )


@register(name="dirichlet", registry_class=HorizontalBoundary)
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
    core=placeholder,
):
    """ Dispatch based on the grid size. """
    if nx == 1:
        return Dirichlet1DY(1, ny, nb, backend, dtype, core)
    elif ny == 1:
        return Dirichlet1DX(nx, 1, nb, backend, dtype, core)
    else:
        return Dirichlet(nx, ny, nb, backend, dtype, core)
