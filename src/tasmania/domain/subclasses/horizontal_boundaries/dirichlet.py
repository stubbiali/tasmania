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
import inspect
import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.subclasses.horizontal_boundaries.utils import (
    change_dims,
    repeat_axis,
)
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.register import register


def placeholder(time, grid, slice_x, slice_y, field_name, field_units):
    pass


class Dirichlet(HorizontalBoundary):
    """Dirichlet boundary conditions."""

    def __init__(self, grid, nb, backend, storage_options, core=placeholder):
        """
        Parameters
        ----------
        grid : Grid
            The underlying three-dimensional physical grid.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        storage_options : StorageOptions
            Storage-related options.
        core : `callable`, optional
            Callable object providing the boundary layers values.
        """
        nx, ny = grid.nx, grid.ny
        assert nx > 1, "Number of grid points along first dimension should be larger " "than 1."
        assert ny > 1, "Number of grid points along second dimension should be larger " "than 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed ny/2."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        signature = inspect.signature(core)
        error_msg = (
            "The signature of the core function should be "
            "core(time, grid, slice_x=None, slice_y=None, field_name=None, "
            "field_units=None)"
        )
        assert tuple(signature.parameters.keys())[0] == "time", error_msg
        assert tuple(signature.parameters.keys())[1] == "grid", error_msg
        assert "slice_x" in signature.parameters, error_msg
        assert "slice_y" in signature.parameters, error_msg
        assert "field_name" in signature.parameters, error_msg
        assert "field_units" in signature.parameters, error_msg

        super().__init__(grid, nb, backend=backend, storage_options=storage_options)

        self._kwargs["core"] = core

    @property
    def ni(self):
        return self.nx

    @property
    def nj(self):
        return self.ny

    def get_numerical_xaxis(self, dims=None):
        return change_dims(self.physical_grid.x, dims)

    def get_numerical_xaxis_staggered(self, dims=None):
        return change_dims(self.physical_grid.x_at_u_locations, dims)

    def get_numerical_yaxis(self, dims=None):
        return change_dims(self.physical_grid.y, dims)

    def get_numerical_yaxis_staggered(self, dims=None):
        return change_dims(self.physical_grid.y_at_v_locations, dims)

    def get_numerical_field(self, field, field_name=None):
        return field

    def get_physical_field(self, field, field_name=None):
        return field

    def enforce_field(self, field, field_name=None, field_units=None, time=None):
        nb, core = self.nb, self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.ny
        )

        ngrid = self.numerical_grid
        field[:nb, :mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(0, nb),
                slice(0, mj),
                field_name,
                field_units,
            )
        )
        field[mi - nb : mi, :mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(mi - nb, mi),
                slice(0, mj),
                field_name,
                field_units,
            )
        )
        field[nb : mi - nb, :nb] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(nb, mi - nb),
                slice(0, nb),
                field_name,
                field_units,
            )
        )
        field[nb : mi - nb, mj - nb : mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(nb, mi - nb),
                slice(mj - nb, mj),
                field_name,
                field_units,
            )
        )

    def set_outermost_layers_x(self, field, field_name=None, field_units=None, time=None):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.ny
        )

        ngrid = self.numerical_grid
        field[:1, :mj] = self.as_storage(
            data=core(time, ngrid, slice(0, 1), slice(0, mj), field_name, field_units)
        )
        field[mi - 1 : mi, :mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(mi - 1, mi),
                slice(0, mj),
                field_name,
                field_units,
            )
        )

    def set_outermost_layers_y(self, field, field_name=None, field_units=None, time=None):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.ny
        )

        ngrid = self.numerical_grid
        field[:mi, :1] = self.as_storage(
            data=core(time, ngrid, slice(0, mi), slice(0, 1), field_name, field_units)
        )
        field[:mi, mj - 1 : mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(0, mi),
                slice(mj - 1, mj),
                field_name,
                field_units,
            )
        )


class Dirichlet1DX(HorizontalBoundary):
    """Dirichlet boundary conditions for a physical grid with ``ny=1``."""

    def __init__(self, grid, nb, backend, storage_options, core=placeholder):
        """
        Parameters
        ----------
        grid : Grid
            The underlying three-dimensional physical grid.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        storage_options : StorageOptions
            Storage-related options.
        core : `callable`, optional
            Callable object providing the boundary layers values.
        """
        nx, ny = grid.nx, grid.ny
        assert nx > 1, "Number of grid points along first dimension should be larger " "than 1."
        assert ny == 1, "Number of grid points along second dimension must be 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed nx/2."

        signature = inspect.signature(core)
        error_msg = (
            "The signature of the core function should be "
            "core(time, grid, slice_x=None, slice_y=None, field_name=None, "
            "field_units=None)"
        )
        assert tuple(signature.parameters.keys())[0] == "time", error_msg
        assert tuple(signature.parameters.keys())[1] == "grid", error_msg
        assert "slice_x" in signature.parameters, error_msg
        assert "slice_y" in signature.parameters, error_msg
        assert "field_name" in signature.parameters, error_msg
        assert "field_units" in signature.parameters, error_msg

        super().__init__(grid, nb, backend=backend, storage_options=storage_options)

        self._kwargs["core"] = core

    @property
    def ni(self):
        return self.nx

    @property
    def nj(self):
        return 2 * self.nb + 1

    def get_numerical_xaxis(self, dims=None):
        return change_dims(self.physical_grid.x, dims)

    def get_numerical_xaxis_staggered(self, dims=None):
        return change_dims(self.physical_grid.x_at_u_locations, dims)

    def get_numerical_yaxis(self, dims=None):
        return repeat_axis(self.physical_grid.y, self.nb, dims)

    def get_numerical_yaxis_staggered(self, dims=None):
        return repeat_axis(self.physical_grid.y_at_v_locations, self.nb, dims)

    def get_numerical_field(self, field, field_name=None):
        try:
            li, lj, lk = field.shape
            trg = self.zeros(shape=(li, lj + 2 * self.nb, lk))
            src = field
        except ValueError:
            # just resort to numpy for 2d arrays
            li, lj = field.shape
            trg = np.zeros((li, lj + 2 * self.nb), dtype=self.storage_options.dtype)
            src = to_numpy(field)

        trg[:, : self.nb + 1] = src[:, :1]
        trg[:, self.nb + 1 :] = src[:, -1:]

        return trg

    def get_physical_field(self, field, field_name=None):
        return field[:, self.nb : -self.nb]

    def enforce_field(self, field, field_name=None, field_units=None, time=None):
        nb, core = self.nb, self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )

        ngrid = self.numerical_grid
        field[:nb, nb : mj - nb] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(0, nb),
                slice(nb, mj - nb),
                field_name,
                field_units,
            )
        )
        field[mi - nb : mi, nb : mj - nb] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(mi - nb, mi),
                slice(nb, mj - nb),
                field_name,
                field_units,
            )
        )
        field[:mi, :nb] = field[:mi, nb : nb + 1]
        field[:mi, mj - nb : mj] = field[:mi, mj - nb - 1 : mj - nb]

    def set_outermost_layers_x(self, field, field_name=None, field_units=None, time=None):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )

        ngrid = self.numerical_grid
        field[:1, :mj] = self.as_storage(
            data=core(time, ngrid, slice(0, 1), slice(0, mj), field_name, field_units)
        )
        field[mi - 1 : mi, :mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(mi - 1, mi),
                slice(0, mj),
                field_name,
                field_units,
            )
        )

    def set_outermost_layers_y(self, field, field_name=None, field_units=None, time=None):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.nx + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.nx
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )

        ngrid = self.numerical_grid
        field[:mi, :1] = self.as_storage(
            data=core(time, ngrid, slice(0, mi), slice(0, 1), field_name, field_units)
        )
        field[:mi, mj - 1 : mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(0, mi),
                slice(mj - 1, mj),
                field_name,
                field_units,
            )
        )


class Dirichlet1DY(HorizontalBoundary):
    """Dirichlet boundary conditions for a physical grid with ``nx=1``."""

    def __init__(self, grid, nb, backend, storage_options, core=placeholder):
        """
        Parameters
        ----------
        grid : Grid
            The underlying three-dimensional physical grid.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        storage_options : StorageOptions
            Storage-related options.
        core : `callable`, optional
            Callable object providing the boundary layers values.
        """
        nx, ny = grid.nx, grid.ny
        assert nx == 1, "Number of grid points along first dimension must be 1."
        assert ny > 1, "Number of grid points along second dimension should be larger " "than 1."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        signature = inspect.signature(core)
        error_msg = (
            "The signature of the core function should be "
            "core(time, grid, slice_x=None, slice_y=None, field_name=None, "
            "field_units=None)"
        )
        assert tuple(signature.parameters.keys())[0] == "time", error_msg
        assert tuple(signature.parameters.keys())[1] == "grid", error_msg
        assert "slice_x" in signature.parameters, error_msg
        assert "slice_y" in signature.parameters, error_msg
        assert "field_name" in signature.parameters, error_msg
        assert "field_units" in signature.parameters, error_msg

        super().__init__(grid, nb, backend=backend, storage_options=storage_options)

        self._kwargs["core"] = core

    @property
    def ni(self):
        return 2 * self.nb + 1

    @property
    def nj(self):
        return self.ny

    def get_numerical_xaxis(self, dims=None):
        return repeat_axis(self.physical_grid.x, self.nb, dims)

    def get_numerical_xaxis_staggered(self, dims=None):
        return repeat_axis(self.physical_grid.x_at_u_locations, self.nb, dims)

    def get_numerical_yaxis(self, dims=None):
        return change_dims(self.physical_grid.y, dims)

    def get_numerical_yaxis_staggered(self, dims=None):
        return change_dims(self.physical_grid.y_at_v_locations, dims)

    def get_numerical_field(self, field, field_name=None):
        try:
            li, lj, lk = field.shape
            trg = self.zeros(shape=(li + 2 * self.nb, lj, lk))
            src = field
        except ValueError:
            # resort to numpy for 2d arrays
            li, lj = field.shape
            trg = np.zeros((li + 2 * self.nb, lj), dtype=self.storage_options.dtype)
            src = to_numpy(field)

        trg[: self.nb + 1, :] = src[:1, :]
        trg[self.nb + 1 :, :] = src[-1:, :]

        return trg

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, :]

    def enforce_field(self, field, field_name=None, field_units=None, time=None):
        nb, core = self.nb, self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.ny
        )

        ngrid = self.numerical_grid
        field[nb : mi - nb, :nb] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(nb, mi - nb),
                slice(0, nb),
                field_name,
                field_units,
            )
        )
        field[nb : mi - nb, mj - nb : mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(nb, mi - nb),
                slice(mj - nb, mj),
                field_name,
                field_units,
            )
        )
        field[:nb, :mj] = field[nb : nb + 1, :mj]
        field[mi - nb : mi, :mj] = field[mi - nb - 1 : mi - nb, :mj]

    def set_outermost_layers_x(self, field, field_name=None, field_units=None, time=None):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.ny
        )

        ngrid = self.numerical_grid
        field[:1, :mj] = self.as_storage(
            data=core(time, ngrid, slice(0, 1), slice(0, mj), field_name, field_units)
        )
        field[mi - 1 : mi, :mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(mi - 1, mi),
                slice(0, mj),
                field_name,
                field_units,
            )
        )

    def set_outermost_layers_y(self, field, field_name=None, field_units=None, time=None):
        core = self._kwargs["core"]

        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.ny + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.ny
        )

        ngrid = self.numerical_grid
        field[:mi, :1] = self.as_storage(
            data=core(time, ngrid, slice(0, mi), slice(0, 1), field_name, field_units)
        )
        field[:mi, mj - 1 : mj] = self.as_storage(
            data=core(
                time,
                ngrid,
                slice(0, mi),
                slice(mj - 1, mj),
                field_name,
                field_units,
            )
        )


@register(name="dirichlet", registry_class=HorizontalBoundary)
def dispatch(
    grid,
    nb,
    backend="numpy",
    backend_options=None,
    storage_shape=None,
    storage_options=None,
    core=placeholder,
):
    """Instantiate the appropriate class based on the grid size."""
    if grid.nx == 1:
        return Dirichlet1DY(grid, nb, backend, storage_options, core)
    elif grid.ny == 1:
        return Dirichlet1DX(grid, nb, backend, storage_options, core)
    else:
        return Dirichlet(grid, nb, backend, storage_options, core)
