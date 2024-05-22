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
import numpy as np

try:
    import cupy as cp
except (ImportError, ModuleNotFoundError):
    cp = np

from sympl._core.time import Timer

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.subclasses.horizontal_boundaries.utils import (
    change_dims,
    repeat_axis,
)
from tasmania.python.framework.generic_functions import to_numpy
from tasmania.python.framework.register import register
from tasmania.python.utils.storage import get_storage_shape


class Relaxed(HorizontalBoundary):
    """Relaxed boundary conditions."""

    def __init__(
        self,
        grid,
        nb,
        backend,
        backend_options,
        storage_shape,
        storage_options,
        nr=8,
    ):
        """
        Parameters
        ----------
        grid : Grid
            The underlying three-dimensional physical grid.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        backend_options : BackendOptions
            Backend-specific options.
        storage_shape : tuple[int]
            The shape of the storages allocated within the class.
            If not specified, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
            It cannot be smaller than ``(nx + 1, ny + 1, nz + 1)``.
        storage_options : StorageOptions
            Storage-related options.
        nr : `int`, optional
            Depth of each relaxation region close to the
            horizontal boundaries. Minimum is ``nb``, maximum is 8 (default).
        """
        nx, ny = grid.nx, grid.ny
        assert nx > 1, "Number of grid points along first dimension should be larger " "than 1."
        assert ny > 1, "Number of grid points along second dimension should be larger " "than 1."
        assert nr <= nx / 2, "Depth of relaxation region cannot exceed nx/2."
        assert nr <= ny / 2, "Depth of relaxation region cannot exceed ny/2."
        assert nr <= 8, "Depth of relaxation region cannot exceed 8."
        assert nb <= nr, "Number of boundary layers cannot exceed depth of relaxation " "region."

        super().__init__(
            grid,
            nb,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )
        self._kwargs["nr"] = nr

        self._allocate_coefficient_matrix()

        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil = self.compile_stencil("irelax")

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
        # shortcuts
        g = self._gamma
        field_name = field_name or ""

        # extent of the computational domain
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        mk = (
            self.physical_grid.nz + 1
            if "on_interface_levels" in field_name
            else self.physical_grid.nz
        )

        # the boundary values
        field_ref = self.reference_state[field_name].to_units(field_units).data
        # xneg = np.repeat(field_ref[0:1, :], nr, axis=0)
        # xpos = np.repeat(field_ref[-1:, :], nr, axis=0)
        # yneg = np.repeat(field_ref[:, 0:1], nr, axis=1)
        # ypos = np.repeat(field_ref[:, -1:], nr, axis=1)

        # apply the relaxation
        Timer.start(label="stencil")
        self._stencil(
            in_gamma=g,
            in_phi_ref=field_ref,
            inout_phi=field,
            origin=(0, 0, 0),
            domain=(mi, mj, mk),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

    def set_outermost_layers_x(self, field, field_name=None, field_units=None, time=None):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[0, :mj] = field_ref[0, :mj]
        field[mi - 1, :mj] = field_ref[mi - 1, :mj]

    def set_outermost_layers_y(self, field, field_name=None, field_units=None, time=None):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[:mi, 0] = field_ref[:mi, 0]
        field[:mi, mj - 1] = field_ref[:mi, mj - 1]

    def _allocate_coefficient_matrix(self):
        nx, ny, nz = self.nx, self.ny, self.physical_grid.nz
        nb, nr = self.nb, self._kwargs["nr"]

        # the relaxation coefficients
        rel = np.array(
            [
                1.0,
                1.0 - np.tanh(0.5),
                1.0 - np.tanh(1.0),
                1.0 - np.tanh(1.5),
                1.0 - np.tanh(2.0),
                1.0 - np.tanh(2.5),
                1.0 - np.tanh(3.0),
                1.0 - np.tanh(3.5),
            ]
        )
        rel = rel[:nr]
        rel[:nb] = 1.0
        rrel = rel[::-1]

        # the relaxation matrices
        xneg = np.repeat(rel[:, np.newaxis], ny - 2 * nr + 1, axis=1)
        xpos = np.repeat(rrel[:, np.newaxis], ny - 2 * nr + 1, axis=1)
        yneg = np.repeat(rel[np.newaxis, :], nx - 2 * nr + 1, axis=0)
        ypos = np.repeat(rrel[np.newaxis, :], nx - 2 * nr + 1, axis=0)

        # the corner relaxation matrices
        xnegyneg = np.zeros((nr, nr))
        for i in range(nr):
            xnegyneg[i, i:] = rel[i]
            xnegyneg[i:, i] = rel[i]
        xposyneg = xnegyneg[::-1, :]
        xposypos = xposyneg[:, ::-1]
        xnegypos = xnegyneg[:, ::-1]

        # get the proper storage shape
        shape = get_storage_shape(self._storage_shape, min_shape=(nx + 1, ny + 1, nz + 1))
        self._storage_shape = shape

        # create a single coefficient matrix
        self._gamma = self.zeros(shape=shape)

        # fill the coefficient matrix
        g = self._gamma
        g[:nr, :nr] = self.as_storage(data=xnegyneg[:, :, np.newaxis])
        g[:nr, nr : ny - nr] = self.as_storage(data=xneg[:, :-1, np.newaxis])
        g[:nr, ny - nr : ny] = self.as_storage(data=xnegypos[:, :, np.newaxis])
        g[nx - nr : nx, :nr] = self.as_storage(data=xposyneg[:, :, np.newaxis])
        g[nx - nr : nx, nr : ny - nr] = self.as_storage(data=xpos[:, :-1, np.newaxis])
        g[nx - nr : nx, ny - nr : ny] = self.as_storage(data=xposypos[:, :, np.newaxis])
        g[nr : nx - nr, :nr] = self.as_storage(data=yneg[:-1, :, np.newaxis])
        g[nr : nx - nr, ny - nr : ny] = self.as_storage(data=ypos[:-1, :, np.newaxis])
        g[nx : nx + 1, : ny + 1] = 1.0
        g[: nx + 1, ny : ny + 1] = 1.0


class Relaxed1DX(HorizontalBoundary):
    """Relaxed boundary conditions for a physical grid with ``ny=1``."""

    def __init__(
        self,
        grid,
        nb,
        backend,
        backend_options,
        storage_shape,
        storage_options,
        nr=8,
    ):
        """
        Parameters
        ----------
        grid : Grid
            The underlying three-dimensional physical grid.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        backend_options : BackendOptions
            Backend-specific options.
        storage_shape : tuple[int]
            The shape of the storages allocated within the class.
            If not specified, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
            It cannot be smaller than ``(nx + 1, 2 * nb + 2, nz + 1)``.
        storage_options : StorageOptions
            Storage-related options.
        nr : `int`, optional
            Depth of each relaxation region close to the
            horizontal boundaries. Minimum is ``nb``, maximum is 8 (default).
        """
        nx, ny = grid.nx, grid.ny
        assert nx > 1, "Number of grid points along first dimension should be larger " "than 1."
        assert ny == 1, "Number of grid points along second dimension must be 1."
        assert nr <= nx / 2, "Depth of relaxation region cannot exceed nx/2."
        assert nr <= 8, "Depth of relaxation region cannot exceed 8."
        assert nb <= nr, "Number of boundary layers cannot exceed depth of relaxation " "region."

        super().__init__(
            grid,
            nb,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )
        self._kwargs["nr"] = nr

        self._allocate_coefficient_matrix()

        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil = self.compile_stencil("irelax")

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
            li, lj = field.shape
            trg = np.zeros((li, lj + 2 * self.nb), dtype=self.storage_options.dtype)
            src = to_numpy(field)

        trg[:, : self.nb + 1] = src[:, :1]
        trg[:, self.nb + 1 :] = src[:, -1:]

        return trg

    def get_physical_field(self, field, field_name=None):
        return field[:, self.nb : -self.nb]

    def enforce_field(self, field, field_name=None, field_units=None, time=None):
        # shortcuts
        nb, nr = self.nb, self._kwargs["nr"]
        g = self._gamma
        field_name = field_name or ""

        # extent of the computational domain
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        mk = (
            self.physical_grid.nz + 1
            if "on_interface_levels" in field_name
            else self.physical_grid.nz
        )
        k = slice(0, mk)

        # the boundary values
        field_ref = self.reference_state[field_name].to_units(field_units).data
        # xneg = np.repeat(field_ref[0:1, nb:-nb], nr, axis=0)
        # xpos = np.repeat(field_ref[-1:, nb:-nb], nr, axis=0)

        # apply relaxation
        Timer.start(label="stencil")
        self._stencil(
            in_gamma=g,
            in_phi_ref=field_ref,
            inout_phi=field,
            origin=(0, nb, 0),
            domain=(mi, mj - nb, mk),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

        # repeat the innermost column(s) along the y-direction
        field[:mi, :nb, k] = field[:mi, nb : nb + 1, k]
        field[:mi, mj - nb : mj, k] = field[:mi, mj - nb - 1 : mj - nb, k]

    def set_outermost_layers_x(self, field, field_name=None, field_units=None, time=None):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[0, :mj] = field_ref[0, :mj]
        field[mi - 1, :mj] = field_ref[mi - 1, :mj]

    def set_outermost_layers_y(self, field, field_name=None, field_units=None, time=None):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[:mi, 0] = field_ref[:mi, 0]
        field[:mi, mj - 1] = field_ref[:mi, mj - 1]

    def _allocate_coefficient_matrix(self):
        nx, ny, nz = self.nx, self.ny, self.physical_grid.nz
        nb, nr = self.nb, self._kwargs["nr"]

        # the relaxation coefficients
        rel = np.array(
            [
                1.0,
                1.0 - np.tanh(0.5),
                1.0 - np.tanh(1.0),
                1.0 - np.tanh(1.5),
                1.0 - np.tanh(2.0),
                1.0 - np.tanh(2.5),
                1.0 - np.tanh(3.0),
                1.0 - np.tanh(3.5),
            ]
        )
        rel = rel[:nr]
        rel[:nb] = 1.0
        rrel = rel[::-1]

        # the relaxation matrices
        xneg = np.repeat(rel[:, np.newaxis], 2, axis=1)
        xpos = np.repeat(rrel[:, np.newaxis], 2, axis=1)

        # get the proper storage shape
        shape = get_storage_shape(self._storage_shape, min_shape=(nx + 1, 2 * nb + 2, nz + 1))
        self._storage_shape = shape

        # create a single coefficient matrix
        self._gamma = self.zeros(shape=shape)

        # fill the coefficient matrix
        g = self._gamma
        g[:nr, nb : nb + 2] = self.as_storage(data=xneg[:, :, np.newaxis])
        g[nx - nr : nx, nb : nb + 2] = self.as_storage(data=xpos[:, :, np.newaxis])
        g[nx : nx + 1, nb : nb + 2] = 1.0


class Relaxed1DY(HorizontalBoundary):
    """Relaxed boundary conditions for a physical grid with ``nx=1``."""

    def __init__(
        self,
        grid,
        nb,
        backend,
        backend_options,
        storage_shape,
        storage_options,
        nr=8,
    ):
        """
        Parameters
        ----------
        grid : Grid
            The underlying three-dimensional physical grid.
        nb : int
            Number of boundary layers.
        backend : str
            The backend.
        backend_options : BackendOptions
            Backend-specific options.
        storage_shape : tuple[int]
            The shape of the storages allocated within the class.
            If not specified, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
            It cannot be smaller than ``(2 * nb + 2, ny + 1, nz + 1)``.
        storage_options : StorageOptions
            Storage-related options.
        nr : `int`, optional
            Depth of each relaxation region close to the
            horizontal boundaries. Minimum is ``nb``, maximum is 8 (default).
        nz : `int`, optional
            Number of vertical main levels.
            If not provided, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
        """
        nx, ny = grid.nx, grid.ny
        assert nx == 1, "Number of grid points along first dimension must be 1."
        assert ny > 1, "Number of grid points along second dimension should be larger " "than 1."
        assert nr <= ny / 2, "Depth of relaxation region cannot exceed ny/2."
        assert nr <= 8, "Depth of relaxation region cannot exceed 8."
        assert nb <= nr, "Number of boundary layers cannot exceed depth of relaxation " "region."

        super().__init__(
            grid,
            nb,
            backend=backend,
            backend_options=backend_options,
            storage_shape=storage_shape,
            storage_options=storage_options,
        )
        self._kwargs["nr"] = nr

        self._allocate_coefficient_matrix()

        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil = self.compile_stencil("irelax")

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
            li, lj = field.shape
            trg = np.zeros((li + 2 * self.nb, lj), dtype=self.storage_options.dtype)
            src = to_numpy(field)

        trg[: self.nb + 1, :] = src[:1, :]
        trg[self.nb + 1 :, :] = src[-1:, :]

        return trg

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, :]

    def enforce_field(self, field, field_name=None, field_units=None, time=None):
        # shortcuts
        nb, nr = self.nb, self._kwargs["nr"]
        g = self._gamma
        field_name = field_name or ""

        # extent of the computational domain
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        mk = (
            self.physical_grid.nz + 1
            if "on_interface_levels" in field_name
            else self.physical_grid.nz
        )
        k = slice(0, mk)

        # the boundary values
        field_ref = self.reference_state[field_name].to_units(field_units).data
        # yneg = np.repeat(field_ref[nb:-nb, 0:1], nr, axis=1)
        # ypos = np.repeat(field_ref[nb:-nb, -1:], nr, axis=1)

        # apply relaxation
        Timer.start(label="stencil")
        self._stencil(
            in_gamma=g,
            in_phi_ref=field_ref,
            inout_phi=field,
            origin=(nb, 0, 0),
            domain=(mi - nb, mj, mk),
            exec_info=self.backend_options.exec_info,
            validate_args=self.backend_options.validate_args,
        )
        Timer.stop()

        # repeat the innermost row(s) along the x-direction
        field[:nb, :mj, k] = field[nb : nb + 1, :mj, k]
        field[mi - nb : mi, :mj, k] = field[mi - nb - 1 : mi - nb, :mj, k]

    def set_outermost_layers_x(self, field, field_name=None, field_units=None, time=None):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[0, :mj] = field_ref[0, :mj]
        field[mi - 1, :mj] = field_ref[mi - 1, :mj]

    def set_outermost_layers_y(self, field, field_name=None, field_units=None, time=None):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[:mi, 0] = field_ref[:mi, 0]
        field[:mi, mj - 1] = field_ref[:mi, mj - 1]

    def _allocate_coefficient_matrix(self):
        nx, ny, nz = self.nx, self.ny, self.physical_grid.nz
        nb, nr = self.nb, self._kwargs["nr"]

        # the relaxation coefficients
        rel = np.array(
            [
                1.0,
                1.0 - np.tanh(0.5),
                1.0 - np.tanh(1.0),
                1.0 - np.tanh(1.5),
                1.0 - np.tanh(2.0),
                1.0 - np.tanh(2.5),
                1.0 - np.tanh(3.0),
                1.0 - np.tanh(3.5),
            ]
        )
        rel = rel[:nr]
        rel[:nb] = 1.0
        rrel = rel[::-1]

        # the relaxation matrices
        yneg = np.repeat(rel[np.newaxis, :], 2, axis=0)
        ypos = np.repeat(rrel[np.newaxis, :], 2, axis=0)

        # get the proper storage shape
        shape = get_storage_shape(self._storage_shape, min_shape=(2 * nb + 2, ny + 1, nz + 1))
        self._storage_shape = shape

        # create a single coefficient matrix
        self._gamma = self.zeros(shape=shape)

        # fill the coefficient matrix
        g = self._gamma
        g[nb : nb + 2, :nr] = self.as_storage(data=yneg[:, :, np.newaxis])
        g[nb : nb + 2, ny - nr : ny] = self.as_storage(data=ypos[:, :, np.newaxis])
        g[nb : nb + 2, ny : ny + 1] = 1.0


@register(name="relaxed", registry_class=HorizontalBoundary)
def dispatch(
    grid,
    nb,
    backend="numpy",
    backend_options=None,
    storage_shape=None,
    storage_options=None,
    nr=8,
):
    """Instantiate the appropriate class based on the grid size."""
    if grid.nx == 1:
        return Relaxed1DY(
            grid,
            nb,
            backend,
            backend_options,
            storage_shape,
            storage_options,
            nr,
        )
    elif grid.ny == 1:
        return Relaxed1DX(
            grid,
            nb,
            backend,
            backend_options,
            storage_shape,
            storage_options,
            nr,
        )
    else:
        return Relaxed(
            grid,
            nb,
            backend,
            backend_options,
            storage_shape,
            storage_options,
            nr,
        )
