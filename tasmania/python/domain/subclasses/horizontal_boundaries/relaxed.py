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

import gt4py as gt

from tasmania.python.domain.horizontal_boundary import HorizontalBoundary
from tasmania.python.domain.subclasses.horizontal_boundaries.utils import (
    repeat_axis,
    shrink_axis,
)
from tasmania.python.utils.framework_utils import register
from tasmania.python.utils.gtscript_utils import stencil_irelax_defs
from tasmania.python.utils.storage_utils import get_asarray_function, zeros
from tasmania.python.utils.utils import get_gt_backend, is_gt


class Relaxed(HorizontalBoundary):
    """ Relaxed boundary conditions. """

    def __init__(
        self,
        nx,
        ny,
        nb,
        backend,
        backend_opts,
        dtype,
        build_info,
        exec_info,
        default_origin,
        rebuild,
        storage_shape,
        managed_memory,
        nr=8,
        nz=None,
    ):
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
        backend_opts : dict
            Dictionary of backend-specific options.
        dtype : data-type
            The data type of the storages.
        build_info : dict
            Dictionary of building options.
        exec_info : dict
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : tuple[int]
            Storage default origin.
        rebuild : bool
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        storage_shape : tuple[int]
            The shape of the storages allocated within the class.
            If not specified, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
            It cannot be smaller than ``(nx + 1, ny + 1, nz + 1)``.
        managed_memory : bool
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.
        nr : `int`, optional
            Depth of each relaxation region close to the
            horizontal boundaries. Minimum is ``nb``, maximum is 8 (default).
        nz : `int`, optional
            Number of vertical main levels.
            If not provided, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nr <= nx / 2, "Depth of relaxation region cannot exceed nx/2."
        assert nr <= ny / 2, "Depth of relaxation region cannot exceed ny/2."
        assert nr <= 8, "Depth of relaxation region cannot exceed 8."
        assert (
            nb <= nr
        ), "Number of boundary layers cannot exceed depth of relaxation region."

        super().__init__(
            nx,
            ny,
            nb,
            backend,
            backend_opts,
            dtype,
            build_info,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
            managed_memory,
        )
        self._kwargs["nr"] = nr
        self._kwargs["nz"] = nz

        self._ready2go = False
        self._allocate_coefficient_matrix()

        if is_gt(backend):
            self._stencil = gt.gtscript.stencil(
                definition=stencil_irelax_defs,
                backend=get_gt_backend(backend),
                build_info=build_info,
                dtypes={"dtype": dtype},
                rebuild=rebuild,
                **backend_opts
            )

    @property
    def ni(self):
        return self.nx

    @property
    def nj(self):
        return self.ny

    def get_numerical_xaxis(self, paxis, dims=None):
        cdims = dims if dims is not None else paxis.dims[0]
        return DataArray(
            paxis.values,
            coords=[paxis.values],
            dims=cdims,
            attrs={"units": paxis.attrs["units"]},
        )

    def get_numerical_yaxis(self, paxis, dims=None):
        return self.get_numerical_xaxis(paxis, dims)

    def get_numerical_field(self, field, field_name=None):
        return np.asarray(field)

    def get_physical_xaxis(self, caxis, dims=None):
        pdims = dims if dims is not None else caxis.dims[0]
        return DataArray(
            caxis.values,
            coords=[caxis.values],
            dims=pdims,
            attrs={"units": caxis.attrs["units"]},
        )

    def get_physical_yaxis(self, caxis, dims=None):
        pdims = dims if dims is not None else caxis.dims[0]
        return DataArray(
            caxis.values,
            coords=[caxis.values],
            dims=pdims,
            attrs={"units": caxis.attrs["units"]},
        )

    def get_physical_field(self, field, field_name=None):
        return np.asarray(field)

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        # shortcuts
        nx, ny = self.nx, self.ny
        nb, nr = self.nb, self._kwargs["nr"]
        field_name = field_name or ""

        # convenient definitions
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )

        # if needed, allocate the coefficient matrix
        if not self._ready2go:
            if self._kwargs["nz"] is None:
                if grid is not None:
                    nz = grid.nz
                elif field.ndim <= 2:
                    nz = 1
                else:
                    nz = (
                        field.shape[2] - 1
                        if "on_interface_levels" in field_name
                        else field.shape[2]
                    )
                self._kwargs["nz"] = nz
            self._allocate_coefficient_matrix(field.shape)

        # the coefficient matrix
        g = self._gamma

        # the vertical slice to fill
        mk = (
            self._kwargs["nz"] + 1
            if "on_interface_levels" in field_name
            else self._kwargs["nz"]
        )
        k = slice(0, mk)

        # the boundary values
        field_ref = self.reference_state[field_name].to_units(field_units).data
        # xneg = np.repeat(field_ref[0:1, :], nr, axis=0)
        # xpos = np.repeat(field_ref[-1:, :], nr, axis=0)
        # yneg = np.repeat(field_ref[:, 0:1], nr, axis=1)
        # ypos = np.repeat(field_ref[:, -1:], nr, axis=1)

        if not is_gt(self._backend):
            # field[:mi, :mj, :mk] -= g[:mi, :mj] * (
            #     field[:mi, :mj, :mk] - field_ref[:mi, :mj, :mk]
            # )

            # set the outermost layers
            field[:nb, :mj, k] = field_ref[:nb, :mj, k]
            field[mi - nb : mi, :mj, k] = field_ref[mi - nb : mi, :mj, k]
            field[nb : mi - nb, :nb, k] = field_ref[nb : mi - nb, :nb, k]
            field[nb : mi - nb, mj - nb : mj, k] = field_ref[
                nb : mi - nb, mj - nb : mj, k
            ]

            # apply the relaxed boundary conditions in the negative x-direction
            i, j = slice(nb, nr), slice(nb, nr)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
            i, j = slice(nb, nr), slice(nr, mj - nr)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
            i, j = slice(nb, nr), slice(mj - nr, mj - nb)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )

            # apply the relaxed boundary conditions in the positive x-direction
            i, j = slice(nx - nr, mi - nb), slice(nb, nr)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
            i, j = slice(nx - nr, mi - nb), slice(nr, mj - nr)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
            i, j = slice(nx - nr, mi - nb), slice(mj - nr, mj - nb)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )

            # apply the relaxed boundary conditions in the negative y-direction
            i, j = slice(nr, nx - nr), slice(nb, nr)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )

            # apply the relaxed boundary conditions in the positive y-direction
            i, j = slice(nr, nx - nr), slice(ny - nr, mj - nb)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
        else:
            self._stencil(
                in_gamma=g,
                in_phi_ref=field_ref,
                inout_phi=field,
                origin=(0, 0, 0),
                domain=(mi, mj, mk),
                validate_args=True
            )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[0, :mj] = field_ref[0, :mj]
        field[mi - 1, :mj] = field_ref[mi - 1, :mj]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[:mi, 0] = field_ref[:mi, 0]
        field[:mi, mj - 1] = field_ref[:mi, mj - 1]

    def _allocate_coefficient_matrix(self, field_shape=None):
        nx, ny = self.nx, self.ny
        nb, nr = self.nb, self._kwargs["nr"]
        backend, dtype = self._backend, self._dtype
        nz = self._kwargs["nz"]

        if nz is None or (is_gt(backend) and field_shape is None):
            self._ready2go = False
            return
        else:
            self._ready2go = True

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

        # inspect the backend properties to load the proper asarray function
        asarray = get_asarray_function(backend or "numpy")

        if not is_gt(backend):
            # create a single coefficient matrix
            self._gamma = zeros(
                (nx + 1, ny + 1, 1), backend=backend, dtype=dtype
            )
        else:
            # get the proper storage shape
            min_shape = (nx + 1, ny + 1, nz + 1)
            fshape = self._storage_shape or min_shape
            shape = tuple(
                max(fshape[i], field_shape[i]) for i in range(len(field_shape))
            )
            self._storage_shape = shape

            # create a single coefficient matrix
            self._gamma = zeros(
                shape,
                backend=backend,
                dtype=dtype,
                default_origin=self._default_origin,
                managed_memory=self._managed_memory,
            )

        # fill the coefficient matrix
        g = self._gamma
        g[:nr, :nr] = asarray(xnegyneg[:, :, np.newaxis])
        g[:nr, nr : ny - nr] = asarray(xneg[:, :-1, np.newaxis])
        g[:nr, ny - nr : ny] = asarray(xnegypos[:, :, np.newaxis])
        g[nx - nr : nx, :nr] = asarray(xposyneg[:, :, np.newaxis])
        g[nx - nr : nx, nr : ny - nr] = asarray(xpos[:, :-1, np.newaxis])
        g[nx - nr : nx, ny - nr : ny] = asarray(xposypos[:, :, np.newaxis])
        g[nr : nx - nr, :nr] = asarray(yneg[:-1, :, np.newaxis])
        g[nr : nx - nr, ny - nr : ny] = asarray(ypos[:-1, :, np.newaxis])
        g[nx : nx + 1, : ny + 1] = 1.0
        g[: nx + 1, ny : ny + 1] = 1.0


class Relaxed1DX(HorizontalBoundary):
    """ Relaxed boundary conditions for a physical grid with ``ny=1``. """

    def __init__(
        self,
        nx,
        ny,
        nb,
        backend,
        backend_opts,
        dtype,
        build_info,
        exec_info,
        default_origin,
        rebuild,
        storage_shape,
        managed_memory,
        nr=8,
        nz=None,
    ):
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
        backend_opts : dict
            Dictionary of backend-specific options.
        dtype : data-type
            The data type of the storages.
        build_info : dict
            Dictionary of building options.
        exec_info : dict
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : tuple[int]
            Storage default origin.
        rebuild : bool
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        storage_shape : tuple[int]
            The shape of the storages allocated within the class.
            If not specified, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
            It cannot be smaller than ``(nx + 1, 2 * nb + 2, nz + 1)``.
        managed_memory : bool
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.
        nr : `int`, optional
            Depth of each relaxation region close to the
            horizontal boundaries. Minimum is ``nb``, maximum is 8 (default).
        nz : `int`, optional
            Number of vertical main levels.
            If not provided, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert (
            ny == 1
        ), "Number of grid points along second dimension must be 1."
        assert nr <= nx / 2, "Depth of relaxation region cannot exceed nx/2."
        assert nr <= 8, "Depth of relaxation region cannot exceed 8."
        assert (
            nb <= nr
        ), "Number of boundary layers cannot exceed depth of relaxation region."

        super().__init__(
            nx,
            ny,
            nb,
            backend,
            backend_opts,
            dtype,
            build_info,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
            managed_memory,
        )
        self._kwargs["nr"] = nr
        self._kwargs["nz"] = nz

        self._ready2go = False
        self._allocate_coefficient_matrix()

        if is_gt(backend):
            self._stencil = gt.gtscript.stencil(
                definition=stencil_irelax_defs,
                backend=get_gt_backend(backend),
                build_info=build_info,
                dtypes={"dtype": dtype},
                rebuild=rebuild,
                **backend_opts
            )

    @property
    def ni(self):
        return self.nx

    @property
    def nj(self):
        return 2 * self.nb + 1

    def get_numerical_xaxis(self, paxis, dims=None):
        cdims = dims if dims is not None else paxis.dims[0]
        return DataArray(
            paxis.values,
            coords=[paxis.values],
            dims=[cdims],
            attrs={"units": paxis.attrs["units"]},
        )

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
        pdims = dims if dims is not None else caxis.dims[0]
        return DataArray(
            caxis.values,
            coords=[caxis.values],
            dims=pdims,
            attrs={"units": caxis.attrs["units"]},
        )

    def get_physical_yaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_field(self, field, field_name=None):
        return np.asarray(field[:, self.nb : -self.nb])

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        # shortcuts
        nx = self.nx
        nb, nr = self.nb, self._kwargs["nr"]
        field_name = field_name or ""

        # convenient definitions
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )

        # if needed, allocate the coefficient matrix
        if not self._ready2go:
            if self._kwargs["nz"] is None:
                if grid is not None:
                    nz = grid.nz
                elif field.ndim <= 2:
                    nz = 1
                else:
                    nz = (
                        field.shape[2] - 1
                        if "on_interface_levels" in field_name
                        else field.shape[2]
                    )
                self._kwargs["nz"] = nz
            self._allocate_coefficient_matrix(field.shape)

        # the coefficient matrix
        g = self._gamma

        # the vertical slice to fill
        mk = (
            self._kwargs["nz"] + 1
            if "on_interface_levels" in field_name
            else self._kwargs["nz"]
        )
        k = slice(0, mk)

        # the boundary values
        field_ref = self.reference_state[field_name].to_units(field_units).data
        # xneg = np.repeat(field_ref[0:1, nb:-nb], nr, axis=0)
        # xpos = np.repeat(field_ref[-1:, nb:-nb], nr, axis=0)

        if not is_gt(self._backend):
            # set the outermost layers
            field[:nb, nb : mj - nb, k] = field_ref[:nb, nb : mj - nb, k]
            field[mi - nb : mi, nb : mj - nb, k] = field_ref[
                mi - nb : mi, nb : mj - nb, k
            ]

            # apply the relaxed boundary conditions in the negative x-direction
            i, j = slice(nb, nr), slice(nb, mj - nb)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )

            # apply the relaxed boundary conditions in the positive x-direction
            i, j = slice(nx - nr, mi - nb), slice(nb, mj - nb)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
        else:
            self._stencil(
                in_gamma=g,
                in_phi_ref=field_ref,
                inout_phi=field,
                origin=(0, nb, 0),
                domain=(mi, mj - nb, mk),
                validate_args=True
            )

        # repeat the innermost column(s) along the y-direction
        field[:mi, :nb, k] = field[:mi, nb : nb + 1, k]
        field[:mi, mj - nb : mj, k] = field[:mi, mj - nb - 1 : mj - nb, k]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[0, :mj] = field_ref[0, :mj]
        field[mi - 1, :mj] = field_ref[mi - 1, :mj]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[:mi, 0] = field_ref[:mi, 0]
        field[:mi, mj - 1] = field_ref[:mi, mj - 1]

    def _allocate_coefficient_matrix(self, field_shape=None):
        nx, ny = self.nx, self.ny
        nb, nr = self.nb, self._kwargs["nr"]
        backend, dtype = self._backend, self._dtype
        nz = self._kwargs["nz"]

        if nz is None or (is_gt(backend) and field_shape is None):
            self._ready2go = False
            return
        else:
            self._ready2go = True

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

        # inspect the backend properties to load the proper asarray function
        asarray = get_asarray_function(backend or "numpy")

        if not is_gt(backend):
            # create a single coefficient matrix
            self._gamma = zeros(
                (nx + 1, 2 * nb + 2, 1), backend=backend, dtype=dtype
            )
        else:
            # get the proper storage shape
            min_shape = (nx + 1, 2 * nb + 2, nz + 1)
            fshape = self._storage_shape or min_shape
            shape = tuple(
                max(fshape[i], field_shape[i]) for i in range(len(field_shape))
            )
            self._storage_shape = shape

            # create a single coefficient matrix
            self._gamma = zeros(
                shape,
                backend=backend,
                dtype=dtype,
                default_origin=self._default_origin,
                managed_memory=self._managed_memory,
            )

        # fill the coefficient matrix
        g = self._gamma
        g[:nr, nb : nb + 2] = asarray(xneg[:, :, np.newaxis])
        g[nx - nr : nx, nb : nb + 2] = asarray(xpos[:, :, np.newaxis])
        g[nx : nx + 1, nb : nb + 2] = 1.0


class Relaxed1DY(HorizontalBoundary):
    """ Relaxed boundary conditions for a physical grid with ``nx=1``. """

    def __init__(
        self,
        nx,
        ny,
        nb,
        backend,
        backend_opts,
        dtype,
        build_info,
        exec_info,
        default_origin,
        rebuild,
        storage_shape,
        managed_memory,
        nr=8,
        nz=None,
    ):
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
        backend_opts : dict
            Dictionary of backend-specific options.
        dtype : data-type
            The data type of the storages.
        build_info : dict
            Dictionary of building options.
        exec_info : dict
            Dictionary which will store statistics and diagnostics gathered at
            run time.
        default_origin : tuple[int]
            Storage default origin.
        rebuild : bool
            ``True`` to trigger the stencils compilation at any class
            instantiation, ``False`` to rely on the caching mechanism
            implemented by the backend.
        storage_shape : tuple[int]
            The shape of the storages allocated within the class.
            If not specified, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
            It cannot be smaller than ``(2 * nb + 2, ny + 1, nz + 1)``.
        managed_memory : bool
            ``True`` to allocate the storages as managed memory,
            ``False`` otherwise.
        nr : `int`, optional
            Depth of each relaxation region close to the
            horizontal boundaries. Minimum is ``nb``, maximum is 8 (default).
        nz : `int`, optional
            Number of vertical main levels.
            If not provided, it will be inferred from the arguments passed to
            to the ``enforce_field`` method the first time it is invoked.
        """
        assert (
            nx == 1
        ), "Number of grid points along first dimension must be 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nr <= ny / 2, "Depth of relaxation region cannot exceed ny/2."
        assert nr <= 8, "Depth of relaxation region cannot exceed 8."
        assert (
            nb <= nr
        ), "Number of boundary layers cannot exceed depth of relaxation region."

        super().__init__(
            nx,
            ny,
            nb,
            backend,
            backend_opts,
            dtype,
            build_info,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
            managed_memory,
        )
        self._kwargs["nr"] = nr
        self._kwargs["nz"] = nz

        self._ready2go = False
        self._allocate_coefficient_matrix()

        if is_gt(backend):
            self._stencil = gt.gtscript.stencil(
                definition=stencil_irelax_defs,
                backend=get_gt_backend(backend),
                build_info=build_info,
                dtypes={"dtype": dtype},
                rebuild=rebuild,
                **backend_opts
            )

    @property
    def ni(self):
        return 2 * self.nb + 1

    @property
    def nj(self):
        return self.ny

    def get_numerical_xaxis(self, paxis, dims=None):
        return repeat_axis(paxis, self.nb, dims)

    def get_numerical_yaxis(self, paxis, dims=None):
        cdims = dims if dims is not None else paxis.dims[0]
        return DataArray(
            paxis.values,
            coords=[paxis.values],
            dims=cdims,
            attrs={"units": paxis.attrs["units"]},
        )

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
        pdims = dims if dims is not None else caxis.dims[0]
        return DataArray(
            caxis.values,
            coords=[caxis.values],
            dims=pdims,
            attrs={"units": caxis.attrs["units"]},
        )

    def get_physical_field(self, field, field_name=None):
        return np.asarray(field[self.nb : -self.nb, :])

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        # shortcuts
        ny = self.ny
        nb, nr = self.nb, self._kwargs["nr"]
        field_name = field_name or ""

        # convenient definitions
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )
        # if needed, allocate the coefficient matrix
        if not self._ready2go:
            if self._kwargs["nz"] is None:
                if grid is not None:
                    nz = grid.nz
                elif field.ndim <= 2:
                    nz = 1
                else:
                    nz = (
                        field.shape[2] - 1
                        if "on_interface_levels" in field_name
                        else field.shape[2]
                    )
                self._kwargs["nz"] = nz
            self._allocate_coefficient_matrix(field.shape)

        # the coefficient matrix
        g = self._gamma

        # the vertical slice to fill
        mk = (
            self._kwargs["nz"] + 1
            if "on_interface_levels" in field_name
            else self._kwargs["nz"]
        )
        k = slice(0, mk)

        # the boundary values
        field_ref = self.reference_state[field_name].to_units(field_units).data
        # yneg = np.repeat(field_ref[nb:-nb, 0:1], nr, axis=1)
        # ypos = np.repeat(field_ref[nb:-nb, -1:], nr, axis=1)

        if not is_gt(self._backend):
            # set the outermost layers
            field[nb : mi - nb, :nb, k] = field_ref[nb : mi - nb, :nb, k]
            field[nb : mi - nb, mj - nb : mj, k] = field_ref[
                nb : mi - nb, mj - nb : mj, k
            ]

            # apply the relaxed boundary conditions in the negative y-direction
            i, j = slice(nb, mi - nb), slice(nb, nr)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )

            # apply the relaxed boundary conditions in the positive y-direction
            i, j = slice(nb, mi - nb), slice(ny - nr, mj - nb)
            field[i, j, k] -= g[i, j, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
        else:
            self._stencil(
                in_gamma=g,
                in_phi_ref=field_ref,
                inout_phi=field,
                origin=(nb, 0, 0),
                domain=(mi - nb, mj, mk),
                validate_args=True
            )

        # repeat the innermost row(s) along the x-direction
        field[:nb, :mj, k] = field[nb : nb + 1, :mj, k]
        field[mi - nb : mi, :mj, k] = field[mi - nb - 1 : mi - nb, :mj, k]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[0, :mj] = field_ref[0, :mj]
        field[mi - 1, :mj] = field_ref[mi - 1, :mj]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        field_name = field_name or ""
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name
            or "at_uv_locations" in field_name
            else self.ni
        )
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name
            or "at_uv_locations" in field_name
            else self.nj
        )
        field_ref = self.reference_state[field_name].to_units(field_units).data
        field[:mi, 0] = field_ref[:mi, 0]
        field[:mi, mj - 1] = field_ref[:mi, mj - 1]

    def _allocate_coefficient_matrix(self, field_shape=None):
        nx, ny = self.nx, self.ny
        nb, nr = self.nb, self._kwargs["nr"]
        backend, dtype = self._backend, self._dtype
        nz = self._kwargs["nz"]

        if nz is None or (is_gt(backend) and field_shape is None):
            self._ready2go = False
            return
        else:
            self._ready2go = True

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

        # inspect the backend properties to load the proper asarray function
        asarray = get_asarray_function(backend or "numpy")

        if not is_gt(backend):
            # create a single coefficient matrix
            self._gamma = zeros(
                (2 * nb + 2, ny + 1, 1), backend=backend, dtype=dtype
            )
        else:
            # get the proper storage shape
            min_shape = (2 * nb + 2, ny + 1, nz + 1)
            fshape = self._storage_shape or min_shape
            shape = tuple(
                max(fshape[i], field_shape[i]) for i in range(len(field_shape))
            )
            self._storage_shape = shape

            # create a single coefficient matrix
            self._gamma = zeros(
                shape,
                backend=backend,
                dtype=dtype,
                default_origin=self._default_origin,
                managed_memory=self._managed_memory,
            )

        # fill the coefficient matrix
        g = self._gamma
        g[nb : nb + 2, :nr] = asarray(yneg[:, :, np.newaxis])
        g[nb : nb + 2, ny - nr : ny] = asarray(ypos[:, :, np.newaxis])
        g[nb : nb + 2, ny : ny + 1] = 1.0


@register(name="relaxed", registry_class=HorizontalBoundary)
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
    nr=8,
    nz=None,
):
    """ Dispatch based on the grid size. """
    if nx == 1:
        return Relaxed1DY(
            1,
            ny,
            nb,
            backend,
            backend_opts,
            dtype,
            build_info,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
            managed_memory,
            nr,
            nz,
        )
    elif ny == 1:
        return Relaxed1DX(
            nx,
            1,
            nb,
            backend,
            backend_opts,
            dtype,
            build_info,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
            managed_memory,
            nr,
            nz,
        )
    else:
        return Relaxed(
            nx,
            ny,
            nb,
            backend,
            backend_opts,
            dtype,
            build_info,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
            managed_memory,
            nr,
            nz,
        )
