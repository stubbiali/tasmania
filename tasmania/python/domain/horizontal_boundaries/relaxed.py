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
from tasmania.python.domain.horizontal_boundaries.utils import repeat_axis, shrink_axis
from tasmania.python.utils.framework_utils import register
from tasmania.python.utils.gtscript_utils import stencil_irelax_defs
from tasmania.python.utils.storage_utils import zeros


class Relaxed(HorizontalBoundary):
    """ Relaxed boundary conditions. """

    def __init__(self, nx, ny, nb, gt_powered, backend, dtype, nr=8, nz=None):
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
        gt_powered : bool
            ``True`` to harness GT4Py, ``False`` for a vanilla NumPy implementation.
        backend : str
            The GT4Py backend.
        dtype : data-type
            The data type of the storages.
        nr : `int`, optional
            Depth of each relaxation region close to the
            horizontal boundaries. Minimum is ``nb``, maximum is 8 (default).
        nz : `int`, optional
            Number of vertical main levels. Only needed if ``gt_powered=True``.
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

        super().__init__(nx, ny, nb, gt_powered, backend, dtype)
        self._kwargs["nr"] = nr
        self._kwargs["nz"] = nz
        self._allocate_coefficient_matrices()

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
        nb, nr = self.nb, self._kwargs["nr"]
        field_name = field_name or ""

        # convenient definitions
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mi_int = mi - 2 * nr
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )
        mj_int = mj - 2 * nr

        # if needed, allocate the coefficient matrices
        if self._xneg is None:
            nz = 1 if field.ndim <= 2 else field.shape[2]
            self._kwargs["nz"] = nz
            self._allocate_coefficient_matrices()

        # the boundary values
        field_ref = self.reference_state[field_name].to_units(field_units).values
        # xneg = np.repeat(field_ref[0:1, :], nr, axis=0)
        # xpos = np.repeat(field_ref[-1:, :], nr, axis=0)
        # yneg = np.repeat(field_ref[:, 0:1], nr, axis=1)
        # ypos = np.repeat(field_ref[:, -1:], nr, axis=1)

        if not self._gt_powered:
            # set the outermost layers
            field[:nb, :] = field_ref[:nb, :]
            field[mi - nb : mi, :] = field_ref[mi - nb : mi, :]
            field[nb : mi - nb, :nb] = field_ref[nb : mi - nb, :nb]
            field[nb : mi - nb, mj - nb : mj] = field_ref[nb : mi - nb, mj - nb : mj]

            # apply the relaxed boundary conditions in the negative x-direction
            i, j = slice(nb, nr), slice(nr, mj - nr)
            field[i, j] -= self._xneg[nb:, :mj_int] * (field[i, j] - field_ref[i, j])
            i, j = slice(nb, nr), slice(nb, nr)
            field[i, j] -= self._xnegyneg[nb:, nb:] * (field[i, j] - field_ref[i, j])
            i, j = slice(nb, nr), slice(mj - nr, mj - nb)
            field[i, j] -= self._xnegypos[nb:, :-nb] * (field[i, j] - field_ref[i, j])

            # apply the relaxed boundary conditions in the positive x-direction
            i, j = slice(mi - nr, mi - nb), slice(nr, mj - nr)
            field[i, j] -= self._xpos[:-nb, :mj_int] * (field[i, j] - field_ref[i, j])
            i, j = slice(mi - nr, mi - nb), slice(nb, nr)
            field[i, j] -= self._xposyneg[:-nb, nb:] * (field[i, j] - field_ref[i, j])
            i, j = slice(mi - nr, mi - nb), slice(mj - nr, mj - nb)
            field[i, j] -= self._xposypos[:-nb, :-nb] * (field[i, j] - field_ref[i, j])

            # apply the relaxed boundary conditions in the negative y-direction
            i, j = slice(nr, mi - nr), slice(nb, nr)
            field[i, j] -= self._yneg[:mi_int, nb:] * (field[i, j] - field_ref[i, j])

            # apply the relaxed boundary conditions in the positive y-direction
            i, j = slice(nr, mi - nr), slice(mj - nr, mj - nb)
            field[i, j] -= self._ypos[:mi_int, :-nb] * (field[i, j] - field_ref[i, j])
        else:
            mk = (
                self._kwargs["nz"] + 1
                if "on_interface_levels" in field_name
                else self._kwargs["nz"]
            )
            k = slice(0, mk)

            # set the outermost layers
            field[:nb, :, k] = field_ref[:nb, :, k]
            field[mi - nb : mi, :, k] = field_ref[mi - nb : mi, :, k]
            field[nb : mi - nb, :nb, k] = field_ref[nb : mi - nb, :nb, k]
            field[nb : mi - nb, mj - nb : mj, k] = field_ref[
                nb : mi - nb, mj - nb : mj, k
            ]

            # apply the relaxed boundary conditions in the negative x-direction
            i, j = slice(nb, nr), slice(nr, mj - nr)
            field[i, j, k] -= self._xneg[nb:, :mj_int, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
            i, j = slice(nb, nr), slice(nb, nr)
            field[i, j, k] -= self._xnegyneg[nb:, nb:, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
            i, j = slice(nb, nr), slice(mj - nr, mj - nb)
            field[i, j, k] -= self._xnegypos[nb:, :-nb, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )

            # apply the relaxed boundary conditions in the positive x-direction
            i, j = slice(mi - nr, mi - nb), slice(nr, mj - nr)
            field[i, j, k] -= self._xpos[:-nb, :mj_int, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
            i, j = slice(mi - nr, mi - nb), slice(nb, nr)
            field[i, j, k] -= self._xposyneg[:-nb, nb:, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )
            i, j = slice(mi - nr, mi - nb), slice(mj - nr, mj - nb)
            field[i, j, k] -= self._xposypos[:-nb, :-nb, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )

            # apply the relaxed boundary conditions in the negative y-direction
            i, j = slice(nr, mi - nr), slice(nb, nr)
            field[i, j, k] -= self._yneg[:mi_int, nb:, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )

            # apply the relaxed boundary conditions in the positive y-direction
            i, j = slice(nr, mi - nr), slice(mj - nr, mj - nb)
            field[i, j, k] -= self._ypos[:mi_int, :-nb, k] * (
                field[i, j, k] - field_ref[i, j, k]
            )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
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
        field_ref = self.reference_state[field_name].to_units(field_units).values
        field[0, :mj] = field_ref[0, :mj]
        field[mi - 1, :mj] = field_ref[mi - 1, :mj]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
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
        field_ref = self.reference_state[field_name].to_units(field_units).values
        field[:mi, 0] = field_ref[:mi, 0]
        field[:mi, mj - 1] = field_ref[:mi, mj - 1]

    def _allocate_coefficient_matrices(self):
        nx, ny = self.nx, self.ny
        nb, nr = self.nb, self._kwargs["nr"]
        backend, dtype = self._backend, self._dtype

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
        backend = backend or "numpy"
        if self._gt_powered:
            device = gt.backend.from_name(backend).storage_info["device"]
        else:
            device = "cpu"
        asarray = cp.asarray if device == "gpu" else np.asarray

        if not self._gt_powered:
            # make all matrices three-dimensional to harness array broadcasting
            self._xneg = asarray(xneg[:, :, np.newaxis])
            self._xpos = asarray(xpos[:, :, np.newaxis])
            self._yneg = asarray(yneg[:, :, np.newaxis])
            self._ypos = asarray(ypos[:, :, np.newaxis])
            self._xnegyneg = asarray(xnegyneg[:, :, np.newaxis])
            self._xnegypos = asarray(xnegypos[:, :, np.newaxis])
            self._xposyneg = asarray(xposyneg[:, :, np.newaxis])
            self._xposypos = asarray(xposypos[:, :, np.newaxis])
        else:
            nz = self._kwargs["nz"]

            if nz is None:
                self._xneg = None
                self._xpos = None
                self._yneg = None
                self._ypos = None
                self._xnegyneg = None
                self._xnegypos = None
                self._xposyneg = None
                self._xposypos = None
            else:
                # convert all matrices into gt storages
                self._xneg = zeros(
                    (nr, ny - 2 * nr + 1, nz + 1),
                    gt_powered=True,
                    backend=backend,
                    dtype=dtype,
                )
                self._xneg[...] = xneg[:, :, np.newaxis]
                self._xpos = zeros(
                    (nr, ny - 2 * nr + 1, nz + 1),
                    gt_powered=True,
                    backend=backend,
                    dtype=dtype,
                )
                self._xpos[...] = xpos[:, :, np.newaxis]

                self._yneg = zeros(
                    (nx - 2 * nr + 1, nr, nz + 1),
                    gt_powered=True,
                    backend=backend,
                    dtype=dtype,
                )
                self._yneg[...] = yneg[:, :, np.newaxis]
                self._ypos = zeros(
                    (nx - 2 * nr + 1, nr, nz + 1),
                    gt_powered=True,
                    backend=backend,
                    dtype=dtype,
                )
                self._ypos[...] = ypos[:, :, np.newaxis]

                self._xnegyneg = zeros(
                    (nr, nr, nz + 1), gt_powered=True, backend=backend, dtype=dtype
                )
                self._xnegyneg[...] = xnegyneg[:, :, np.newaxis]
                self._xnegypos = zeros(
                    (nr, nr, nz + 1), gt_powered=True, backend=backend, dtype=dtype
                )
                self._xnegypos[...] = xnegypos[:, :, np.newaxis]
                self._xposyneg = zeros(
                    (nr, nr, nz + 1), gt_powered=True, backend=backend, dtype=dtype
                )
                self._xposyneg[...] = xposyneg[:, :, np.newaxis]
                self._xposypos = zeros(
                    (nr, nr, nz + 1), gt_powered=True, backend=backend, dtype=dtype
                )
                self._xposypos[...] = xposypos[:, :, np.newaxis]


class Relaxed1DX(HorizontalBoundary):
    """ Relaxed boundary conditions for a physical grid with ``ny=1``. """

    def __init__(self, nx, ny, nb, gt_powered, backend, dtype, nr=8):
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
        gt_powered : bool
            ``True`` to harness GT4Py, ``False`` for a vanilla NumPy implementation.
        backend : str
            The GT4Py backend.
        dtype : data-type
            The data type of the storages.
        nr : `int`, optional
            Depth of each relaxation region close to the
            horizontal boundaries. Minimum is ``nb``, maximum is 8 (default).
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert ny == 1, "Number of grid points along second dimension must be 1."
        assert nr <= nx / 2, "Depth of relaxation region cannot exceed nx/2."
        assert nr <= 8, "Depth of relaxation region cannot exceed 8."
        assert (
            nb <= nr
        ), "Number of boundary layers cannot exceed depth of relaxation region."

        super().__init__(nx, ny, nb, gt_powered, backend, dtype)
        self._kwargs["nr"] = nr

        nr = nr if nb <= nr <= 8 else 8

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
        backend = backend or "numpy"
        device = gt.backend.from_name(backend).storage_info["device"]
        asarray = cp.asarray if device == "gpu" else np.asarray

        # made all matrices three-dimensional to harness array broadcasting
        self._xneg = asarray(xneg[:, :, np.newaxis])
        self._xpos = asarray(xpos[:, :, np.newaxis])

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
        nb, nr = self.nb, self._xneg.shape[0]
        field_name = field_name or ""

        # convenient definitions
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
        mj_int = mj - 2 * nb

        # the boundary values
        field_ref = self.reference_state[field_name].to_units(field_units).values
        # xneg = np.repeat(field_ref[0:1, nb:-nb], nr, axis=0)
        # xpos = np.repeat(field_ref[-1:, nb:-nb], nr, axis=0)

        # set the outermost layers
        field[:nb, nb : mj - nb] = field_ref[:nb, nb : mj - nb]
        field[mi - nb : mi, nb : mj - nb] = field_ref[mi - nb : mi, nb : mj - nb]

        # apply the relaxed boundary conditions in the negative x-direction
        i, j = slice(nb, nr), slice(nb, mj - nb)
        field[i, j] -= self._xneg[nb:, :mj_int] * (field[i, j] - field_ref[i, j])

        # apply the relaxed boundary conditions in the positive x-direction
        i, j = slice(mi - nr, mi - nb), slice(nb, mj - nb)
        field[i, j] -= self._xpos[:-nb, :mj_int] * (field[i, j] - field_ref[i, j])

        # repeat the innermost column(s) along the y-direction
        field[:mi, :nb] = field[:mi, nb : nb + 1]
        field[:mi, mj - nb : mj] = field[:mi, mj - nb - 1 : mj - nb]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
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
        field_ref = self.reference_state[field_name].to_units(field_units).values
        field[0, :mj] = field_ref[0, :mj]
        field[mi - 1, :mj] = field_ref[mi - 1, :mj]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
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
        field_ref = self.reference_state[field_name].to_units(field_units).values
        field[:mi, 0] = field_ref[:mi, 0]
        field[:mi, mj - 1] = field_ref[:mi, mj - 1]


class Relaxed1DY(HorizontalBoundary):
    """ Relaxed boundary conditions for a physical grid with ``nx=1``. """

    def __init__(self, nx, ny, nb, gt_powered, backend, dtype, nr=8):
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
        gt_powered : bool
            ``True`` to harness GT4Py, ``False`` for a vanilla NumPy implementation.
        backend : str
            The GT4Py backend.
        dtype : data-type
            The data type of the storages.
        nr : `int`, optional
            Depth of each relaxation region close to the
            horizontal boundaries. Minimum is ``nb``, maximum is 8 (default).
        """
        assert nx == 1, "Number of grid points along first dimension must be 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nr <= ny / 2, "Depth of relaxation region cannot exceed ny/2."
        assert nr <= 8, "Depth of relaxation region cannot exceed 8."
        assert (
            nb <= nr
        ), "Number of boundary layers cannot exceed depth of relaxation region."

        super().__init__(nx, ny, nb, gt_powered, backend, dtype)
        self._kwargs["nr"] = nr

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
        backend = backend or "numpy"
        device = gt.backend.from_name(backend).storage_info["device"]
        asarray = cp.asarray if device == "gpu" else np.asarray

        # made all matrices three-dimensional to harness array broadcasting
        self._yneg = asarray(yneg[:, :, np.newaxis])
        self._ypos = asarray(ypos[:, :, np.newaxis])

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
        nb, nr = self.nb, self._yneg.shape[1]
        field_name = field_name or ""

        # convenient definitions
        mi = (
            self.ni + 1
            if "at_u_locations" in field_name or "at_uv_locations" in field_name
            else self.ni
        )
        mi_int = mi - 2 * nb
        mj = (
            self.nj + 1
            if "at_v_locations" in field_name or "at_uv_locations" in field_name
            else self.nj
        )

        # the boundary values
        field_ref = self.reference_state[field_name].to_units(field_units).values
        # yneg = np.repeat(field_ref[nb:-nb, 0:1], nr, axis=1)
        # ypos = np.repeat(field_ref[nb:-nb, -1:], nr, axis=1)

        # set the outermost layers
        field[nb : mi - nb, :nb] = field_ref[nb : mi - nb, :nb]
        field[nb : mi - nb, mj - nb : mj] = field_ref[nb : mi - nb, mj - nb : mj]

        # apply the relaxed boundary conditions in the negative y-direction
        i, j = slice(nb, mi - nb), slice(nb, nr)
        field[i, j] -= self._yneg[:mi_int, nb:] * (field[i, j] - field_ref[i, j])

        # apply the relaxed boundary conditions in the positive y-direction
        i, j = slice(nb, mi - nb), slice(mj - nr, mj - nb)
        field[i, j] -= self._ypos[:mi_int, :-nb] * (field[i, j] - field_ref[i, j])

        # repeat the innermost row(s) along the x-direction
        field[:nb, :mj] = field[nb : nb + 1, :mj]
        field[mi - nb : mi, :mj] = field[mi - nb - 1 : mi - nb, :mj]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
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
        field_ref = self.reference_state[field_name].to_units(field_units).values
        field[0, :mj] = field_ref[0, :mj]
        field[mi - 1, :mj] = field_ref[mi - 1, :mj]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
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
        field_ref = self.reference_state[field_name].to_units(field_units).values
        field[:mi, 0] = field_ref[:mi, 0]
        field[:mi, mj - 1] = field_ref[:mi, mj - 1]


@register(name="relaxed", registry_class=HorizontalBoundary)
def dispatch(nx, ny, nb, gt_powered, backend="numpy", dtype=np.float64, nr=8, nz=None):
    """ Dispatch based on the grid size. """
    if nx == 1:
        return Relaxed1DY(1, ny, nb, gt_powered, backend, dtype, nr)
    elif ny == 1:
        return Relaxed1DX(nx, 1, nb, gt_powered, backend, dtype, nr)
    else:
        return Relaxed(nx, ny, nb, gt_powered, backend, dtype, nr, nz)
