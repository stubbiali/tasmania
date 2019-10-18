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
"""
This module contains:
    Relaxed(HorizontalBoundary)
    RelaxedXZ(HorizontalBoundary)
    RelaxedYZ(HorizontalBoundary)
"""
import cupy as cp
import inspect
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.grids.horizontal_boundary import HorizontalBoundary


def repeat_axis(paxis, nb, dims):
    pvalues = paxis.values
    dims = dims if dims is not None else paxis.dims[0]
    name = paxis.name
    attrs = paxis.attrs
    dtype = paxis.dtype

    if pvalues[0] <= pvalues[-1]:
        padneg = np.array(tuple(pvalues[0] - nb + i for i in range(nb)), dtype=dtype)
        padpos = np.array(tuple(pvalues[-1] + i + 1 for i in range(nb)), dtype=dtype)
    else:
        padneg = np.array(tuple(pvalues[0] + nb - i for i in range(nb)), dtype=dtype)
        padpos = np.array(tuple(pvalues[-1] - i - 1 for i in range(nb)), dtype=dtype)

    cvalues = np.concatenate((padneg, pvalues, padpos), axis=0)

    return DataArray(cvalues, coords=[cvalues], dims=dims, name=name, attrs=attrs)


def shrink_axis(caxis, nb, dims):
    cvalues = caxis.values
    dims = dims if dims is not None else caxis.dims[0]
    name = caxis.name
    attrs = caxis.attrs

    pvalues = cvalues[nb:-nb]

    return DataArray(pvalues, coords=[pvalues], dims=dims, name=name, attrs=attrs)


class Relaxed(HorizontalBoundary):
    """
    Relaxed boundary conditions.
    """

    def __init__(self, nx, ny, nb, nr=8):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension.
        nb : `int`, optional
            Number of boundary layers.
        nr : `int`, optional
            Depth of the each relaxation region close to the
            horizontal boundaries. Minimum is `nb`, maximum is 8 (default).
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

        super().__init__(nx, ny, nb)
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
        xneg = np.repeat(rel[:, np.newaxis], ny - 2 * nr + 1, axis=1)
        xpos = np.repeat(rrel[:, np.newaxis], ny - 2 * nr + 1, axis=1)
        yneg = np.repeat(rel[np.newaxis, :], nx - 2 * nr + 1, axis=0)
        ypos = np.repeat(rrel[np.newaxis, :], nx - 2 * nr + 1, axis=0)

        # the corner relaxation matrices
        xnegyneg = np.zeros((nr, nr))
        for i in range(nr):
            xnegyneg[i, i:] = rel[i]
            xnegyneg[i:, i] = rel[i]
        xposyneg = xnegyneg[:, ::-1]
        xposypos = np.transpose(xnegyneg)
        xnegypos = np.transpose(xposyneg)

        # made all matrices three-dimensional to harness numpy's broadcasting
        self._xneg = xneg[:, :, np.newaxis]
        self._xpos = xpos[:, :, np.newaxis]
        self._yneg = yneg[:, :, np.newaxis]
        self._ypos = ypos[:, :, np.newaxis]
        self._xnegyneg = xnegyneg[:, :, np.newaxis]
        self._xnegypos = xnegypos[:, :, np.newaxis]
        self._xposyneg = xposyneg[:, :, np.newaxis]
        self._xposypos = xposypos[:, :, np.newaxis]

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
        return field

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
        return field

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        # shortcuts
        nb, nr = self.nb, self._xneg.shape[0]
        mi, mj, mk = field.shape

        # convenient definitions
        mi_int = mi - 2 * nr
        mj_int = mj - 2 * nr

        # the boundary values
        field_ref = (
            self.reference_state[field_name].to_units(field_units).values[:mi, :mj, :mk]
        )
        xneg = np.repeat(field_ref[0:1, :], nr, axis=0)
        xpos = np.repeat(field_ref[-1:, :], nr, axis=0)
        yneg = np.repeat(field_ref[:, 0:1], nr, axis=1)
        ypos = np.repeat(field_ref[:, -1:], nr, axis=1)

        # set the outermost layers
        field[:nb, nb:-nb] = field_ref[:nb, nb:-nb]
        field[-nb:, nb:-nb] = field_ref[-nb:, nb:-nb]
        field[:, :nb] = field_ref[:, :nb]
        field[:, -nb:] = field_ref[:, -nb:]

        # apply the relaxed boundary conditions in the negative x-direction
        field[nb:nr, nr:-nr] -= self._xneg[nb:, :mj_int] * (
            field[nb:nr, nr:-nr] - xneg[nb:, nr:-nr]
        )
        field[nb:nr, nb:nr] -= self._xnegyneg[nb:, nb:] * (
            field[nb:nr, nb:nr] - xneg[nb:, nb:nr]
        )
        field[nb:nr, -nr:-nb] -= self._xnegypos[nb:, :-nb] * (
            field[nb:nr, -nr:-nb] - xneg[nb:, -nr:-nb]
        )

        # apply the relaxed boundary conditions in the positive x-direction
        field[-nr:-nb, nr:-nr] -= self._xpos[:-nb, :mj_int] * (
            field[-nr:-nb, nr:-nr] - xpos[:-nb, nr:-nr]
        )
        field[-nr:-nb, nb:nr, :] -= self._xposyneg[:-nb, :-nb] * (
            field[-nr:-nb, nb:nr] - xpos[:-nb, nb:nr]
        )
        field[-nr:-nb, -nr:-nb] -= self._xposypos[:-nb, nb:] * (
            field[-nr:-nb, -nr:-nb] - xpos[:-nb, -nr:-nb]
        )

        # apply the relaxed boundary conditions in the negative y-direction
        field[nr:-nr, nb:nr] -= self._yneg[:mi_int, nb:] * (
            field[nr:-nr, nb:nr] - yneg[nr:-nr, nb:]
        )

        # apply the relaxed boundary conditions in the positive y-direction
        field[nr:-nr, -nr:-nb] -= self._ypos[:mi_int, :-nb] * (
            field[nr:-nr, -nr:-nb] - ypos[nr:-nr, :-nb]
        )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        mi, mj, mk = field.shape
        field_ref = (
            self.reference_state[field_name].to_units(field_units).values[:mi, :mj, :mk]
        )
        field[0, :] = field_ref[0, :]
        field[-1, :] = field_ref[-1, :]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        mi, mj, mk = field.shape
        field_ref = (
            self.reference_state[field_name].to_units(field_units).values[:mi, :mj, :mk]
        )
        field[:, 0] = field_ref[:, 0]
        field[:, -1] = field_ref[:, -1]


class Relaxed1DX(HorizontalBoundary):
    """
    Relaxed boundary conditions on a grid with only one point
    along the second horizontal dimension.
    """

    def __init__(self, nx, ny, nb, nr=8):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension. It must be 1.
        nb : int
            Number of boundary layers.
        nr : `int`, optional
            Depth of the each relaxation region close to the
            horizontal boundaries. Minimum is `nb`, maximum is 8 (default).
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

        super().__init__(nx, ny, nb)
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

        # made all matrices three-dimensional to harness numpy's broadcasting
        self._xneg = xneg[:, :, np.newaxis]
        self._xpos = xpos[:, :, np.newaxis]

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
        padneg = np.repeat(field[:, 0:1], self.nb, axis=1)
        padpos = np.repeat(field[:, -1:], self.nb, axis=1)
        return np.concatenate((padneg, field, padpos), axis=1)

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
        return field[:, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        # shortcuts
        nb, nr = self.nb, self._xneg.shape[0]
        mi, mj, mk = field.shape
        lj = mj - 2 * nb

        # the boundary values
        field_ref = (
            self.reference_state[field_name].to_units(field_units).values[:mi, :mj, :mk]
        )
        xneg = np.repeat(field_ref[0:1, nb:-nb], nr, axis=0)
        xpos = np.repeat(field_ref[-1:, nb:-nb], nr, axis=0)

        # set the outermost layers
        field[:nb, nb:-nb] = field_ref[:nb, nb:-nb]
        field[-nb:, nb:-nb] = field_ref[-nb:, nb:-nb]

        # apply the relaxed boundary conditions in the negative x-direction
        field[nb:nr, nb:-nb] -= self._xneg[nb:, :lj] * (
            field[nb:nr, nb:-nb] - xneg[nb:, :]
        )

        # apply the relaxed boundary conditions in the positive x-direction
        field[-nr:-nb, nb:-nb] -= self._xpos[:-nb, :lj] * (
            field[-nr:-nb, nb:-nb] - xpos[:-nb, :]
        )

        # repeat the innermost column(s) along the y-direction
        field[:, :nb] = field[:, nb : nb + 1]
        field[:, -nb:] = field[:, -nb - 1 : -nb]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        mi, mj, mk = field.shape
        field_ref = (
            self.reference_state[field_name].to_units(field_units).values[:mi, :mj, :mk]
        )
        field[0, :] = field_ref[0, :]
        field[-1, :] = field_ref[-1, :]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        mi, mj, mk = field.shape
        field_ref = (
            self.reference_state[field_name].to_units(field_units).values[:mi, :mj, :mk]
        )
        field[:, 0] = field_ref[:, 0]
        field[:, -1] = field_ref[:, -1]


class Relaxed1DY(HorizontalBoundary):
    """
    Relaxed boundary conditions on a grid with only one point
    along the first horizontal dimension.
    """

    def __init__(self, nx, ny, nb, nr=8):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension. It must be 1.
        nb : int
            Number of boundary layers.
        nr : `int`, optional
            Depth of the each relaxation region close to the
            horizontal boundaries. Minimum is `nb`, maximum is 8 (default).
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

        super().__init__(nx, ny, nb)
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

        # made all matrices three-dimensional to harness numpy's broadcasting
        self._yneg = yneg[:, :, np.newaxis]
        self._ypos = ypos[:, :, np.newaxis]

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
        padneg = np.repeat(field[0:1, :], self.nb, axis=0)
        padpos = np.repeat(field[-1:, :], self.nb, axis=0)
        return np.concatenate((padneg, field, padpos), axis=0)

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
        return field[self.nb : -self.nb, :]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        # shortcuts
        nb, nr = self.nb, self._yneg.shape[1]
        mi, mj, mk = field.shape
        li = mi - 2 * nb

        # the boundary values
        field_ref = (
            self.reference_state[field_name].to_units(field_units).values[:mi, :mj, :mk]
        )
        yneg = np.repeat(field_ref[nb:-nb, 0:1], nr, axis=1)
        ypos = np.repeat(field_ref[nb:-nb, -1:], nr, axis=1)

        # set the outermost layers
        field[nb:-nb, :nb] = field_ref[nb:-nb, :nb]
        field[nb:-nb, -nb:] = field_ref[nb:-nb, -nb:]

        # apply the relaxed boundary conditions in the negative y-direction
        field[nb:-nb, nb:nr] -= self._yneg[:li, nb:] * (
            field[nb:-nb, nb:nr] - yneg[:, nb:]
        )

        # apply the relaxed boundary conditions in the positive y-direction
        field[nb:-nb, -nr:-nb] -= self._ypos[:li, :-nb] * (
            field[nb:-nb, -nr:-nb] - ypos[:, :-nb]
        )

        # repeat the innermost row(s) along the x-direction
        field[:nb, :] = field[nb : nb + 1, :]
        field[-nb:, :] = field[-nb - 1 : -nb, :]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        mi, mj, mk = field.shape
        field_ref = (
            self.reference_state[field_name].to_units(field_units).values[:mi, :mj, :mk]
        )
        field[0, :] = field_ref[0, :]
        field[-1, :] = field_ref[-1, :]

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        mi, mj, mk = field.shape
        field_ref = (
            self.reference_state[field_name].to_units(field_units).values[:mi, :mj, :mk]
        )
        field[:, 0] = field_ref[:, 0]
        field[:, -1] = field_ref[:, -1]


class Periodic(HorizontalBoundary):
    """
    Periodic boundary conditions.
    """

    def __init__(self, nx, ny, nb):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension. It must be 1.
        nb : int
            Number of boundary layers.
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed ny/2."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(nx, ny, nb)

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

        try:
            mi, mj, mk = field.shape
            cfield = np.zeros((mi + 2 * nb, mj + 2 * nb, mk), dtype=dtype)
        except ValueError:
            mi, mj = field.shape
            cfield = np.zeros((mi + 2 * nb, mj + 2 * nb), dtype=dtype)

        cfield[nb:-nb, nb:-nb] = field[:, :]
        cfield[:nb, nb:-nb] = cfield[nx - 1 : nx - 1 + nb, nb:-nb]
        cfield[-nb:, nb:-nb] = (
            cfield[nb + 1 : 2 * nb + 1, nb:-nb]
            if mi == nx
            else cfield[nb + 2 : 2 * nb + 2, nb:-nb]
        )
        cfield[:, :nb] = cfield[:, ny - 1 : ny - 1 + nb]
        cfield[:, -nb:] = (
            cfield[:, nb + 1 : 2 * nb + 1] if mj == ny else cfield[:, nb + 2 : 2 * nb + 2]
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
        mi = field.shape[0] - 2 * nb
        mj = field.shape[1] - 2 * nb

        field[:nb, nb:-nb] = field[nx - 1 : nx - 1 + nb, nb:-nb]
        field[-nb:, nb:-nb] = (
            field[nb + 1 : 2 * nb + 1, nb:-nb]
            if mi == nx
            else field[nb + 2 : 2 * nb + 2, nb:-nb]
        )
        field[:, :nb] = field[:, ny - 1 : ny - 1 + nb]
        field[:, -nb:] = (
            field[:, nb + 1 : 2 * nb + 1] if mj == ny else field[:, nb + 2 : 2 * nb + 2]
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
    """
    Periodic boundary conditions on a grid with only one point
    along the second horizontal dimension.
    """

    def __init__(self, nx, ny, nb):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension. It must be 1.
        nb : int
            Number of boundary layers.
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert ny == 1, "Number of grid points along second dimension must be 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed nx/2."

        super().__init__(nx, ny, nb)

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

        try:
            mi, mj, mk = field.shape
            cfield = np.zeros((mi + 2 * nb, mj + 2 * nb, mk), dtype=dtype)
        except ValueError:
            mi, mj = field.shape
            cfield = np.zeros((mi + 2 * nb, mj + 2 * nb), dtype=dtype)

        cfield[nb:-nb, nb:-nb] = field[:, :]
        cfield[:nb, nb:-nb] = cfield[nx - 1 : nx - 1 + nb, nb:-nb]
        cfield[-nb:, nb:-nb] = (
            cfield[nb + 1 : 2 * nb + 1, nb:-nb]
            if mi == nx
            else cfield[nb + 2 : 2 * nb + 2, nb:-nb]
        )
        cfield[:, :nb] = cfield[:, nb : nb + 1]
        cfield[:, -nb:] = cfield[:, -nb - 1 : -nb]

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
        mi = field.shape[0] - 2 * nb

        field[:nb, nb:-nb] = field[nx - 1 : nx - 1 + nb, nb:-nb]
        field[-nb:, nb:-nb] = (
            field[nb + 1 : 2 * nb + 1, nb:-nb]
            if mi == nx
            else field[nb + 2 : 2 * nb + 2, nb:-nb]
        )
        field[:, :nb] = field[:, nb : nb + 1]
        field[:, -nb:] = field[:, -nb - 1 : -nb]

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
    """
    Periodic boundary conditions on a grid with only one point
    along the first horizontal dimension.
    """

    def __init__(self, nx, ny, nb):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension. It must be 1.
        nb : int
            Number of boundary layers.
        """
        assert nx == 1, "Number of grid points along first dimension must be 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(nx, ny, nb)

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

        try:
            mi, mj, mk = field.shape
            cfield = np.zeros((mi + 2 * nb, mj + 2 * nb, mk), dtype=dtype)
        except ValueError:
            mi, mj = field.shape
            cfield = np.zeros((mi + 2 * nb, mj + 2 * nb), dtype=dtype)

        cfield[nb:-nb, nb:-nb] = field[:, :]
        cfield[nb:-nb, :nb] = cfield[nb:-nb, ny - 1 : ny - 1 + nb]
        cfield[nb:-nb, -nb:] = (
            cfield[nb:-nb, nb + 1 : 2 * nb + 1]
            if mj == ny
            else cfield[nb:-nb, nb + 2 : 2 * nb + 2]
        )
        cfield[:nb, :] = cfield[nb : nb + 1, :]
        cfield[-nb:, :] = cfield[-nb - 1 : -nb, :]

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
        mj = field.shape[1] - 2 * nb

        field[nb:-nb, :nb] = field[nb:-nb, ny - 1 : ny - 1 + nb]
        field[nb:-nb, -nb:] = (
            field[nb:-nb, nb + 1 : 2 * nb + 1]
            if mj == ny
            else field[nb:-nb, nb + 2 : 2 * nb + 2]
        )
        field[:nb, :] = field[nb : nb + 1, :]
        field[-nb:, :] = field[-nb - 1 : -nb, :]

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


def placeholder(time, grid, slice_x, slice_y, field_name):
    pass


class Dirichlet(HorizontalBoundary):
    """
    Dirichlet boundary conditions.
    """

    def __init__(self, nx, ny, nb, core=placeholder):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension.
        nb : int
            Number of boundary layers.
        core : `callable`, optional
            Callable object actually providing the boundary layers values.
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
            "core(time, grid, slice_x=None, slice_y=None, field_name=None)"
        )
        assert tuple(signature.parameters.keys())[0] == "time", error_msg
        assert tuple(signature.parameters.keys())[1] == "grid", error_msg
        assert "slice_x" in signature.parameters, error_msg
        assert "slice_y" in signature.parameters, error_msg
        assert "field_name" in signature.parameters, error_msg

        super().__init__(nx, ny, nb)

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
        return field

    def get_physical_xaxis(self, caxis, dims=None):
        return caxis

    def get_physical_yaxis(self, caxis, dims=None):
        return caxis

    def get_physical_field(self, field, field_name=None):
        return field

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        nb, core = self.nb, self._kwargs["core"]
        mi, mj = field.shape[0], field.shape[1]

        if isinstance(field, gt.storage.storage.GPUStorage):
            field[:nb, :] = cp.asarray(
                core(time, grid, slice(0, nb), slice(0, mj), field_name, field_units)
            )
            field[-nb:, :] = cp.asarray(
                core(
                    time, grid, slice(mi - nb, mi), slice(0, mj), field_name, field_units
                )
            )
            field[nb:-nb, :nb] = cp.asarray(
                core(
                    time, grid, slice(nb, mi - nb), slice(0, nb), field_name, field_units
                )
            )
            field[nb:-nb, -nb:] = cp.asarray(
                core(
                    time,
                    grid,
                    slice(nb, mi - nb),
                    slice(mj - nb, mj),
                    field_name,
                    field_units,
                )
            )
        else:
            field[:nb, :] = core(
                time, grid, slice(0, nb), slice(0, mj), field_name, field_units
            )
            field[-nb:, :] = core(
                time, grid, slice(mi - nb, mi), slice(0, mj), field_name, field_units
            )
            field[nb:-nb, :nb] = core(
                time, grid, slice(nb, mi - nb), slice(0, nb), field_name, field_units
            )
            field[nb:-nb, -nb:] = core(
                time,
                grid,
                slice(nb, mi - nb),
                slice(mj - nb, mj),
                field_name,
                field_units,
            )

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]
        mi, mj = field.shape[0], field.shape[1]

        field[:1, :] = core(
            time, grid, slice(0, 1), slice(0, mj), field_name, field_units
        )
        field[-1:, :] = core(
            time, grid, slice(mi - 1, mi), slice(0, mj), field_name, field_units
        )

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]
        mi, mj = field.shape[0], field.shape[1]

        field[:, :1] = core(
            time, grid, slice(0, mi), slice(0, 1), field_name, field_units
        )
        field[:, -1:] = core(
            time, grid, slice(0, mi), slice(mj - 1, mj), field_name, field_units
        )


class Dirichlet1DX(HorizontalBoundary):
    """
    Dirichlet boundary conditions on a grid with only one point
    along the second horizontal dimension.
    """

    def __init__(self, nx, ny, nb, core=placeholder):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension. It must be 1.
        nb : int
            Number of boundary layers.
        core : `callable`, optional
            Callable object actually providing the boundary layers values.
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert ny == 1, "Number of grid points along second dimension must be 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed nx/2."

        signature = inspect.signature(core)
        error_msg = (
            "The signature of the core function should be "
            "core(time, grid, slice_x=None, slice_y=None, field_name=None)"
        )
        assert tuple(signature.parameters.keys())[0] == "time", error_msg
        assert tuple(signature.parameters.keys())[1] == "grid", error_msg
        assert "slice_x" in signature.parameters, error_msg
        assert "slice_y" in signature.parameters, error_msg
        assert "field_name" in signature.parameters, error_msg

        super().__init__(nx, ny, nb)

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
        padneg = np.repeat(field[:, 0:1], self.nb, axis=1)
        padpos = np.repeat(field[:, -1:], self.nb, axis=1)
        return np.concatenate((padneg, field, padpos), axis=1)

    def get_physical_xaxis(self, caxis, dims=None):
        return caxis

    def get_physical_yaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_field(self, field, field_name=None):
        return field[:, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        nb, core = self.nb, self._kwargs["core"]
        mi, mj = field.shape[0], field.shape[1]

        field[:nb, nb:-nb] = core(
            time, grid, slice(0, nb), slice(nb, mj - nb), field_name, field_units
        )
        field[-nb:, nb:-nb] = core(
            time, grid, slice(mi - nb, mi), slice(nb, mj - nb), field_name, field_units
        )

        field[:, :nb] = field[:, nb : nb + 1]
        field[:, -nb:] = field[:, -nb - 1 : -nb]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]
        mi, mj = field.shape[0], field.shape[1]

        field[:1, :] = core(
            time, grid, slice(0, 1), slice(0, mj), field_name, field_units
        )
        field[-1:, :] = core(
            time, grid, slice(mi - 1, mi), slice(0, mj), field_name, field_units
        )

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]
        mi, mj = field.shape[0], field.shape[1]

        field[:, :1] = core(
            time, grid, slice(0, mi), slice(0, 1), field_name, field_units
        )
        field[:, -1:] = core(
            time, grid, slice(0, mi), slice(mj - 1, mj), field_name, field_units
        )


class Dirichlet1DY(HorizontalBoundary):
    """
    Dirichlet boundary conditions on a grid with only one point
    along the first horizontal dimension.
    """

    def __init__(self, nx, ny, nb, core=placeholder):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension. It must be 1.
        nb : int
            Number of boundary layers.
        core : `callable`, optional
            Callable object actually providing the boundary layers values.
        """
        assert nx == 1, "Number of grid points along first dimension must be 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        signature = inspect.signature(core)
        error_msg = (
            "The signature of the core function should be "
            "core(time, grid, slice_x=None, slice_y=None, field_name=None)"
        )
        assert tuple(signature.parameters.keys())[0] == "time", error_msg
        assert tuple(signature.parameters.keys())[1] == "grid", error_msg
        assert "slice_x" in signature.parameters, error_msg
        assert "slice_y" in signature.parameters, error_msg
        assert "field_name" in signature.parameters, error_msg

        super().__init__(nx, ny, nb)

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
        padneg = np.repeat(field[0:1, :], self.nb, axis=0)
        padpos = np.repeat(field[-1:, :], self.nb, axis=0)
        return np.concatenate((padneg, field, padpos), axis=0)

    def get_physical_xaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_yaxis(self, caxis, dims=None):
        return caxis

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, :]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        nb, core = self.nb, self._kwargs["core"]
        mi, mj = field.shape[0], field.shape[1]

        field[nb:-nb, :nb] = core(
            time, grid, slice(nb, mi - nb), slice(0, nb), field_name, field_units
        )
        field[nb:-nb, -nb:] = core(
            time, grid, slice(nb, mi - nb), slice(mj - nb, mj), field_name, field_units
        )

        field[:nb, :] = field[nb : nb + 1, :]
        field[-nb:, :] = field[-nb - 1 : -nb, :]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]
        mi, mj = field.shape[0], field.shape[1]

        field[:1, :] = core(
            time, grid, slice(0, 1), slice(0, mj), field_name, field_units
        )
        field[-1:, :] = core(
            time, grid, slice(mi - 1, mi), slice(0, mj), field_name, field_units
        )

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        core = self._kwargs["core"]
        mi, mj = field.shape[0], field.shape[1]

        field[:, :1] = core(
            time, grid, slice(0, mi), slice(0, 1), field_name, field_units
        )
        field[:, -1:] = core(
            time, grid, slice(0, mi), slice(mj - 1, mj), field_name, field_units
        )


class Identity(HorizontalBoundary):
    """
    *Identity* boundary conditions.
    """

    def __init__(self, nx, ny, nb):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension.
        nb : int
            Number of boundary layers.
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed ny/2."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(nx, ny, nb)

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
        return field

    def get_physical_xaxis(self, caxis, dims=None):
        return caxis

    def get_physical_yaxis(self, caxis, dims=None):
        return caxis

    def get_physical_field(self, field, field_name=None):
        return field

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
    """
    *Identity* boundary conditions on a grid with only one point
    along the second horizontal dimension.
    """

    def __init__(self, nx, ny, nb):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension. It must be 1.
        nb : int
            Number of boundary layers.
        """
        assert (
            nx > 1
        ), "Number of grid points along first dimension should be larger than 1."
        assert ny == 1, "Number of grid points along second dimension must be 1."
        assert nb <= nx / 2, "Number of boundary layers cannot exceed nx/2."

        super().__init__(nx, ny, nb)

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
        padneg = np.repeat(field[:, 0:1], self.nb, axis=1)
        padpos = np.repeat(field[:, -1:], self.nb, axis=1)
        return np.concatenate((padneg, field, padpos), axis=1)

    def get_physical_xaxis(self, caxis, dims=None):
        return caxis

    def get_physical_yaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_field(self, field, field_name=None):
        return field[:, self.nb : -self.nb]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        nb = self.nb
        field[:, :nb] = field[:, nb : nb + 1]
        field[:, -nb:] = field[:, -nb - 1 : -nb]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass


class Identity1DY(HorizontalBoundary):
    """
    *Identity* boundary conditions on a grid with only one point
    along the first horizontal dimension.
    """

    def __init__(self, nx, ny, nb):
        """
        Parameters
        ----------
        nx : int
            Number of points featured by the *physical* grid
            along the first horizontal dimension.
        ny : int
            Number of points featured by the *physical* grid
            along the second horizontal dimension. It must be 1.
        nb : int
            Number of boundary layers.
        """
        assert nx == 1, "Number of grid points along first dimension must be 1."
        assert (
            ny > 1
        ), "Number of grid points along second dimension should be larger than 1."
        assert nb <= ny / 2, "Number of boundary layers cannot exceed ny/2."

        super().__init__(nx, ny, nb)

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
        padneg = np.repeat(field[0:1, :], self.nb, axis=0)
        padpos = np.repeat(field[-1:, :], self.nb, axis=0)
        return np.concatenate((padneg, field, padpos), axis=0)

    def get_physical_xaxis(self, caxis, dims=None):
        return shrink_axis(caxis, self.nb, dims)

    def get_physical_yaxis(self, caxis, dims=None):
        return caxis

    def get_physical_field(self, field, field_name=None):
        return field[self.nb : -self.nb, :]

    def enforce_field(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        nb = self.nb
        field[:nb, :] = field[nb : nb + 1, :]
        field[-nb:, :] = field[-nb - 1 : -nb, :]

    def set_outermost_layers_x(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass

    def set_outermost_layers_y(
        self, field, field_name=None, field_units=None, time=None, grid=None
    ):
        pass
