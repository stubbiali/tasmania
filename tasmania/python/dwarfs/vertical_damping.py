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
import abc
import math
import numpy as np
from sympl import DataArray

import gridtools as gt
from tasmania.python.utils.storage_utils import zeros
from tasmania.python.utils.utils import greater_or_equal_than as ge

try:
    from tasmania.conf import datatype
except ImportError:
    from numpy import float32 as datatype


class VerticalDamping(abc.ABC):
    """
    Abstract base class whose derived classes implement different
    vertical damping, i.e., wave absorbing, techniques.
    """

    def __init__(
        self,
        grid,
        damp_depth,
        damp_coeff_max,
        time_units,
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        default_origin,
        rebuild,
        storage_shape,
    ):
        """
        Parameters
        ----------
        grid : tasmania.Grid
            The underlying grid.
        damp_depth : int
            Number of vertical layers in the damping region.
        damp_coeff_max : float
            Maximum value for the damping coefficient.
        time_units : str
            Time units to be used throughout the class.
        backend : str
            The GT4Py backend.
        backend_opts : dict
            Dictionary of backend-specific options.
        build_info : dict
            Dictionary of building options.
        dtype : numpy.dtype
            Data type of the storages.
        exec_info : dict
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : tuple
            Storage default origin.
        rebuild : bool
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        """
        # safety-guard checks
        assert damp_depth <= grid.nz, (
            "The depth of the damping region ({}) should be smaller or equal than "
            "the number of main vertical levels ({}).".format(damp_depth, grid.nz)
        )

        # store input arguments needed at run-time
        self._damp_depth = damp_depth
        self._tunits = time_units
        self._exec_info = exec_info
        storage_shape = (
            (grid.nx, grid.ny, grid.nz) if storage_shape is None else storage_shape
        )
        assert grid.nz <= storage_shape[2] <= grid.nz + 1
        self._shape = storage_shape

        # compute lower-bound of damping region
        lb = grid.z.values[damp_depth - 1]

        # compute the damping matrix
        z = (
            np.concatenate((grid.z.values, np.array([0])), axis=0)
            if storage_shape[2] == grid.nz + 1
            else grid.z.values
        )
        zt = grid.z_on_interface_levels.values[0]
        za = z[damp_depth - 1]
        r = ge(z, za) * damp_coeff_max * (1 - np.cos(math.pi * (z - za) / (zt - za)))
        self._rmat = zeros(
            storage_shape, backend, dtype, default_origin=default_origin, mask=(True, True, True)
        )
        self._rmat[...] = r[np.newaxis, np.newaxis, :]

        # instantiate the underlying stencil
        decorator = gt.stencil(
            backend, backend_opts=backend_opts, build_info=build_info, rebuild=rebuild
        )
        self._stencil = decorator(self._stencil_defs)

    @abc.abstractmethod
    def __call__(self, dt, field_now, field_new, field_ref, field_out):
        """
        Apply vertical damping to a generic field.
        As this method is marked as abstract, its implementation
        is delegated to the derived classes.

        Parameters
        ----------
        dt : timedelta
            The time step.
        field_now : gridtools.storage.Storage
            The field at the current time level.
        field_new : gridtools.storage.Storage
            The field at the next time level, on which the absorber will be applied.
        field_ref : gridtools.storage.Storage
            A reference value for the field.
        field_out : gridtools.storage.Storage
            Buffer into which writing the output, vertically damped field.
        """
        pass

    @staticmethod
    def factory(
        damp_type,
        grid,
        damp_depth,
        damp_coeff_max,
        time_units="s",
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None
    ):
        """
        Static method which returns an instance of the derived class
        implementing the damping method specified by :data:`damp_type`.

        Parameters
        ----------
        damp_type : str
            String specifying the damper to implement. Either:

                * 'rayleigh', for a Rayleigh damper.

        grid : tasmania.Grid
            The underlying grid.
        damp_depth : int
            Number of vertical layers in the damping region.
        damp_coeff_max : float
            Maximum value for the damping coefficient.
        time_units : `str`, optional
            Time units to be used throughout the class. Defaults to 's'.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `numpy.dtype`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple`, optional
            Storage default origin.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.

        Return
        ------
        obj :
            An instance of the appropriate derived class.
        """
        args = [
            grid,
            damp_depth,
            damp_coeff_max,
            time_units,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
        ]
        if damp_type == "rayleigh":
            return Rayleigh(*args)
        else:
            raise ValueError("Unknown damping scheme. Available options: " "rayleigh" ".")

    @staticmethod
    @abc.abstractmethod
    def _stencil_defs(
        in_phi_now: gt.storage.f64_sd,
        in_phi_new: gt.storage.f64_sd,
        in_phi_ref: gt.storage.f64_sd,
        in_rmat: gt.storage.f64_sd,
        out_phi: gt.storage.f64_sd,
        *,
        dt: float
    ):
        pass


class Rayleigh(VerticalDamping):
    """
    This class inherits	:class:`~tasmania.VerticalDamping`
    to implement a Rayleigh absorber.
    """

    def __init__(
        self,
        grid,
        damp_depth=15,
        damp_coeff_max=0.0002,
        time_units="s",
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None,
    ):
        super().__init__(
            grid,
            damp_depth,
            damp_coeff_max,
            time_units,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
        )

    def __call__(self, dt, field_now, field_new, field_ref, field_out):
        # shortcuts
        ni, nj, nk = self._shape
        dnk = self._damp_depth

        # convert the timestep to seconds
        dt_da = DataArray(dt.total_seconds(), attrs={"units": "s"})
        dt_raw = dt_da.to_units(self._tunits).values.item()

        if dnk > 0:
            # run the stencil
            self._stencil(
                in_phi_now=field_now,
                in_phi_new=field_new,
                in_phi_ref=field_ref,
                in_rmat=self._rmat,
                out_phi=field_out,
                dt=dt_raw,
                origin={"_all_": (0, 0, 0)},
                domain=(ni, nj, dnk),
                exec_info=self._exec_info,
            )

        # set the lowermost layers, outside of the damping region
        field_out[:, :, dnk:] = field_new[:, :, dnk:]

    @staticmethod
    def _stencil_defs(
        in_phi_now: gt.storage.f64_sd,
        in_phi_new: gt.storage.f64_sd,
        in_phi_ref: gt.storage.f64_sd,
        in_rmat: gt.storage.f64_sd,
        out_phi: gt.storage.f64_sd,
        *,
        dt: float
    ):
        out_phi = in_phi_new[0, 0, 0] - dt * in_rmat[0, 0, 0] * (
            in_phi_now[0, 0, 0] - in_phi_ref[0, 0, 0]
        )
