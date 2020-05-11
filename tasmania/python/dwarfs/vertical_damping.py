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
from typing import Optional, TYPE_CHECKING

from gt4py import gtscript

# from gt4py.__gtscript__ import computation, interval, PARALLEL

from tasmania.python.utils import taz_types
from tasmania.python.utils.gtscript_utils import stencil_copy_defs
from tasmania.python.utils.storage_utils import get_asarray_function, zeros
from tasmania.python.utils.utils import greater_or_equal_than as ge

if TYPE_CHECKING:
    from tasmania.python.domain.grid import Grid


class VerticalDamping(abc.ABC):
    """
    Abstract base class whose derived classes implement different
    vertical damping, i.e., wave absorbing, techniques.
    """

    def __init__(
        self,
        grid: "Grid",
        damp_depth: int,
        damp_coeff_max: float,
        time_units: str,
        gt_powered: bool,
        backend: str,
        backend_opts: taz_types.options_dict_t,
        build_info: taz_types.options_dict_t,
        dtype: taz_types.dtype_t,
        exec_info: taz_types.mutable_options_dict_t,
        default_origin: taz_types.triplet_int_t,
        rebuild: bool,
        storage_shape: taz_types.triplet_int_t,
        managed_memory: bool,
    ) -> None:
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
        gt_powered : bool
            `True` to harness GT4Py, `False` for a vanilla Numpy implementation.
        backend : str
            The GT4Py backend.
        backend_opts : dict
            Dictionary of backend-specific options.
        build_info : dict
            Dictionary of building options.
        dtype : data-type
            Data type of the storages.
        exec_info : dict
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : tuple[int]
            Storage default origin.
        rebuild : bool
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.
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
            storage_shape,
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            mask=(True, True, True),
            managed_memory=managed_memory,
        )
        asarray = get_asarray_function(gt_powered, backend)
        self._rmat[...] = asarray(r[np.newaxis, np.newaxis, :])

        # instantiate the underlying stencil
        self._gt_powered = gt_powered
        if gt_powered:
            self._stencil = gtscript.stencil(
                definition=self._stencil_gt_defs,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                **(backend_opts or {})
            )
        else:
            self._stencil = self._stencil_numpy

    @abc.abstractmethod
    def __call__(
        self,
        dt: taz_types.timedelta_t,
        field_now: taz_types.gtstorage_t,
        field_new: taz_types.gtstorage_t,
        field_ref: taz_types.gtstorage_t,
        field_out: taz_types.gtstorage_t,
    ) -> None:
        """
        Apply vertical damping to a generic field.
        As this method is marked as abstract, its implementation
        is delegated to the derived classes.

        Parameters
        ----------
        dt : datetime.timedelta
            The time step.
        field_now : gt4py.storage.storage.Storage
            The field at the current time level.
        field_new : gt4py.storage.storage.Storage
            The field at the next time level, on which the absorber will be applied.
        field_ref : gt4py.storage.storage.Storage
            A reference value for the field.
        field_out : gt4py.storage.storage.Storage
            Buffer into which the output vertically damped field is written.
        """
        pass

    @staticmethod
    def factory(
        damp_type: str,
        grid: "Grid",
        damp_depth: int,
        damp_coeff_max: float,
        time_units: str = "s",
        gt_powered: bool = True,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        storage_shape: Optional[taz_types.triplet_int_t] = None,
        managed_memory: bool = False
    ) -> "VerticalDamping":
        """
        Static method which returns an instance of the derived class
        implementing the damping method specified by `damp_type`.

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
        gt_powered : `bool`, optional
            `True` to harness GT4Py, `False` for a vanilla Numpy implementation.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple[int]`, optional
            Storage default origin.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.

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
            gt_powered,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
            managed_memory,
        ]
        if damp_type == "rayleigh":
            return Rayleigh(*args)
        else:
            raise ValueError(
                "Unknown damping scheme. Available options: " "rayleigh" "."
            )

    @staticmethod
    @abc.abstractmethod
    def _stencil_numpy(
        in_phi_now: np.ndarray,
        in_phi_new: np.ndarray,
        in_phi_ref: np.ndarray,
        in_rmat: np.ndarray,
        out_phi: np.ndarray,
        *,
        dt: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        pass

    @staticmethod
    @abc.abstractmethod
    def _stencil_gt_defs(
        in_phi_now: gtscript.Field["dtype"],
        in_phi_new: gtscript.Field["dtype"],
        in_phi_ref: gtscript.Field["dtype"],
        in_rmat: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dt: float
    ) -> None:
        pass


class Rayleigh(VerticalDamping):
    """
    This class inherits	:class:`tasmania.VerticalDamping`
    to implement a Rayleigh absorber.
    """

    def __init__(
        self,
        grid,
        damp_depth=15,
        damp_coeff_max=0.0002,
        time_units="s",
        gt_powered=True,
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
        super().__init__(
            grid,
            damp_depth,
            damp_coeff_max,
            time_units,
            gt_powered,
            backend,
            backend_opts,
            build_info,
            dtype,
            exec_info,
            default_origin,
            rebuild,
            storage_shape,
            managed_memory,
        )
        if gt_powered:
            self._stencil_copy = gtscript.stencil(
                definition=stencil_copy_defs,
                backend=backend,
                build_info=build_info,
                rebuild=rebuild,
                dtypes={"dtype": dtype},
                **(backend_opts or {})
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
                origin=(0, 0, 0),
                domain=(ni, nj, dnk),
                exec_info=self._exec_info,
            )

        # set the lowermost layers, outside of the damping region
        if not self._gt_powered:
            field_out[:, :, dnk:] = field_new[:, :, dnk:]
        else:
            self._stencil_copy(
                field_new, field_out, origin=(0, 0, dnk), domain=(ni, nj, nk - dnk)
            )

    @staticmethod
    def _stencil_numpy(
        in_phi_now: np.ndarray,
        in_phi_new: np.ndarray,
        in_phi_ref: np.ndarray,
        in_rmat: np.ndarray,
        out_phi: np.ndarray,
        *,
        dt: float,
        origin: taz_types.triplet_int_t,
        domain: taz_types.triplet_int_t,
        **kwargs  # catch-all
    ) -> None:
        i = slice(origin[0], origin[0] + domain[0])
        j = slice(origin[1], origin[1] + domain[1])
        k = slice(origin[2], origin[2] + domain[2])

        out_phi[i, j, k] = in_phi_new[i, j, k] - dt * in_rmat[i, j, k] * (
            in_phi_now[i, j, k] - in_phi_ref[i, j, k]
        )

    @staticmethod
    def _stencil_gt_defs(
        in_phi_now: gtscript.Field["dtype"],
        in_phi_new: gtscript.Field["dtype"],
        in_phi_ref: gtscript.Field["dtype"],
        in_rmat: gtscript.Field["dtype"],
        out_phi: gtscript.Field["dtype"],
        *,
        dt: float
    ) -> None:
        with computation(PARALLEL), interval(...):
            out_phi = in_phi_new - dt * in_rmat * (in_phi_now - in_phi_ref)
