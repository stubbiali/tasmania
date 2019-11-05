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
import numpy as np

from tasmania.python.utils.storage_utils import zeros

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float32


# convenient aliases
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicPrognostic(abc.ABC):
    """
    Abstract base class whose derived classes implement different
    schemes to carry out the prognostic steps of the three-dimensional
    moist, isentropic dynamical core. The schemes might be *semi-implicit* -
    they treat horizontal advection explicitly and the pressure gradient
    implicitly. The vertical advection, the Coriolis acceleration and
    the sedimentation motion are not included in the dynamics, but rather
    parameterized. The conservative form of the governing equations is used.
    """

    def __init__(
        self,
        horizontal_flux_class,
        horizontal_flux_scheme,
        grid,
        hb,
        moist,
        backend,
        backend_opts,
        build_info,
        dtype,
        exec_info,
        default_origin,
        rebuild,
        storage_shape,
        managed_memory,
    ):
        """
        Parameters
        ----------
        horizontal_flux_class : IsentropicHorizontalFlux, IsentropicMinimal
            Either :class:`~tasmania.IsentropicHorizontalFlux`
            or :class:`~tasmania.IsentropicMinimalHorizontalFlux`.
        horizontal_flux_scheme : str
            The numerical horizontal flux scheme to implement.
            See :class:`~tasmania.IsentropicHorizontalFlux` and
            :class:`~tasmania.IsentropicMinimalHorizontalFlux`
            for the complete list of the available options.
        grid : tasmania.Grid
            The underlying grid.
        hb : tasmania.HorizontalBoundary
            The object handling the lateral boundary conditions.
        moist : bool
            :obj:`True` for a moist dynamical core, :obj:`False` otherwise.
        backend : str
            TODO
        backend_opts : dict
            TODO
        build_info : dict
            TODO
        dtype : numpy.dtype
            TODO
        exec_info : dict
            TODO
        default_origin : tuple
            TODO
        rebuild : bool
            TODO
        storage_shape : tuple
            TODO
        managed_memory : bool
            `True` to allocate the storages as managed memory, `False` otherwise.
        """
        # store input arguments needed at compile- and run-time
        self._grid = grid
        self._hb = hb
        self._moist = moist
        self._backend = backend
        self._backend_opts = backend_opts or {}
        self._build_info = build_info
        self._dtype = dtype
        self._exec_info = exec_info
        self._default_origin = default_origin
        self._rebuild = rebuild
        self._managed_memory = managed_memory

        nx, ny, nz = grid.nx, grid.ny, grid.nz
        storage_shape = (nx, ny, nz + 1) if storage_shape is None else storage_shape
        error_msg = "storage_shape must be larger or equal than {}.".format(
            (nx, ny, nz + 1)
        )
        assert storage_shape[0] >= nx, error_msg
        assert storage_shape[1] >= ny, error_msg
        assert storage_shape[2] >= nz + 1, error_msg
        self._storage_shape = storage_shape

        # instantiate the class computing the numerical horizontal fluxes
        self._hflux = horizontal_flux_class.factory(horizontal_flux_scheme)
        assert hb.nb >= self._hflux.extent, (
            "The number of lateral boundary layers is {}, but should be "
            "greater or equal than {}.".format(hb.nb, self._hflux.extent)
        )
        assert grid.nx >= 2 * hb.nb + 1, (
            "The number of grid points along the first horizontal "
            "dimension is {}, but should be greater or equal than {}.".format(
                grid.nx, 2 * hb.nb + 1
            )
        )
        assert grid.ny >= 2 * hb.nb + 1, (
            "The number of grid points along the second horizontal "
            "dimension is {}, but should be greater or equal than {}.".format(
                grid.ny, 2 * hb.nb + 1
            )
        )

        # allocate the gt storages collecting the output fields computed by
        # the underlying stencils
        self._stencils_allocate_outputs()

        # initialize the pointers to the storages collecting the physics tendencies
        self._s_tnd = None
        self._su_tnd = None
        self._sv_tnd = None
        self._qv_tnd = None
        self._qc_tnd = None
        self._qr_tnd = None

    @property
    @abc.abstractmethod
    def stages(self):
        """
        Return
        ------
        int :
            The number of stages performed by the time-integration scheme.
        """

    @abc.abstractmethod
    def stage_call(self, stage, timestep, state, tendencies=None):
        """
        Perform a stage.

        Parameters
        ----------
        stage : int
            The stage to perform.
        timestep : timedelta
            :class:`datetime.timedelta` representing the time step.
        state : dict
            Dictionary whose keys are strings indicating model variables,
            and values are :class:`numpy.ndarray`\s representing the values
            for those variables.
        tendencies : dict
            Dictionary whose keys are strings indicating model variables,
            and values are :class:`numpy.ndarray`\s representing (slow and
            intermediate) physical tendencies for those variables.

        Return
        ------
        dict :
            Dictionary whose keys are strings indicating the conservative
            prognostic model variables, and values are :class:`numpy.ndarray`\s
            containing new values for those variables.
        """
        pass

    @staticmethod
    def factory(
        time_integration_scheme,
        horizontal_flux_scheme,
        grid,
        hb,
        moist=False,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        storage_shape=None,
        managed_memory=False,
        **kwargs
    ):
        """
        Static method returning an instance of the derived class implementing
        the time stepping scheme specified by ``time_scheme``.

        Parameters
        ----------
        time_integration_scheme : str
            The time stepping method to implement. Available options are:

                * 'forward_euler_si', for the semi-implicit forward Euler scheme;
                * 'centered_si', for the semi-implicit centered scheme;
                * 'rk3ws_si', for the semi-implicit three-stages RK scheme;
                * 'sil3', for the semi-implicit Lorenz three cycle scheme.

        horizontal_flux_scheme : str
            The numerical horizontal flux scheme to implement.
            See :class:`~tasmania.IsentropicHorizontalFlux` and
            :class:`~tasmania.IsentropicMinimalHorizontalFlux`
            for the complete list of the available options.
        grid : tasmania.Grid
            The underlying grid.
        hb : tasmania.HorizontalBoundary
            The object handling the lateral boundary conditions.
        moist : `bool`, optional
            :obj:`True` for a moist dynamical core, :obj:`False` otherwise.
            Defaults to :obj:`False`.
        backend : `str`, optional
            TODO
        backend_opts : `dict`, optional
            TODO
        build_info : `dict`, optional
            TODO
        dtype : `numpy.dtype`, optional
            TODO
        exec_info : `dict`, optional
            TODO
        default_origin : `tuple`, optional
            TODO
        rebuild : `bool`, optional
            TODO
        storage_shape : `tuple`, optional
            TODO
        managed_memory : `bool`, optional
            `True` to allocate the storages as managed memory, `False` otherwise.

        Return
        ------
        obj :
            An instance of the derived class implementing ``time_integration_scheme``.
        """
        from .implementations.prognostic import ForwardEulerSI, CenteredSI, RK3WSSI, SIL3

        args = (
            horizontal_flux_scheme,
            grid,
            hb,
            moist,
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

        available = ("forward_euler_si", "centered_si", "rk3ws_si", "sil3")

        if time_integration_scheme == "forward_euler_si":
            return ForwardEulerSI(*args, **kwargs)
        elif time_integration_scheme == "centered_si":
            return CenteredSI(*args, **kwargs)
        elif time_integration_scheme == "rk3ws_si":
            return RK3WSSI(*args, **kwargs)
        elif time_integration_scheme == "sil3":
            return SIL3(*args, **kwargs)
        else:
            raise ValueError(
                "Unknown time integration scheme {}. Available options are "
                "{}.".format(time_integration_scheme, ",".join(available))
            )

    def _stencils_allocate_outputs(self):
        """
        Allocate the storages which collect the output fields calculated
        by the underlying gt4py stencils.
        """
        storage_shape = self._storage_shape
        backend = self._backend
        dtype = self._dtype
        default_origin = self._default_origin
        managed_memory = self._managed_memory

        self._s_new = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._su_new = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        self._sv_new = zeros(
            storage_shape, backend, dtype, default_origin, managed_memory=managed_memory
        )
        if self._moist:
            self._sqv_new = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )
            self._sqc_new = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )
            self._sqr_new = zeros(
                storage_shape,
                backend,
                dtype,
                default_origin,
                managed_memory=managed_memory,
            )
