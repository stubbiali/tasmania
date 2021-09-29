# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from typing import Any, Dict, Optional, Sequence, TYPE_CHECKING, Type, Union

from sympl._core.factory import AbstractFactory

from tasmania.python.framework.base_components import (
    DomainComponent,
    GridComponent,
)
from tasmania.python.framework.register import factorize
from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.utils import typingx as ty

if TYPE_CHECKING:
    from sympl._core.typingx import NDArrayLikeDict

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )
    from tasmania.python.isentropic.dynamics.horizontal_fluxes import (
        IsentropicHorizontalFlux,
        IsentropicMinimalHorizontalFlux,
    )
    from tasmania.python.utils.typingx import TimeDelta

# convenient aliases
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class IsentropicPrognostic(AbstractFactory, DomainComponent, StencilFactory):
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
        horizontal_flux_class: Union[
            Type["IsentropicHorizontalFlux"],
            Type["IsentropicMinimalHorizontalFlux"],
        ],
        horizontal_flux_scheme: str,
        domain: "Domain",
        moist: bool,
        backend: str,
        backend_options: "BackendOptions",
        storage_shape: Sequence[int],
        storage_options: "StorageOptions",
    ) -> None:
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
        domain : tasmania.Domain
            The underlying domain.
        moist : bool
            ``True`` for a moist dynamical core, ``False`` otherwise.
        backend : str
            The backend.
        backend_options : BackendOptions
            Backend-specific options.
        storage_shape : Sequence[int]
            The shape of the storages allocated within the class.
        storage_options : StorageOptions
            Storage-related options.
        """
        # initialize parent classes
        super().__init__(domain, "numerical")
        super(GridComponent, self).__init__(
            backend, backend_options, storage_options
        )

        # save arguments needed at compile and run time
        self._moist = moist

        # set proper storage shape
        g = self.grid
        self._storage_shape = self.get_storage_shape(
            storage_shape, (g.nx, g.ny, g.nz + 1)
        )

        # instantiate the class computing the numerical horizontal fluxes
        self._hflux = horizontal_flux_class.factory(
            horizontal_flux_scheme, backend=backend
        )
        hb = self.horizontal_boundary
        assert hb.nb >= self._hflux.extent, (
            f"The number of lateral boundary layers is {hb.nb}, but should be "
            f"greater or equal than {self._hflux.extent}."
        )
        assert g.nx >= 2 * hb.nb + 1, (
            f"The number of grid points along the first horizontal "
            f"dimension is {g.nx}, but should be greater or equal than "
            f"{2 * hb.nb + 1}."
        )
        assert g.ny >= 2 * hb.nb + 1, (
            f"The number of grid points along the second horizontal "
            f"dimension is {g.ny}, but should be greater or equal than "
            f"{2 * hb.nb + 1}."
        )

    @property
    @abc.abstractmethod
    def stages(self) -> int:
        """
        Return
        ------
        int :
            The number of stages performed by the time-integration scheme.
        """
        pass

    @property
    @abc.abstractmethod
    def substep_fractions(self) -> Union[float, Sequence[float]]:
        pass

    @abc.abstractmethod
    def stage_call(
        self,
        stage: int,
        timestep: "TimeDelta",
        state: "NDArrayLikeDict",
        tendencies: "NDArrayLikeDict",
        out_state: "NDArrayLikeDict",
    ) -> None:
        """
        Perform a stage.

        Parameters
        ----------
        stage : int
            The stage to perform.
        timestep : datetime.timedelta
            The time step.
        state : dict[str, array-like]
            The (raw) state at the current stage.
        tendencies : dict[str, array-like]
            The (raw) tendencies for the prognostic model variables.
        out_state : dict[str, array-like]
            The (raw) state at the next stage.
        """
        pass

    def _stencils_allocate_temporaries(self) -> None:
        pass
