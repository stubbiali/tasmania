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
from typing import Dict, Mapping, Optional, Sequence, TYPE_CHECKING, Tuple

import sympl
from sympl._core.time import FakeTimer as Timer

# from sympl._core.time import Timer

from tasmania.python.framework.base_components import (
    DomainComponent,
    PhysicalConstantsComponent,
    GridComponent,
)

from tasmania.python.framework.stencil import StencilFactory

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict, NDArrayLike
    from xarray import DataArray

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )
    from tasmania.python.utils.typingx import TimeDelta, TripletInt


class DiagnosticComponent(
    DomainComponent,
    PhysicalConstantsComponent,
    StencilFactory,
    sympl.DiagnosticComponent,
):
    """
    Custom version of :class:`sympl.DiagnosticComponent` which is aware
    of the spatial domain over which the component is instantiated.
    """

    def __init__(
        self: "DiagnosticComponent",
        domain: "Domain",
        grid_type: str = "numerical",
        physical_constants: Optional[Mapping[str, "DataArray"]] = None,
        *,
        enable_checks: bool = True,
        backend: Optional[str] = None,
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : `str`, optional
            The type of grid over which instantiating the class.
            Either "physical" or "numerical" (default).
        physical_constants : `dict`, optional
            Dictionary of physical constants.
        enable_checks : `bool`, optional
            ``True`` to make Sympl run all sanity checks, ``False`` otherwise.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        super().__init__(domain, grid_type)
        super(GridComponent, self).__init__(physical_constants)
        super(PhysicalConstantsComponent, self).__init__(
            backend, backend_options, storage_options
        )
        super(StencilFactory, self).__init__(enable_checks=enable_checks)
        self.storage_shape = self.get_storage_shape(storage_shape)

    def __call__(
        self: "DiagnosticComponent",
        state: "DataArrayDict",
        *,
        out: Optional["DataArrayDict"] = None
    ) -> "DataArrayDict":
        Timer.start(label=self.__class__.__name__)
        out = super().__call__(state, out=out)
        Timer.stop()
        return out

    def allocate_diagnostic(self, name: str) -> "NDArrayLike":
        return self.zeros(shape=self.get_field_storage_shape(name))

    def get_field_storage_shape(
        self, name: str, default_storage_shape: Optional["TripletInt"] = None
    ) -> "TripletInt":
        return super().get_field_storage_shape(
            name, default_storage_shape or self.storage_shape
        )


class ImplicitTendencyComponent(
    DomainComponent,
    PhysicalConstantsComponent,
    StencilFactory,
    sympl.ImplicitTendencyComponent,
):
    """
    Customized version of :class:`sympl.ImplicitTendencyComponent` which is
    aware of the grid over which the component is instantiated.
    """

    def __init__(
        self: "ImplicitTendencyComponent",
        domain: "Domain",
        grid_type: str = "numerical",
        physical_constants: Optional[Mapping[str, "DataArray"]] = None,
        tendencies_in_diagnostics: bool = False,
        name: Optional[str] = None,
        *,
        enable_checks: bool = True,
        backend: Optional[str] = None,
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : `str`, optional
            The type of grid over which instantiating the class.
            Either "physical" or "numerical" (default).
        physical_constants : `dict`, optional
            Dictionary of physical constants.
        tendencies_in_diagnostics : `bool`, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : `str`, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        enable_checks : `bool`, optional
            ``True`` to make Sympl run all sanity checks, ``False`` otherwise.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        super().__init__(domain, grid_type)
        super(GridComponent, self).__init__(physical_constants)
        super(PhysicalConstantsComponent, self).__init__(
            backend, backend_options, storage_options
        )
        super(StencilFactory, self).__init__(
            tendencies_in_diagnostics, name, enable_checks=enable_checks
        )
        self.storage_shape = self.get_storage_shape(storage_shape)

    def __call__(
        self: "ImplicitTendencyComponent",
        state: "DataArrayDict",
        timestep: "TimeDelta",
        *,
        out_tendencies: Optional["DataArrayDict"] = None,
        out_diagnostics: Optional["DataArrayDict"] = None,
        overwrite_tendencies: Optional[Dict[str, bool]] = None
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        Timer.start(label=self.__class__.__name__)
        tendencies, diagnostics = super().__call__(
            state,
            timestep,
            out_tendencies=out_tendencies,
            out_diagnostics=out_diagnostics,
            overwrite_tendencies=overwrite_tendencies,
        )
        Timer.stop()
        return tendencies, diagnostics

    def allocate_tendency(self, name: str) -> "NDArrayLike":
        return self.zeros(shape=self.get_field_storage_shape(name))

    def allocate_diagnostic(self, name: str) -> "NDArrayLike":
        return self.zeros(shape=self.get_field_storage_shape(name))

    def get_field_storage_shape(
        self, name: str, default_storage_shape: Optional["TripletInt"] = None
    ) -> "TripletInt":
        return super().get_field_storage_shape(
            name, default_storage_shape or self.storage_shape
        )


class Stepper(
    DomainComponent, PhysicalConstantsComponent, StencilFactory, sympl.Stepper
):
    """
    Customized version of :class:`sympl.Stepper` which is aware
    of the grid over which the component is instantiated.
    """

    def __init__(
        self: "Stepper",
        domain: "Domain",
        grid_type: str = "numerical",
        physical_constants: Optional[Mapping[str, "DataArray"]] = None,
        tendencies_in_diagnostics: bool = False,
        name: Optional[str] = None,
        *,
        enable_checks: bool = True,
        backend: Optional[str] = None,
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : `str`, optional
            The type of grid over which instantiating the class.
            Either "physical" or "numerical" (default).
        physical_constants : `dict`, optional
            Dictionary of physical constants.
        tendencies_in_diagnostics : `bool`, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : `str`, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        enable_checks : `bool`, optional
            ``True`` to make Sympl run all sanity checks, ``False`` otherwise.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        super().__init__(domain, grid_type)
        super(GridComponent, self).__init__(physical_constants)
        super(PhysicalConstantsComponent, self).__init__(
            backend, backend_options, storage_options
        )
        super(StencilFactory, self).__init__(
            tendencies_in_diagnostics, name, enable_checks=enable_checks
        )
        self.storage_shape = self.get_storage_shape(storage_shape)


class TendencyComponent(
    DomainComponent,
    PhysicalConstantsComponent,
    StencilFactory,
    sympl.TendencyComponent,
):
    """
    Customized version of :class:`sympl.TendencyComponent` which is aware
    of the grid over which the component is instantiated.
    """

    def __init__(
        self: "TendencyComponent",
        domain: "Domain",
        grid_type: str = "numerical",
        physical_constants: Optional[Mapping[str, "DataArray"]] = None,
        tendencies_in_diagnostics: bool = False,
        name: Optional[str] = None,
        *,
        enable_checks: bool = True,
        backend: Optional[str] = None,
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : `str`, optional
            The type of grid over which instantiating the class.
            Either "physical" or "numerical" (default).
        physical_constants : `dict`, optional
            Dictionary of physical constants.
        tendencies_in_diagnostics : `bool`, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : `str`, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
        enable_checks : `bool`, optional
            ``True`` to make Sympl run all sanity checks, ``False`` otherwise.
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_shape : `Sequence[int]`, optional
            The shape of the storages allocated within the class.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        super().__init__(domain, grid_type)
        super(GridComponent, self).__init__(physical_constants)
        super(PhysicalConstantsComponent, self).__init__(
            backend, backend_options, storage_options
        )
        super(StencilFactory, self).__init__(
            tendencies_in_diagnostics, name, enable_checks=enable_checks
        )
        self.storage_shape = self.get_storage_shape(storage_shape)

    def __call__(
        self: "TendencyComponent",
        state: "DataArrayDict",
        *,
        out_tendencies: Optional["DataArrayDict"] = None,
        out_diagnostics: Optional["DataArrayDict"] = None,
        overwrite_tendencies: Optional[Dict[str, bool]] = None
    ) -> Tuple["DataArrayDict", "DataArrayDict"]:
        Timer.start(label=self.__class__.__name__)
        tendencies, diagnostics = super().__call__(
            state,
            out_tendencies=out_tendencies,
            out_diagnostics=out_diagnostics,
            overwrite_tendencies=overwrite_tendencies,
        )
        Timer.stop()
        return tendencies, diagnostics

    def allocate_tendency(self, name: str) -> "NDArrayLike":
        return self.zeros(shape=self.get_field_storage_shape(name))

    def allocate_diagnostic(self, name: str) -> "NDArrayLike":
        return self.zeros(shape=self.get_field_storage_shape(name))

    def get_field_storage_shape(
        self, name: str, default_storage_shape: Optional["TripletInt"] = None
    ) -> "TripletInt":
        return super().get_field_storage_shape(
            name, default_storage_shape or self.storage_shape
        )
