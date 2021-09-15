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
from typing import Optional, TYPE_CHECKING

from sympl._core.composite import (
    ImplicitTendencyComponentComposite,
    TendencyComponentComposite,
)
from sympl._core.core_components import (
    ImplicitTendencyComponent,
    TendencyComponent,
)
from sympl._core.factory import AbstractFactory
from sympl._core.steppers import (
    SequentialTendencyStepper as SymplSequentialTendencyStepper,
    TendencyStepper as SymplTendencyStepper,
)

from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.utils import typingx as ty
from tasmania.python.utils.dict import DataArrayDictOperator

if TYPE_CHECKING:
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class TendencyStepper(AbstractFactory, SymplTendencyStepper):
    """
    Callable abstract base class which steps a model state based on the
    tendencies calculated by a set of wrapped prognostic components.
    """

    allowed_component_type = (
        TendencyComponent,
        TendencyComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
        ConcurrentCoupling,
    )

    def __init__(
        self,
        *args: ty.TendencyComponent,
        execution_policy: str = "serial",
        enforce_horizontal_boundary: bool = False,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        args :
            Instances of

                * :class:`sympl.TendencyComponent`,
                * :class:`sympl.TendencyComponentComposite`,
                * :class:`sympl.ImplicitTendencyComponent`,
                * :class:`sympl.ImplicitTendencyComponentComposite`, or
                * :class:`tasmania.ConcurrentCoupling`

            providing tendencies for the prognostic variables.
        execution_policy : `str`, optional
            String specifying the runtime mode in which parameterizations
            should be invoked. See :class:`tasmania.ConcurrentCoupling`.
        enforce_horizontal_boundary : `bool`, optional
            ``True`` if the class should enforce the lateral boundary
            conditions after each stage of the time integrator,
            ``False`` otherwise. Defaults to ``False``.
            This argument is considered only if at least one of the wrapped
            objects is an instance of

                * :class:`tasmania.TendencyComponent`, or
                * :class:`tasmania.ImplicitTendencyComponent`.

        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        tendency_component = (
            ConcurrentCoupling(
                *args,
                execution_policy=execution_policy,
                enable_checks=enable_checks,
                backend=backend,
                backend_options=backend_options,
                storage_options=storage_options
            )
            if len(args) > 1
            else args[0]
        )

        super(AbstractFactory, self).__init__(
            tendency_component, enable_checks=enable_checks
        )

        enforce_hb = enforce_horizontal_boundary
        if enforce_hb:
            found = False
            for component in args:
                if not found:

                    try:  # tasmania's component
                        self._hb = component.horizontal_boundary
                        self._enforce_hb = True
                        found = True

                        break
                    except AttributeError:  # sympl's component
                        pass

            if not found:
                self._enforce_hb = False
        else:
            self._enforce_hb = False

        self._dict_op = DataArrayDictOperator(
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )


class SequentialTendencyStepper(
    AbstractFactory, SymplSequentialTendencyStepper
):
    """
    Callable abstract base class which steps a model state based on the
    tendencies calculated by a set of wrapped prognostic components.
    """

    allowed_component_type = (
        TendencyComponent,
        TendencyComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
        ConcurrentCoupling,
    )

    def __init__(
        self,
        *args: ty.TendencyComponent,
        execution_policy: str = "serial",
        enforce_horizontal_boundary: bool = False,
        enable_checks: bool = True,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        args :
            Instances of

                * :class:`sympl.TendencyComponent`,
                * :class:`sympl.TendencyComponentComposite`,
                * :class:`sympl.ImplicitTendencyComponent`,
                * :class:`sympl.ImplicitTendencyComponentComposite`, or
                * :class:`tasmania.ConcurrentCoupling`

            providing tendencies for the prognostic variables.
        execution_policy : `str`, optional
            String specifying the runtime mode in which parameterizations
            should be invoked. See :class:`tasmania.ConcurrentCoupling`.
        enforce_horizontal_boundary : `bool`, optional
            ``True`` if the class should enforce the lateral boundary
            conditions after each stage of the time integrator,
            ``False`` otherwise. Defaults to ``False``.
            This argument is considered only if at least one of the wrapped
            objects is an instance of

                * :class:`tasmania.TendencyComponent`, or
                * :class:`tasmania.ImplicitTendencyComponent`.

        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        tendency_component = (
            ConcurrentCoupling(
                *args,
                execution_policy=execution_policy,
                enable_checks=enable_checks,
                backend=backend,
                backend_options=backend_options,
                storage_options=storage_options
            )
            if len(args) > 1
            else args[0]
        )

        super(AbstractFactory, self).__init__(
            tendency_component, enable_checks=enable_checks
        )

        enforce_hb = enforce_horizontal_boundary
        if enforce_hb:
            found = False
            for component in args:
                if not found:

                    try:  # tasmania's component
                        self._hb = component.horizontal_boundary
                        self._enforce_hb = True
                        found = True

                        break
                    except AttributeError:  # sympl's component
                        pass

            if not found:
                self._enforce_hb = False
        else:
            self._enforce_hb = False

        self._dict_op = DataArrayDictOperator(
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )
