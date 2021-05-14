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
from typing import Optional, Sequence, TYPE_CHECKING

from sympl._core.dynamic_checkers import (
    InflowComponentChecker,
    OutflowComponentChecker,
)
from sympl._core.dynamic_operators import (
    InflowComponentOperator,
    OutflowComponentOperator,
)
from sympl._core.static_checkers import StaticComponentChecker

from tasmania.python.framework._base import (
    BaseFromDiagnosticToTendency,
    BaseFromTendencyToDiagnostic,
)
from tasmania.python.framework.base_components import (
    DomainComponent,
    GridComponent,
)
from tasmania.python.framework.promoter_utils import StaticOperator
from tasmania.python.framework.stencil import StencilFactory

if TYPE_CHECKING:
    from sympl._core.typingx import (
        DataArrayDict,
        NDArrayLike,
        NDArrayLikeDict,
        PropertyDict,
    )

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )
    from tasmania.python.utils.typingx import TripletInt


allowed_grid_types = ("physical", "numerical")


class FromDiagnosticToTendency(
    BaseFromDiagnosticToTendency, DomainComponent, StencilFactory
):
    """
    Promote a variable from dictionary of diagnostics to a dictionary
    of tendencies.
    """

    def __init__(
        self,
        domain: "Domain",
        grid_type: str,
        *,
        enable_checks: bool = True,
        backend: Optional[str] = None,
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        # initialize parent classes
        super().__init__()
        super(BaseFromDiagnosticToTendency, self).__init__(domain, grid_type)
        super(GridComponent, self).__init__(
            backend, backend_options, storage_options
        )
        self._enable_checks = enable_checks
        self.storage_shape = self.get_storage_shape(storage_shape)

        if enable_checks:
            # check input_properties
            StaticComponentChecker.factory("input_properties").check(self)

        # infer tendency_properties
        self.tendency_properties = StaticOperator.get_tendency_properties(self)

        if enable_checks:
            # instantiate dynamic checker
            self._input_checker = InflowComponentChecker.factory(
                "input_properties", self
            )
            self._tendency_inflow_checker = InflowComponentChecker.factory(
                "tendency_properties", self
            )
            self._tendency_outflow_checker = OutflowComponentChecker.factory(
                "tendency_properties", self
            )

        # instantiate dynamic operators
        self._input_operator = InflowComponentOperator.factory(
            "input_properties", self
        )
        self._tendency_inflow_operator = InflowComponentOperator.factory(
            "tendency_properties", self
        )
        self._tendency_outflow_operator = OutflowComponentOperator.factory(
            "tendency_properties", self
        )

        # instantiate stencil
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil_copy = self.compile("copy")

    @property
    @abc.abstractmethod
    def input_properties(self) -> "PropertyDict":
        pass

    def __call__(
        self,
        diagnostics: "DataArrayDict",
        *,
        out: Optional["DataArrayDict"] = None
    ) -> "DataArrayDict":
        # inflow checks
        if self._enable_checks:
            self._input_checker.check(diagnostics)

        # extract raw diagnostics
        raw_diagnostics = self._input_operator.get_ndarray_dict(diagnostics)

        # run checks on out
        out = out if out is not None else {}
        if self._enable_checks:
            self._tendency_inflow_checker.check(out, diagnostics)

        # extract or allocate output buffers
        raw_tendencies = self._tendency_inflow_operator.get_ndarray_dict(out)
        raw_tendencies.update(
            {
                name: self.allocate_tendency(name)
                for name in self.tendency_properties
                if name not in out
            }
        )

        # run checks on raw_tendencies
        if self._enable_checks:
            self._tendency_outflow_checker.check(raw_tendencies, diagnostics)

        # compute
        self.array_call(raw_diagnostics, raw_tendencies)

        # outflow checks
        if self._enable_checks:
            self._tendency_outflow_checker.check(raw_tendencies, diagnostics)

        # wrap arrays in dataarrays
        tendencies = self._tendency_outflow_operator.get_dataarray_dict(
            raw_tendencies, diagnostics, out=out
        )

        return tendencies

    def allocate_tendency(self, name: str) -> "NDArrayLike":
        return self.zeros(shape=self.get_field_storage_shape(name))

    def get_field_storage_shape(
        self, name: str, default_storage_shape: Optional["TripletInt"] = None
    ) -> "TripletInt":
        return super().get_field_storage_shape(
            name, default_storage_shape or self.storage_shape
        )

    def array_call(
        self, diagnostics: "NDArrayLikeDict", out: "NDArrayLikeDict"
    ) -> None:
        for name in self.input_properties:
            tendency_name = self.input_properties[name].get(
                "tendency_name", name.replace("tendency_of_", "")
            )
            grid_shape = self.get_field_grid_shape(tendency_name)
            self._stencil_copy(
                src=diagnostics[name],
                dst=out[tendency_name],
                origin=(0, 0, 0),
                domain=grid_shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )


class FromTendencyToDiagnostic(
    BaseFromTendencyToDiagnostic, DomainComponent, StencilFactory
):
    """
    Promote a variable from dictionary of tendencies to a dictionary
    of diagnostics.
    """

    def __init__(
        self,
        domain: "Domain",
        grid_type: str,
        *,
        enable_checks: bool = True,
        backend: Optional[str] = None,
        backend_options: Optional["BackendOptions"] = None,
        storage_shape: Optional[Sequence[int]] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        # initialize parent classes
        super().__init__()
        super(BaseFromTendencyToDiagnostic, self).__init__(domain, grid_type)
        super(GridComponent, self).__init__(
            backend, backend_options, storage_options
        )
        self._enable_checks = enable_checks
        self.storage_shape = self.get_storage_shape(storage_shape)

        if enable_checks:
            # check input_tendency_properties
            StaticComponentChecker.factory("input_tendency_properties").check(
                self
            )

        # infer diagnostic_properties
        self.diagnostic_properties = StaticOperator.get_diagnostic_properties(
            self
        )

        if enable_checks:
            # instantiate dynamic checker
            self._input_checker = InflowComponentChecker.factory(
                "input_tendency_properties", self
            )
            self._diagnostic_inflow_checker = InflowComponentChecker.factory(
                "diagnostic_properties", self
            )
            self._diagnostic_outflow_checker = OutflowComponentChecker.factory(
                "diagnostic_properties", self
            )

        # instantiate dynamic operators
        self._input_operator = InflowComponentOperator.factory(
            "input_tendency_properties", self
        )
        self._diagnostic_inflow_operator = InflowComponentOperator.factory(
            "diagnostic_properties", self
        )
        self._diagnostic_outflow_operator = OutflowComponentOperator.factory(
            "diagnostic_properties", self
        )

        # instantiate stencil
        dtype = self.storage_options.dtype
        self.backend_options.dtypes = {"dtype": dtype}
        self._stencil_copy = self.compile("copy")

    @property
    @abc.abstractmethod
    def input_tendency_properties(self) -> "PropertyDict":
        pass

    def __call__(
        self,
        tendencies: "DataArrayDict",
        *,
        out: Optional["DataArrayDict"] = None
    ) -> "DataArrayDict":
        # inflow checks
        if self._enable_checks:
            self._input_checker.check(tendencies)

        # extract raw diagnostics
        raw_tendencies = self._input_operator.get_ndarray_dict(tendencies)

        # run checks on out
        out = out if out is not None else {}
        if self._enable_checks:
            self._diagnostic_inflow_checker.check(out, tendencies)

        # extract or allocate output buffers
        raw_diagnostics = self._diagnostic_inflow_operator.get_ndarray_dict(
            out
        )
        raw_diagnostics.update(
            {
                name: self.allocate_diagnostic(name)
                for name in self.diagnostic_properties
                if name not in out
            }
        )

        # run checks on raw_tendencies
        if self._enable_checks:
            self._diagnostic_outflow_checker.check(raw_diagnostics, tendencies)

        # compute
        self.array_call(raw_tendencies, raw_diagnostics)

        # outflow checks
        if self._enable_checks:
            self._diagnostic_outflow_checker.check(raw_diagnostics, tendencies)

        # wrap arrays in dataarrays
        diagnostics = self._diagnostic_outflow_operator.get_dataarray_dict(
            raw_diagnostics, tendencies, out=out
        )

        return diagnostics

    def allocate_diagnostic(self, name: str) -> "NDArrayLike":
        return self.zeros(shape=self.get_field_storage_shape(name))

    def get_field_storage_shape(
        self, name: str, default_storage_shape: Optional["TripletInt"] = None
    ) -> "TripletInt":
        return super().get_field_storage_shape(
            name, default_storage_shape or self.storage_shape
        )

    def array_call(
        self, tendencies: "NDArrayLikeDict", out: "NDArrayLikeDict"
    ) -> None:
        for name in self.input_tendency_properties:
            diagnostic_name = self.input_tendency_properties[name].get(
                "diagnostic_name", "tendency_of_" + name
            )
            grid_shape = self.get_field_grid_shape(diagnostic_name)
            self._stencil_copy(
                src=tendencies[name],
                dst=out[diagnostic_name],
                origin=(0, 0, 0),
                domain=grid_shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
