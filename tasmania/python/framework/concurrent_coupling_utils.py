# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
from typing import Dict, List, Optional, Sequence, TYPE_CHECKING

from sympl._core.dynamic_operators import OutflowComponentOperator
from sympl._core.static_operators import StaticComponentOperator

from tasmania.python.framework._base import (
    BaseFromDiagnosticToTendency,
    BaseFromTendencyToDiagnostic,
)
from tasmania.python.framework.exceptions import (
    CouplingError,
    IncompatibleDimensionsError,
    IncompatibleUnitsError,
)
from tasmania.python.framework.static_checkers import (
    check_dims_are_compatible,
    check_units_are_compatible,
)
from tasmania.python.framework.static_operators import merge_dims

if TYPE_CHECKING:
    from sympl._core.typingx import (
        DataArrayDict,
        NDArrayLike,
        NDArrayLikeDict,
        PropertyDict,
    )

    from tasmania.python.domain.domain import Domain
    from tasmania.python.framework.concurrent_coupling import (
        ConcurrentCoupling,
    )
    from tasmania.python.utils.typingx import Component


class StaticOperator:
    input_operator = StaticComponentOperator.factory("input_properties")
    diagnostic_operator = StaticComponentOperator.factory(
        "diagnostic_properties"
    )
    tendency_operator = StaticComponentOperator.factory("tendency_properties")
    sco = {
        "input_properties": input_operator,
        "diagnostic_properties": diagnostic_operator,
        "tendency_properties": tendency_operator,
    }

    @classmethod
    def get_input_properties(
        cls, coupler: "ConcurrentCoupling"
    ) -> "PropertyDict":
        components = coupler.components
        ignore_diagnostics = coupler.execution_policy == "as_parallel"
        return cls._get_input_properties(components, ignore_diagnostics)

    @classmethod
    def get_diagnostic_properties(
        cls, coupler: "ConcurrentCoupling"
    ) -> "PropertyDict":
        return cls._get_diagnostic_properties(coupler.components)

    @classmethod
    def get_tendency_properties(
        cls, coupler: "ConcurrentCoupling"
    ) -> "PropertyDict":
        return cls._get_tendency_properties(coupler.components)

    @classmethod
    def get_overwrite_tendencies(
        cls, coupler: "ConcurrentCoupling"
    ) -> List[Dict[str, bool]]:
        tendencies = set()
        out = []
        for component in coupler.components:
            tendency_properties = cls.tendency_operator.get_properties(
                component
            )
            overwrite_tendencies = {}
            for name in tendency_properties:
                overwrite_tendencies[name] = name not in tendencies
                tendencies.add(name)
            out.append(overwrite_tendencies)
        return out

    @classmethod
    def get_horizontal_boundary(
        cls, coupler: "ConcurrentCoupling"
    ) -> Optional["Domain"]:
        for component in coupler.components:
            if hasattr(component, "horizontal_boundary"):
                return component.horizontal_boundary
        return None

    @classmethod
    def _get_input_properties(
        cls, components: Sequence["Component"], ignore_diagnostics: bool
    ) -> "PropertyDict":
        available_inputs = set()
        out = {}

        for component in components:
            input_properties = cls.input_operator.get_properties(component)
            input_dims = cls.input_operator.get_dims(component)
            input_units = {
                name: input_properties[name]["units"]
                for name in input_properties
            }
            input_aliases = cls.input_operator.get_aliases(component)
            for name in input_properties:
                if name in out:
                    out[name]["dims"] = merge_dims(
                        out[name]["dims"], input_dims[name]
                    )
                elif name not in available_inputs:
                    out[name] = {
                        "dims": input_dims[name],
                        "units": input_units[name],
                    }
                    if name in input_aliases:
                        out[name]["alias"] = input_aliases[name]
                    available_inputs.add(name)

            if not ignore_diagnostics:
                diagnostic_properties = cls.diagnostic_operator.get_properties(
                    component
                )
                available_inputs.update(diagnostic_properties.keys())

        return out

    @classmethod
    def _get_diagnostic_properties(
        cls, components: Sequence["Component"]
    ) -> "PropertyDict":
        out = {}

        for component in components:
            diagnostic_properties = cls.diagnostic_operator.get_properties(
                component
            )
            diagnostic_dims = cls.diagnostic_operator.get_dims(component)
            diagnostic_units = {
                name: diagnostic_properties[name]["units"]
                for name in diagnostic_properties
            }
            diagnostic_aliases = cls.diagnostic_operator.get_aliases(component)
            for name in diagnostic_properties:
                out[name] = {
                    "dims": diagnostic_dims[name],
                    "units": diagnostic_units[name],
                }
                if name in diagnostic_aliases:
                    out[name]["alias"] = diagnostic_aliases[name]

        return out

    @classmethod
    def _get_tendency_properties(
        cls, components: Sequence["Component"]
    ) -> "PropertyDict":
        out = {}

        for component in components:
            tendency_properties = cls.tendency_operator.get_properties(
                component
            )
            tendency_dims = cls.tendency_operator.get_dims(component)
            tendency_units = {
                name: tendency_properties[name]["units"]
                for name in tendency_properties
            }
            tendency_aliases = cls.tendency_operator.get_aliases(component)
            for name in tendency_properties:
                if name in out:
                    out[name]["dims"] = merge_dims(
                        out[name]["dims"], tendency_dims[name]
                    )
                    out[name]["units"] = tendency_units[name]
                else:
                    out[name] = {
                        "dims": tendency_dims[name],
                        "units": tendency_units[name],
                    }
                    if name in tendency_aliases:
                        out[name]["alias"] = tendency_aliases[name]

        return out


class StaticChecker:
    so = StaticOperator

    @classmethod
    def check_components_type(
        cls, coupler: "ConcurrentCoupling", components: Sequence["Component"]
    ) -> None:
        allowed_types = getattr(coupler, "allowed_component_type", object)
        for component in components:
            if not isinstance(component, allowed_types):
                raise CouplingError(
                    f"ConcurrentCoupling: {component.__class__.__name__} "
                    f"should be of type {', '.join(allowed_types)}."
                )

    @classmethod
    def check_input_dims(
        cls, components: Sequence["Component"], ignore_diagnostics: bool
    ) -> None:
        dims = {}
        for component in components:
            input_dims = cls.so.input_operator.get_dims(component)
            for name in input_dims:
                if name in dims:
                    try:
                        check_dims_are_compatible(dims[name], input_dims[name])
                    except IncompatibleDimensionsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )

                    dims[name] = merge_dims(dims[name], input_dims[name])
                else:
                    dims[name] = input_dims[name]

            if not ignore_diagnostics:
                diagnostic_dims = cls.so.diagnostic_operator.get_dims(
                    component
                )
                for name in diagnostic_dims:
                    if name in dims:
                        try:
                            check_dims_are_compatible(
                                dims[name], diagnostic_dims[name]
                            )
                        except IncompatibleDimensionsError as err:
                            raise CouplingError(
                                f"ConcurrentCoupling, "
                                f"{component.__class__.__name__}: {err}."
                            )

                        dims[name] = merge_dims(
                            dims[name], diagnostic_dims[name]
                        )
                    else:
                        dims[name] = diagnostic_dims[name]

    @classmethod
    def check_input_units(
        cls, components: Sequence["Component"], ignore_diagnostics: bool
    ) -> None:
        units = {}
        for component in components:
            input_properties = cls.so.input_operator.get_properties(component)
            for name in input_properties:
                if name in units:
                    try:
                        check_units_are_compatible(
                            units[name], input_properties[name]["units"]
                        )
                    except IncompatibleUnitsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )
                else:
                    units[name] = input_properties[name]["units"]

            if not ignore_diagnostics:
                diagnostic_properties = (
                    cls.so.diagnostic_operator.get_properties(component)
                )
                for name in diagnostic_properties:
                    if name in units:
                        try:
                            check_units_are_compatible(
                                units[name],
                                diagnostic_properties[name]["units"],
                            )
                        except IncompatibleDimensionsError as err:
                            raise CouplingError(
                                f"ConcurrentCoupling, "
                                f"{component.__class__.__name__}: {err}."
                            )
                    else:
                        units[name] = diagnostic_properties[name]["units"]

    @classmethod
    def check_diagnostic_dims(
        cls, components: Sequence["Component"], ignore_diagnostics: bool
    ) -> None:
        input_properties = cls.so._get_input_properties(
            components, ignore_diagnostics=ignore_diagnostics
        )
        dims = {}

        for component in components:
            diagnostic_dims = cls.so.diagnostic_operator.get_dims(component)
            for name in diagnostic_dims:
                if name in dims:
                    try:
                        check_dims_are_compatible(
                            dims[name], diagnostic_dims[name]
                        )
                    except IncompatibleDimensionsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )

                    dims[name] = merge_dims(dims[name], diagnostic_dims[name])
                else:
                    dims[name] = diagnostic_dims[name]

                if name in input_properties:
                    try:
                        check_dims_are_compatible(
                            dims[name], input_properties[name]["dims"]
                        )
                    except IncompatibleDimensionsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )

                    dims[name] = merge_dims(
                        dims[name], input_properties[name]["dims"]
                    )

    @classmethod
    def check_diagnostic_units(
        cls, components: Sequence["Component"], ignore_diagnostics: bool
    ) -> None:
        input_properties = cls.so._get_input_properties(
            components, ignore_diagnostics=ignore_diagnostics
        )
        units = {}

        for component in components:
            diagnostic_properties = cls.so.diagnostic_operator.get_properties(
                component
            )
            for name in diagnostic_properties:
                if name in units:
                    try:
                        check_units_are_compatible(
                            units[name], diagnostic_properties[name]["units"]
                        )
                    except IncompatibleUnitsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )
                else:
                    units[name] = diagnostic_properties[name]["units"]

                if name in input_properties:
                    try:
                        check_units_are_compatible(
                            units[name], input_properties[name]["units"]
                        )
                    except IncompatibleUnitsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )

    @classmethod
    def check_tendency_dims(
        cls, components: Sequence["Component"], ignore_diagnostics: bool
    ) -> None:
        input_properties = cls.so._get_input_properties(
            components, ignore_diagnostics
        )
        diagnostic_properties = cls.so._get_diagnostic_properties(components)
        dims = {}

        for component in components:
            tendency_dims = cls.so.tendency_operator.get_dims(component)
            for name in tendency_dims:
                if name in dims:
                    try:
                        check_dims_are_compatible(
                            dims[name], tendency_dims[name]
                        )
                    except IncompatibleDimensionsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )

                    dims[name] = merge_dims(dims[name], tendency_dims[name])
                else:
                    dims[name] = tendency_dims[name]

                if name in input_properties:
                    try:
                        check_dims_are_compatible(
                            dims[name], input_properties[name]["dims"]
                        )
                    except IncompatibleDimensionsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )

                    dims[name] = merge_dims(
                        dims[name], input_properties[name]["dims"]
                    )

                if name in diagnostic_properties:
                    try:
                        check_dims_are_compatible(
                            dims[name], diagnostic_properties[name]["dims"]
                        )
                    except IncompatibleDimensionsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )

                    dims[name] = merge_dims(
                        dims[name], diagnostic_properties[name]["dims"]
                    )

    @classmethod
    def check_tendency_units(
        cls, components: Sequence["Component"], ignore_diagnostics: bool
    ) -> None:
        input_properties = cls.so._get_input_properties(
            components, ignore_diagnostics=ignore_diagnostics
        )
        diagnostic_properties = cls.so._get_diagnostic_properties(components)
        units = {}

        for component in components:
            tendency_properties = cls.so.tendency_operator.get_properties(
                component
            )
            for name in tendency_properties:
                if name in units:
                    try:
                        check_units_are_compatible(
                            units[name], tendency_properties[name]["units"]
                        )
                    except IncompatibleUnitsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )
                else:
                    units[name] = tendency_properties[name]["units"]

                if name in input_properties:
                    try:
                        check_units_are_compatible(
                            units[name],
                            input_properties[name]["units"] + " s^-1",
                        )
                    except IncompatibleUnitsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )

                if name in diagnostic_properties:
                    try:
                        check_units_are_compatible(
                            units[name],
                            diagnostic_properties[name]["units"] + " s^-1",
                        )
                    except IncompatibleUnitsError as err:
                        raise CouplingError(
                            f"ConcurrentCoupling, "
                            f"{component.__class__.__name__}: {err}."
                        )

    @classmethod
    def check_t2d(cls, components: Sequence["Component"]) -> None:
        tendencies = set()
        for component in components:
            if isinstance(component, BaseFromTendencyToDiagnostic):
                requirements = component.input_properties.keys()
                missing = [
                    key for key in requirements if key not in tendencies
                ]
                if len(missing) > 0:
                    raise CouplingError()
                tendencies.difference_update(missing)
            else:
                tendencies.update(
                    cls.so.tendency_operator.get_properties(component).keys()
                )

    @classmethod
    def check(cls, coupler: "ConcurrentCoupling") -> None:
        components = coupler.components
        ignore_diagnostics = coupler.execution_policy == "as_parallel"

        cls.check_input_dims(components, ignore_diagnostics)
        cls.check_input_units(components, ignore_diagnostics)
        cls.check_diagnostic_dims(components, ignore_diagnostics)
        cls.check_diagnostic_units(components, ignore_diagnostics)
        cls.check_tendency_dims(components, ignore_diagnostics)
        cls.check_tendency_units(components, ignore_diagnostics)
        # cls.check_t2d(components)


class DynamicOperator:
    properties_name: str = None

    def __init__(self, coupler: "ConcurrentCoupling") -> None:
        self.coupler = coupler
        self.sco_tendencies = StaticComponentOperator.factory(
            "tendency_properties"
        )
        self.sco_diagnostics = StaticComponentOperator.factory(
            "diagnostic_properties"
        )
        self.dco_tendencies = OutflowComponentOperator.factory(
            "tendency_properties", coupler
        )
        self.dco_diagnostics = OutflowComponentOperator.factory(
            "diagnostic_properties", coupler
        )

    def allocate_tendency(self, name: str) -> "NDArrayLike":
        for component in self.coupler.components:
            tendency_properties = self.sco_tendencies.get_properties(component)
            if name in tendency_properties:
                allocate_tendency = self.sco_tendencies.get_allocator(
                    component
                )
                if allocate_tendency is not None:
                    return allocate_tendency(name)

    def allocate_tendencies(self, state: "DataArrayDict") -> "DataArrayDict":
        raw_tendencies = {}

        for component in self.coupler.components:
            tendency_properties = self.sco_tendencies.get_properties(component)
            allocate_tendency = self.sco_tendencies.get_allocator(component)
            if allocate_tendency is not None:
                for name in tendency_properties:
                    if (
                        name in self.coupler.tendency_properties
                        and name not in raw_tendencies
                    ):
                        raw_tendencies[name] = allocate_tendency(name)
                        raw_tendencies[name][...] = 0.0

        tendencies = self.dco_tendencies.get_dataarray_dict(
            raw_tendencies, state
        )

        return tendencies

    def allocate_diagnostic(self, name: str) -> "NDArrayLike":
        for component in self.coupler.components:
            diagnostic_properties = self.sco_diagnostics.get_properties(
                component
            )
            if name in diagnostic_properties:
                allocate_diagnostic = self.sco_diagnostics.get_allocator(
                    component
                )
                if allocate_diagnostic is not None:
                    return allocate_diagnostic(name)

    def allocate_diagnostics(self, state: "DataArrayDict") -> "DataArrayDict":
        raw_diagnostics = {}

        for component in self.coupler.components:
            diagnostic_properties = self.sco_diagnostics.get_properties(
                component
            )
            allocate_diagnostic = self.sco_diagnostics.get_allocator(component)
            if allocate_diagnostic is not None:
                for name in diagnostic_properties:
                    if (
                        name in self.coupler.diagnostic_properties
                        and name not in raw_diagnostics
                    ):
                        raw_diagnostics[name] = allocate_diagnostic(name)
                        raw_diagnostics[name][...] = 0.0

        diagnostics = self.dco_diagnostics.get_dataarray_dict(
            raw_diagnostics, state
        )

        return diagnostics

    def allocate_internal_diagnostics(
        self, state: "DataArrayDict"
    ) -> "DataArrayDict":
        occurrences = {}
        raw_internal_diagnostics = {}

        for component in self.coupler.components:
            diagnostic_properties = self.sco_diagnostics.get_properties(
                component
            )
            for name in diagnostic_properties:
                occurrences.setdefault(name, 0)
                occurrences[name] += 1

        for component in self.coupler.components:
            diagnostic_properties = self.sco_diagnostics.get_properties(
                component
            )
            allocate_diagnostic = self.sco_diagnostics.get_allocator(component)
            if allocate_diagnostic is not None:
                for name in diagnostic_properties:
                    if occurrences[name] > 1:
                        raw_internal_diagnostics[name] = allocate_diagnostic(
                            name
                        )
                        raw_internal_diagnostics[name][...] = 0.0
                        occurrences[name] -= 1

        internal_diagnostics = self.dco_diagnostics.get_dataarray_dict(
            raw_internal_diagnostics, state
        )

        return internal_diagnostics

    def set_out_diagnostics(
        self,
        state: "DataArrayDict",
        out_diagnostics: "DataArrayDict",
        internal_diagnostics: Optional["DataArrayDict"],
    ) -> Sequence["DataArrayDict"]:
        pass
