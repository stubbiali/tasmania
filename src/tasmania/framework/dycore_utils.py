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

from __future__ import annotations
from typing import TYPE_CHECKING

from sympl._core.static_operators import StaticComponentOperator

from tasmania.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.framework.static_checkers import check_missing_fields, check_properties_are_compatible

if TYPE_CHECKING:
    from typing import Optional

    from tasmania.framework.dycore import DynamicalCore
    from tasmania.utils.typingx import Component
    from tasmania.utils.typingx import DataArrayDict, PropertyDict


class StaticOperator:
    @classmethod
    def get_input_properties(cls, dycore: DynamicalCore) -> PropertyDict:
        return_dict = {}

        if dycore.fast_tendency_component is None:
            sco = StaticComponentOperator.factory("stage_input_properties")
            return_dict.update(sco.get_properties_with_dims(dycore))
        else:
            sco1 = StaticComponentOperator.factory("input_properties")
            return_dict.update(sco1.get_properties_with_dims(dycore.fast_tendency_component))
            sco2 = StaticComponentOperator.factory("diagnostic_properties")
            fast_diag_properties = sco2.get_properties_with_dims(dycore.fast_tendency_component)
            sco3 = StaticComponentOperator.factory("stage_input_properties")
            stage_input_properties = sco3.get_properties_with_dims(dycore)

            # Add to the requirements the variables to feed the stage with
            # and which are not output by the intermediate parameterizations
            unshared_vars = tuple(
                name
                for name in stage_input_properties
                if not (name in fast_diag_properties or name in return_dict)
            )
            for name in unshared_vars:
                return_dict[name] = {}
                return_dict[name].update(stage_input_properties[name])

        if dycore.substeps > 0:
            sco1 = StaticComponentOperator.factory("input_properties")
            superfast_input_properties = (
                {}
                if dycore.superfast_tendency_component is None
                else sco1.get_properties_with_dims(dycore.superfast_tendency_component)
            )
            sco2 = StaticComponentOperator.factory("diagnostic_properties")
            superfast_diag_properties = (
                {}
                if dycore.superfast_tendency_component is None
                else sco2.get_properties_with_dims(dycore.superfast_tendency_component)
            )

            # Add to the requirements the variables to feed the fast
            # parameterizations with
            unshared_vars = tuple(
                name for name in superfast_input_properties if name not in return_dict
            )
            for name in unshared_vars:
                return_dict[name] = {}
                return_dict[name].update(superfast_input_properties[name])

            # Add to the requirements the variables to feed the substep with
            # and which are not output by the either the intermediate
            # parameterizations or the fast parameterizations
            sco3 = StaticComponentOperator.factory("substep_input_properties")
            substep_input_properties = sco3.get_properties_with_dims(dycore)
            unshared_vars = tuple(
                name
                for name in substep_input_properties
                if not (name in superfast_diag_properties or name in return_dict)
            )
            for name in unshared_vars:
                return_dict[name] = {}
                return_dict[name].update(substep_input_properties[name])

        return return_dict

    @classmethod
    def get_input_tendency_properties(cls, dycore: DynamicalCore) -> PropertyDict:
        return_dict = {}

        if dycore.fast_tendency_component is None:
            sco = StaticComponentOperator.factory("stage_tendency_properties")
            return_dict.update(sco.get_properties_with_dims(dycore))
        else:
            sco1 = StaticComponentOperator.factory("tendency_properties")
            return_dict.update(sco1.get_properties_with_dims(dycore.fast_tendency_component))

            # Add to the requirements on the input slow tendencies those
            # tendencies to feed the dycore with and which are not provided
            # by the intermediate parameterizations
            sco2 = StaticComponentOperator.factory("stage_tendency_properties")
            stage_tendency_properties = sco2.get_properties_with_dims(dycore)
            unshared_vars = tuple(
                name for name in stage_tendency_properties if name not in return_dict
            )
            for name in unshared_vars:
                return_dict[name] = {}
                return_dict[name].update(stage_tendency_properties[name])

        return return_dict

    @classmethod
    def get_output_properties(cls, dycore: DynamicalCore) -> PropertyDict:
        return_dict = {}

        if dycore.substeps == 0:
            # Add to the return dictionary the variables included in
            # the state output by a stage
            sco = StaticComponentOperator.factory("stage_output_properties")
            return_dict.update(sco.get_properties_with_dims(dycore))
        else:
            # Add to the return dictionary the variables included in
            # the state output by a substep
            sco1 = StaticComponentOperator.factory("substep_output_properties")
            return_dict.update(sco1.get_properties_with_dims(dycore))

            if dycore.superfast_diagnostic_component is not None:
                # Add the fast diagnostics to the return dictionary
                sco2 = StaticComponentOperator.factory("diagnostic_properties")
                superfast_diag_properties = sco2.get_properties_with_dims(
                    dycore.superfast_diagnostic_component
                )
                for name, properties in superfast_diag_properties.items():
                    return_dict[name] = {}
                    return_dict[name].update(properties)

            # Add to the return dictionary the non-substepped variables
            sco3 = StaticComponentOperator.factory("stage_output_properties")
            return_dict.update(sco3.get_properties_with_dims(dycore))

        if dycore.fast_diagnostic_component is not None:
            sco4 = StaticComponentOperator.factory("diagnostic_properties")
            fast_diag_properties = sco4.get_properties_with_dims(dycore.fast_diagnostic_component)

            # Add the retrieved diagnostics to the return dictionary
            for name, properties in fast_diag_properties.items():
                return_dict[name] = {}
                return_dict[name].update(properties)

        return return_dict

    @classmethod
    def wrap_component(
        cls, dycore: DynamicalCore, component: Optional[Component]
    ) -> Optional[ConcurrentCoupling]:
        if component is None:
            return component
        else:
            return ConcurrentCoupling(
                component,
                enable_checks=dycore.enable_checks,
                backend=dycore.backend,
                backend_options=dycore.backend_options,
                storage_options=dycore.storage_options,
            )

    @classmethod
    def get_ovewrite_tendencies(cls, dycore: DynamicalCore) -> dict[str, bool]:
        fast_tc = dycore.fast_tendency_component
        fast_dc = dycore.fast_diagnostic_component
        if fast_tc is not None and fast_dc is not None:
            out = {name: True for name in fast_dc.tendency_properties}
        else:
            out = {}
        return out


class StaticChecker:
    @classmethod
    def check_fast_tendency_component(cls, dycore: DynamicalCore) -> None:
        if dycore.fast_tendency_component is not None and not isinstance(
            dycore.fast_tendency_component, dycore.allowed_tendency_type
        ):
            raise TypeError(
                f"fast_tendency_component of {dycore.__class__.__name__} "
                f"is of type "
                f"{dycore.fast_tendency_component.__class__.__name__} "
                f"but should be of type "
                f"{', '.join(el.__name__ for el in dycore.allowed_tendency_type)}."
            )

    @classmethod
    def check_fast_diagnostic_component(cls, dycore: DynamicalCore) -> None:
        if dycore.fast_diagnostic_component is not None and not isinstance(
            dycore.fast_diagnostic_component, dycore.allowed_diagnostic_type
        ):
            raise TypeError(
                f"fast_diagnostic_component of {dycore.__class__.__name__} "
                f"is of type "
                f"{dycore.fast_diagnostic_component.__class__.__name__} "
                f"but should be of type "
                f"{', '.join(el.__name__ for el in dycore.allowed_diagnostic_type)}."
            )

    @classmethod
    def check_superfast_tendency_component(cls, dycore: DynamicalCore) -> None:
        if dycore.substeps > 0:
            if dycore.superfast_tendency_component is not None and not isinstance(
                dycore.superfast_tendency_component,
                dycore.allowed_tendency_type,
            ):
                raise TypeError(
                    f"superfast_tendency_component of "
                    f"{dycore.__class__.__name__} is of type "
                    f"{dycore.superfast_tendency_component.__class__.__name__} "
                    f"but should be of type "
                    f"{', '.join(el.__name__ for el in dycore.allowed_tendency_type)}."
                )

    @classmethod
    def check_superfast_diagnostic_component(cls, dycore: DynamicalCore) -> None:
        if dycore.substeps > 0:
            if dycore.superfast_diagnostic_component is not None and not isinstance(
                dycore.superfast_diagnostic_component,
                dycore.allowed_diagnostic_type,
            ):
                raise TypeError(
                    f"superfast_diagnostic_component of "
                    f"{dycore.__class__.__name__} is of type "
                    f"{dycore.superfast_diagnostic_component.__class__.__name__} "
                    f"but should be of type "
                    f"{', '.join(el.__name__ for el in dycore.allowed_diagnostic_type)}."
                )

    @classmethod
    def check01(cls, dycore: DynamicalCore) -> None:
        """
        Variables contained in both ``stage_input_properties`` and
        ``stage_output_properties`` should have compatible properties
        across the two dictionaries.
        """
        check_properties_are_compatible(
            dycore, "stage_input_properties", dycore, "stage_output_properties"
        )

    @classmethod
    def check02(cls, dycore: DynamicalCore) -> None:
        """
        Variables contained in both ``substep_input_properties`` and
        ``substep_output_properties`` should have compatible properties
        across the two dictionaries.
        """
        check_properties_are_compatible(
            dycore,
            "substep_input_properties",
            dycore,
            "substep_output_properties",
        )

    @classmethod
    def check03(cls, dycore: DynamicalCore) -> None:
        """
        Variables contained in both ``stage_input_properties`` and the
        ``input_properties`` dictionary of ``fast_tendency_component``
        should have compatible properties across the two dictionaries.
        """
        if dycore.fast_tendency_component is not None:
            check_properties_are_compatible(
                dycore.fast_tendency_component,
                "input_properties",
                dycore,
                "stage_input_properties",
            )

    @classmethod
    def check04(cls, dycore: DynamicalCore) -> None:
        """
        Dimensions and units of the variables diagnosed by
        ``fast_tendency_component`` should be compatible with
        the dimensions and units specified in ``stage_input_properties``.
        """
        if dycore.fast_tendency_component is not None:
            check_properties_are_compatible(
                dycore.fast_tendency_component,
                "diagnostic_properties",
                dycore,
                "stage_input_properties",
            )

    @classmethod
    def check05(cls, dycore: DynamicalCore) -> None:
        """
        Any intermediate tendency calculated by
        ``fast_tendency_component`` should be present in the
        ``stage_tendency_properties`` dictionary,
        with compatible dimensions and units.
        """
        if dycore.fast_tendency_component is not None:
            check_properties_are_compatible(
                dycore.fast_tendency_component,
                "tendency_properties",
                dycore,
                "stage_tendency_properties",
            )
            # check_missing_fields(
            #     dycore.fast_tendency_component,
            #     "tendency_properties",
            #     dycore,
            #     "stage_tendency_properties",
            # )

    @classmethod
    def check06(cls, dycore: DynamicalCore) -> None:
        """
        Dimensions and units of the variables diagnosed by
        ``fast_tendency_component`` should be compatible with
        the dimensions and units specified in the ``input_properties``
        dictionary of ``superfast_tendency_component``, or the
        ``substep_input_properties`` dictionary if
        ``superfast_tendency_component`` is not given.
        """
        if dycore.fast_tendency_component is not None:
            if dycore.superfast_tendency_component is not None:
                check_properties_are_compatible(
                    dycore.fast_tendency_component,
                    "diagnostic_properties",
                    dycore.superfast_tendency_component,
                    "input_properties",
                )
            else:
                check_properties_are_compatible(
                    dycore.fast_tendency_component,
                    "diagnostic_properties",
                    dycore,
                    "substep_input_properties",
                )

    @classmethod
    def check07(cls, dycore: DynamicalCore) -> None:
        """
        Variables diagnosed by ``superfast_tendency_component`` should have
        dimensions and units compatible with those specified in the
        ``substep_input_properties`` dictionary.
        """
        if dycore.superfast_tendency_component is not None:
            check_properties_are_compatible(
                dycore.superfast_tendency_component,
                "diagnostic_properties",
                dycore,
                "substep_input_properties",
            )

    @classmethod
    def check08(cls, dycore: DynamicalCore) -> None:
        """
        Variables contained in ``stage_output_properties`` for which
        ``superfast_tendency_component`` prescribes a (fast) tendency should
        have dimensions and units compatible with those specified
        in the ``tendency_properties`` dictionary of
        ``superfast_tendency_component``.
        """
        if dycore.superfast_tendency_component is not None:
            check_properties_are_compatible(
                dycore.superfast_tendency_component,
                "tendency_properties",
                dycore,
                "stage_output_properties",
                units_suffix=" s",
            )

    @classmethod
    def check09(cls, dycore: DynamicalCore) -> None:
        """
        Any fast tendency calculated by ``superfast_tendency_component``
        should be present in the ``substep_tendency_properties``
        dictionary, with compatible dimensions and units.
        """
        if dycore.superfast_tendency_component is not None:
            check_properties_are_compatible(
                dycore.superfast_tendency_component,
                "tendency_properties",
                dycore,
                "substep_tendency_properties",
            )
            check_missing_fields(
                dycore.superfast_tendency_component,
                "tendency_properties",
                dycore,
                "substep_tendency_properties",
            )

    @classmethod
    def check10(cls, dycore: DynamicalCore) -> None:
        """
        Any variable for which``superfast_tendency_component``
        prescribes a (fast) tendency should be present both in
        the ``substep_input_property`` and ``substep_output_property``
        dictionaries, with compatible dimensions and units.
        """
        if dycore.superfast_tendency_component is not None:
            check_properties_are_compatible(
                dycore.superfast_tendency_component,
                "tendency_properties",
                dycore,
                "substep_input_properties",
                units_suffix=" s",
            )
            check_missing_fields(
                dycore.superfast_tendency_component,
                "tendency_properties",
                dycore,
                "substep_input_properties",
            )
            check_properties_are_compatible(
                dycore.superfast_tendency_component,
                "tendency_properties",
                dycore,
                "substep_output_properties",
                units_suffix=" s",
            )
            check_missing_fields(
                dycore.superfast_tendency_component,
                "tendency_properties",
                dycore,
                "substep_output_properties",
            )

    @classmethod
    def check11(cls, dycore: DynamicalCore) -> None:
        """
        Any variable being expected by ``superfast_diagnostic_component``
        should be present in ``substep_output_properties``, with compatible
        dimensions and units.
        """
        if dycore.superfast_diagnostic_component is not None:
            check_properties_are_compatible(
                dycore.superfast_diagnostic_component,
                "input_properties",
                dycore,
                "substep_output_properties",
            )
            check_missing_fields(
                dycore.superfast_diagnostic_component,
                "input_properties",
                dycore,
                "substep_output_properties",
            )

    @classmethod
    def check12(cls, dycore: DynamicalCore) -> None:
        """
        Any variable being expected by ``fast_diagnostic_component``
        should be present either in ``stage_output_properties`` or
        ``substep_output_properties``, with compatible dimensions and units.
        """
        if dycore.fast_diagnostic_component is not None:
            check_properties_are_compatible(
                dycore.fast_diagnostic_component,
                "input_properties",
                dycore,
                "fused_output_properties",
            )
            check_missing_fields(
                dycore.fast_diagnostic_component,
                "input_properties",
                dycore,
                "fused_output_properties",
            )

    @classmethod
    def check13(cls, dycore: DynamicalCore) -> None:
        """
        ``stage_array_call`` should be able to handle any tendency
        prescribed by ``fast_tendency_component``.
        """
        if dycore.fast_diagnostic_component is not None:
            check_properties_are_compatible(
                dycore.fast_diagnostic_component,
                "tendency_properties",
                dycore,
                "stage_tendency_properties",
            )
            check_missing_fields(
                dycore.fast_diagnostic_component,
                "tendency_properties",
                dycore,
                "stage_tendency_properties",
            )

    @classmethod
    def check(cls, dycore: DynamicalCore) -> None:
        cls.check_fast_tendency_component(dycore)
        cls.check_fast_diagnostic_component(dycore)
        cls.check_superfast_tendency_component(dycore)
        cls.check_superfast_diagnostic_component(dycore)
        cls.check01(dycore)
        cls.check02(dycore)
        cls.check03(dycore)
        cls.check04(dycore)
        cls.check05(dycore)
        cls.check06(dycore)
        cls.check07(dycore)
        cls.check08(dycore)
        cls.check09(dycore)
        cls.check10(dycore)
        cls.check11(dycore)
        cls.check12(dycore)
        cls.check13(dycore)


class DynamicOperator:
    def __init__(self, dycore: DynamicalCore):
        self.sco = StaticComponentOperator.factory("tendency_properties")
        self.ftc_tp = self.sco.get_properties(dycore.fast_tendency_component)
        self.fdc_tp = self.sco.get_properties(dycore.fast_diagnostic_component)

    def get_ovewrite_tendencies(self, slow_tendencies: DataArrayDict) -> dict[str, bool]:
        out = {
            name: False for name in self.ftc_tp if name in self.fdc_tp or name in slow_tendencies
        }
        return out

    def get_fast_and_slow_tendencies(
        self, coupler: ConcurrentCoupling, slow_tendencies: DataArrayDict
    ) -> list[str]:
        tendency_properties = self.sco.get_properties(coupler)
        out = [key for key in tendency_properties if key in slow_tendencies]
        return out
