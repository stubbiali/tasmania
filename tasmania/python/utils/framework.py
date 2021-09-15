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
import sympl
from sympl._core.combine_properties import (
    combine_dims,
    units_are_compatible,
    InvalidPropertyDictError,
)
from sympl._core.units import clean_units
from typing import (
    Any,
    Dict,
    Mapping,
    Optional,
    Sequence,
    TYPE_CHECKING,
    Tuple,
    Type,
    TypeVar,
)

from tasmania.python.utils import typingx

if TYPE_CHECKING:
    from tasmania.python.framework._base import BaseFromTendencyToDiagnostic


T = TypeVar("T")


def check_properties_compatibility(
    properties1: typingx.PropertiesDict,
    properties2: typingx.PropertiesDict,
    to_append: Optional[str] = None,
    properties1_name: Optional[str] = None,
    properties2_name: Optional[str] = None,
) -> None:
    _properties1 = {}
    if to_append is None:
        _properties1.update(properties1)
    else:
        _properties1.update(
            {
                name: {
                    "dims": value["dims"],
                    "units": clean_units(value["dims"] + to_append),
                }
                for name, value in properties1.items()
            }
        )

    shared_vars = set(_properties1.keys()).intersection(properties2.keys())
    for name in shared_vars:
        check_property_compatibility(
            properties1[name],
            properties2[name],
            property_name=name,
            origin1_name=properties1_name,
            origin2_name=properties2_name,
        )


def check_property_compatibility(
    property1: Mapping[str, Any],
    property2: Mapping[str, Any],
    property_name: Optional[str] = None,
    origin1_name: Optional[str] = None,
    origin2_name: Optional[str] = None,
) -> None:
    if (
        "dims" not in property1.keys()
        or "units" not in property1.keys()
        or "dims" not in property2.keys()
        or "units" not in property2.keys()
    ):
        raise InvalidPropertyDictError()

    try:
        _ = combine_dims(property1["dims"], property2["dims"])
    except InvalidPropertyDictError:
        raise InvalidPropertyDictError(
            "Incompatibility between dims {} (in {}) and {} (in {}) of quantity {}.".format(
                property1["dims"],
                "properties1" if origin1_name is None else origin1_name,
                property2["dims"],
                "properties2" if origin2_name is None else origin2_name,
                "unknown" if property_name is None else property_name,
            )
        )

    if not units_are_compatible(property1["units"], property2["units"]):
        raise InvalidPropertyDictError(
            "Incompatibility between units {} (in {}) and {} (in {}) of quantity {}.".format(
                property1["units"],
                "properties1" if origin1_name is None else origin1_name,
                property2["units"],
                "properties2" if origin2_name is None else origin2_name,
                "unknown" if property_name is None else property_name,
            )
        )


def check_missing_properties(
    properties1: typingx.PropertiesDict,
    properties2: typingx.PropertiesDict,
    properties1_name: Optional[str] = None,
    properties2_name: Optional[str] = None,
) -> None:
    missing_vars = set(properties1.keys()).difference(properties2.keys())

    if len(missing_vars) > 0:
        raise InvalidPropertyDictError(
            "{} are present in {} but missing in {}.".format(
                ", ".join(missing_vars),
                "properties1"
                if properties1_name is None
                else properties1_name,
                "properties2"
                if properties2_name is None
                else properties2_name,
            )
        )


def resolve_aliases(
    data_dict: typingx.DataArrayDict, properties_dict: typingx.PropertiesDict
) -> typingx.DataArrayDict:
    name_to_alias = _get_name_to_alias_map(data_dict, properties_dict)
    return _replace_aliases(data_dict, name_to_alias)


def _get_name_to_alias_map(
    data_dict: typingx.DataArrayDict, properties_dict: typingx.PropertiesDict
) -> Dict[str, str]:
    return_dict = {}

    for name in properties_dict:
        aliases = [name]
        if properties_dict[name].get("alias", None) is not None:
            aliases.append(properties_dict[name]["alias"])

        for alias in aliases:
            if alias in data_dict:
                return_dict[name] = alias
                break
            else:
                pass

        assert name in return_dict

    return return_dict


def _replace_aliases(
    data_dict: typingx.DataArrayDict, name_to_alias: Mapping[str, str]
) -> typingx.DataArrayDict:
    return_dict = {}

    for name in name_to_alias:
        if name != name_to_alias[name]:
            return_dict[name] = data_dict[name_to_alias[name]]

    return return_dict


def get_input_properties(
    components_list: Sequence[Dict[str, Any]],
    return_dict: Optional[typingx.PropertiesDict] = None,
) -> typingx.PropertiesDict:
    # Initialize the return dictionary, i.e., the list of requirements
    return_dict = return_dict or {}

    # Initialize the properties of the variables which the input state
    # should include
    output_properties = {}

    for component_properties in components_list:
        component = component_properties.get("component", None)
        attribute_name = component_properties.get("attribute_name", None)
        consider_diagnostics = component_properties.get(
            "consider_diagnostics", True
        )

        if component is not None:
            # Extract the desired property dictionary from the component
            component_dict = (
                getattr(component, attribute_name)
                if attribute_name is not None
                else {}
            )

            # Ensure the requirements of the component are compatible
            # with the variables already at disposal
            check_properties_compatibility(
                output_properties,
                component_dict,
                properties1_name="{} of {}".format(
                    attribute_name,
                    getattr(component, "name", str(component.__class__)),
                ),
                properties2_name="output_properties",
            )

            # Check if there exists any variable which the component
            # requires but which is not yet at disposal
            not_at_disposal = set(component_dict.keys()).difference(
                output_properties.keys()
            )

            for name in not_at_disposal:
                # Add the missing variable to the requirements and
                # to the output state
                return_dict[name] = {}
                return_dict[name].update(component_dict[name])
                output_properties[name] = {}
                output_properties[name].update(component_dict[name])

            if consider_diagnostics:
                # Use the diagnostics calculated by the component to update
                # the properties of the output variables
                for (
                    name,
                    properties,
                ) in component.diagnostic_properties.items():
                    if name not in output_properties.keys():
                        output_properties[name] = {}
                    else:
                        check_property_compatibility(
                            output_properties[name],
                            properties,
                            property_name=name,
                            origin1_name="output_properties",
                            origin2_name="diagnostic_properties of {}".format(
                                getattr(
                                    component, "name", str(component.__class__)
                                )
                            ),
                        )

                    output_properties[name].update(properties)

    return return_dict


def get_tendency_properties(
    components_list: Sequence[typingx.Component],
    t2d_type: "Type[BaseFromTendencyToDiagnostic]",
) -> typingx.PropertiesDict:
    """Combine the tendency_properties dictionaries from multiple components."""
    return_dict = {}

    for component in components_list:
        if isinstance(component, t2d_type):  # tendency-to-diagnostic promoter
            inputs = component.input_properties
            for name, props in inputs.items():
                if props.get("remove_from_tendencies", False):
                    return_dict.pop(name, None)
        else:
            tendencies = getattr(component, "tendency_properties", {})

            already_at_disposal = tuple(
                key for key in tendencies if key in return_dict
            )
            for name in already_at_disposal:
                check_property_compatibility(
                    tendencies[name],
                    return_dict[name],
                    origin1_name=type(component),
                    origin2_name="return_dict",
                )

            return_dict.update(tendencies)

    return return_dict


def get_output_properties(
    components_list: Sequence[Dict[str, Any]],
    return_dict: Optional[typingx.PropertiesDict] = None,
) -> typingx.PropertiesDict:
    """
    Ansatz: the output property dictionary of a :class:`sympl.TendencyStepper`
    component is a subset of its input property component.
    """
    # Initialize the return dictionary
    return_dict = {} if return_dict is None else return_dict

    for component_properties in components_list:
        component = component_properties.get("component", None)
        attribute_name = component_properties.get("attribute_name", None)
        consider_diagnostics = component_properties.get(
            "consider_diagnostics", True
        )

        if component is not None:
            # Extract the desired property dictionary from the component
            component_dict = (
                getattr(component, attribute_name)
                if attribute_name is not None
                else {}
            )

            # Ensure the requirements of the component are compatible
            # with the variables already at disposal
            check_properties_compatibility(
                return_dict,
                component_dict,
                properties1_name="return_dict",
                properties2_name="{} of {}".format(
                    attribute_name,
                    getattr(component, "name", str(component.__class__)),
                ),
            )

            # Check if there exists any variable which the component
            # requires but which is not yet at disposal
            not_at_disposal = set(component_dict.keys()).difference(
                return_dict.keys()
            )

            for name in not_at_disposal:
                # Add the missing variable to the return dictionary
                return_dict[name] = {}
                return_dict[name].update(component_dict[name])

            # Consider the diagnostics calculated by the component to update
            # the return dictionary
            if consider_diagnostics:
                for (
                    name,
                    properties,
                ) in component.diagnostic_properties.items():
                    if name not in return_dict.keys():
                        return_dict[name] = {}
                    else:
                        check_property_compatibility(
                            return_dict[name],
                            properties,
                            property_name=name,
                            origin1_name="return_dict",
                            origin2_name="diagnostic_properties of {}".format(
                                getattr(
                                    component, "name", str(component.__class__)
                                )
                            ),
                        )

                    return_dict[name].update(properties)

    return return_dict


def check_t2d(
    components_list: Sequence[typingx.Component],
    t2d_type: "Type[BaseFromTendencyToDiagnostic]",
):
    """Ensure that a tendency is actually computed before moving it around."""
    tendencies = {}

    for component in components_list:
        if isinstance(component, t2d_type):
            requirements = component.input_properties
            check_missing_properties(
                requirements,
                tendencies,
                properties1_name=str(component.__class__),
                properties2_name="tendencies_at_disposal",
            )
        else:
            tendencies.update(getattr(component, "tendency_properties", {}))


def get_increment(
    state: typingx.DataArrayDict,
    timestep: typingx.TimeDelta,
    prognostic: sympl._core.core_components.TendencyComponent,
) -> Tuple[typingx.DataArrayDict, typingx.DataArrayDict]:
    # calculate tendencies and retrieve diagnostics
    tendencies, diagnostics = prognostic(state, timestep)

    # "multiply" the tendencies by the time step
    for name in tendencies:
        if name != "time":
            tendencies[name].attrs["units"] += " s"

    return tendencies, diagnostics


def restore_tendency_units(
    tendencies: typingx.mutable_dataarray_dict_t,
) -> None:
    for name in tendencies:
        if name != "time":
            tendencies[name].attrs["units"] = clean_units(
                tendencies[name].attrs["units"] + " s^-1"
            )
