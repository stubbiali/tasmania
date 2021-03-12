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
import sympl
from sympl._core.base_components import InputChecker
from typing import TYPE_CHECKING

from tasmania.python.framework._base import (
    BaseDiagnostic2Tendency,
    BaseTendency2Diagnostic,
)
from tasmania.python.utils import typing as ty

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain
    from tasmania.python.domain.grid import Grid


allowed_grid_types = ("physical", "numerical")


class Diagnostic2Tendency(BaseDiagnostic2Tendency):
    """ Promote a variable from a diagnostic dict to a tendency dict. """

    __metaclass__ = abc.ABCMeta

    def __init__(self, domain: "Domain", grid_type: str) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : `str`, optional
            The type of grid over which instantiating the class.
            Either "physical" or "numerical" (default).
        """
        assert (
            grid_type in allowed_grid_types
        ), "grid_type is {}, but either ({}) was expected.".format(
            grid_type, ",".join(allowed_grid_types)
        )
        self._grid_type = grid_type
        self._grid = (
            domain.physical_grid
            if grid_type == "physical"
            else domain.numerical_grid
        )

        self._input_checker = InputChecker(self)

    @property
    def grid_type(self) -> str:
        """ The grid type, either "physical" or "numerical". """
        return self._grid_type

    @property
    def grid(self) -> "Grid":
        """ The underlying :class:`~tasmania.Grid`. """
        return self._grid

    @property
    @abc.abstractmethod
    def input_properties(self) -> ty.PropertiesDict:
        """
        Dictionary whose keys are strings identifying the fields to promote,
        and whose values are dictionaries specifying the following properties
        of those fields:

        * "dims": the field dimensions;
        * "units": the field units;
        * "tendency_name": the key associated to the field in the \
            dictionary of tendencies;
        * "remove_from_diagnostics": ``True`` to pop the field out of the \
            dictionary of diagnostics.
        """
        pass

    @property
    def tendency_properties(self) -> ty.PropertiesDict:
        """
        Dictionary whose keys are strings identifying the quantities returned when
        the object is called, and whose values are dictionaries specifying the
        dimensions ('dims') and units ('units') for those quantities.
        """
        return_dict = {}
        for name, props in self.input_properties.items():
            tend_name = props.get(
                "tendency_name", name.replace("tendency_of_", "")
            )
            return_dict[tend_name] = {
                "dims": props["dims"],
                "units": props["units"],
            }
        return return_dict

    @property
    def diagnostic_properties(self) -> ty.PropertiesDict:
        """
        An empty dictionary. This property is provided only for compliance with
        :class:`~sympl.DiagnosticComponent.`
        """
        return {}

    def __call__(self, diagnostics: ty.DataArrayDict) -> ty.DataArrayDict:
        """
        Parameters
        ----------
        diagnostics : dict[str, sympl.DataArray]
            A dictionary of diagnostics.

        Return
        ------
        dict[str, sympl.DataArray] :
            A dictionary containing the promoted diagnostics.
        """
        self._input_checker.check_inputs(diagnostics)

        tendencies = {}

        for name, props in self.input_properties.items():
            dims = props["dims"]
            units = props["units"]
            tend_name = props.get(
                "tendency_name", name.replace("tendency_of_", "")
            )
            rm = props.get("remove_from_diagnostics", False)

            if any(
                src != trg for src, trg in zip(diagnostics[name].dims, dims)
            ):
                tendencies[tend_name] = (
                    diagnostics[name].transpose(*dims).to_units(units)
                )
            else:
                tendencies[tend_name] = diagnostics[name].to_units(units)

            if rm:
                diagnostics.pop(name)

        return tendencies


class Tendency2Diagnostic(BaseTendency2Diagnostic):
    """ Promote a variable from a tendency dict to a diagnostic dict. """

    __metaclass__ = abc.ABCMeta

    def __init__(self, domain: "Domain", grid_type: str) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        grid_type : `str`, optional
            The type of grid over which instantiating the class.
            Either "physical" or "numerical" (default).
        """
        assert (
            grid_type in allowed_grid_types
        ), "grid_type is {}, but either ({}) was expected.".format(
            grid_type, ",".join(allowed_grid_types)
        )
        self._grid_type = grid_type
        self._grid = (
            domain.physical_grid
            if grid_type == "physical"
            else domain.numerical_grid
        )

        self._input_checker = InputChecker(self)

    @property
    def grid_type(self) -> str:
        """ The grid type, either "physical" or "numerical". """
        return self._grid_type

    @property
    def grid(self) -> "Grid":
        """ The underlying :class:`~tasmania.Grid`. """
        return self._grid

    @property
    @abc.abstractmethod
    def input_properties(self) -> ty.PropertiesDict:
        """
        Dictionary whose keys are strings identifying the fields to promote,
        and whose values are dictionaries specifying the following properties
        of those fields:

        * "dims": the field dimensions;
        * "units": the field units;
        * "diagnostic_name": the key associated to the field in the \
            dictionary of diagnostics;
        * "remove_from_tendencies": ``True`` to pop the field out of the \
            dictionary of tendencies.
        """
        pass

    @property
    def tendency_properties(self) -> ty.PropertiesDict:
        """
        An empty dictionary. This property is provided only for compliance with
        :class:`~sympl.TendencyComponent.`
        """
        return {}

    @property
    def diagnostic_properties(self) -> ty.PropertiesDict:
        """
        Dictionary whose keys are strings identifying the quantities returned when
        the object is called, and whose values are dictionaries specifying the
        dimensions ('dims') and units ('units') for those quantities.
        """
        return_dict = {}
        for name, props in self.input_properties.items():
            diag_name = props.get("diagnostic_name", "tendency_of_" + name)
            return_dict[diag_name] = {
                "dims": props["dims"],
                "units": props["units"],
            }
        return return_dict

    def __call__(self, tendencies: ty.DataArrayDict) -> ty.DataArrayDict:
        """
        Parameters
        ----------
        tendencies : dict[str, sympl.DataArray]
            A dictionary of tendencies.

        Return
        ------
        dict[str, sympl.DataArray] :
            A dictionary containing the promoted tendencies.
        """
        self._input_checker.check_inputs(tendencies)

        diagnostics = {}

        for name, props in self.input_properties.items():
            dims = props["dims"]
            units = props["units"]
            diag_name = props.get("diagnostic_name", "tendency_of_" + name)
            rm = props.get("remove_from_tendencies", False)

            if any(
                src != trg for src, trg in zip(tendencies[name].dims, dims)
            ):
                diagnostics[diag_name] = (
                    tendencies[name].transpose(*dims).to_units(units)
                )
            else:
                diagnostics[diag_name] = tendencies[name].to_units(units)

            if rm:
                tendencies.pop(name)

        return diagnostics
