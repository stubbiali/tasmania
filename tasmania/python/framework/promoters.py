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


allowed_grid_types = ("physical", "numerical")


class Tendency2Diagnostic(abc.ABC):
    """ Promote a tendency to a (diagnostic) state variable. """

    def __init__(self, domain, grid_type):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

        """
        assert (
            grid_type in allowed_grid_types
        ), "grid_type is {}, but either ({}) was expected.".format(
            grid_type, ",".join(allowed_grid_types)
        )
        self._grid_type = grid_type
        self._grid = (
            domain.physical_grid if grid_type == "physical" else domain.numerical_grid
        )

        self._input_checker = InputChecker(self)

        self.diagnostic_properties = {}
        for name, props in self.input_properties.items():
            diag_name = props.get("diagnostic_name", "tendency_of_" + name)
            self.diagnostic_properties[diag_name] = {
                "dims": props["dims"],
                "units": props["units"],
            }

        # compliance with TendencyComponent
        self.tendency_properties = {}

    @property
    def grid_type(self):
        """
        Returns
        -------
        str :
            The grid type, either 'physical' or 'numerical'.
        """
        return self._grid_type

    @property
    def grid(self):
        """
        Returns
        -------
        tasmania.Grid :
            The underlying grid.
        """
        return self._grid

    @property
    @abc.abstractmethod
    def input_properties(self):
        pass

    def __call__(self, tendencies):
        """
        Parameters
        ----------
        tendencies : dict[str, sympl.DataArray]
            The dictionary of tendencies.

        Return
        ------
        dict[str, sympl.DataArray] :
            The dictionary of promoted tendencies.
        """
        self._input_checker.check_inputs(tendencies)

        diagnostics = {}

        for name, props in self.input_properties.items():
            dims = props["dims"]
            units = props["units"]
            diag_name = props.get("diagnostic_name", "tendency_of_" + name)
            rm = props.get("remove_from_tendencies", False)

            if any(src != trg for src, trg in zip(tendencies[name].dims, dims)):
                diagnostics[diag_name] = tendencies[name].transpose(*dims).to_units(units)
            else:
                diagnostics[diag_name] = tendencies[name].to_units(units)

            if rm:
                tendencies.pop(name)

        return diagnostics


class Diagnostic2Tendency(abc.ABC):
    """ Promote a diagnostic variable to a tendency. """

    def __init__(self, domain, grid_type):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

        """
        assert (
            grid_type in allowed_grid_types
        ), "grid_type is {}, but either ({}) was expected.".format(
            grid_type, ",".join(allowed_grid_types)
        )
        self._grid_type = grid_type
        self._grid = (
            domain.physical_grid if grid_type == "physical" else domain.numerical_grid
        )

        self._input_checker = InputChecker(self)

        self.tendency_properties = {}
        for name, props in self.input_properties.items():
            tend_name = props.get("tendency_name", name.replace("tendency_of_", ""))
            self.tendency_properties[tend_name] = {
                "dims": props["dims"],
                "units": props["units"],
            }

        # compliance with DiagnosticComponent
        self.diagnostic_properties = {}

    @property
    def grid_type(self):
        """
        Returns
        -------
        str :
            The grid type, either 'physical' or 'numerical'.
        """
        return self._grid_type

    @property
    def grid(self):
        """
        Returns
        -------
        tasmania.Grid :
            The underlying grid.
        """
        return self._grid

    @property
    @abc.abstractmethod
    def input_properties(self):
        pass

    def __call__(self, diagnostics):
        """
        Parameters
        ----------
        diagnostics : dict[str, sympl.DataArray]
            The dictionary of diagnostics.

        Return
        ------
        dict[str, sympl.DataArray] :
            The dictionary of promoted diagnostics.
        """
        self._input_checker.check_inputs(diagnostics)

        tendencies = {}

        for name, props in self.input_properties.items():
            dims = props["dims"]
            units = props["units"]
            tend_name = props.get("tendency_name", name.replace("tendency_of_", ""))
            rm = props.get("remove_from_diagnostics", False)

            if any(src != trg for src, trg in zip(diagnostics[name].dims, dims)):
                tendencies[tend_name] = diagnostics[name].transpose(*dims).to_units(units)
            else:
                tendencies[tend_name] = diagnostics[name].to_units(units)

            if rm:
                diagnostics.pop(name)

        return tendencies
