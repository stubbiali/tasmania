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
from sympl._core.units import clean_units

from tasmania.python.framework.tendency_checkers import SupersetTendencyChecker


allowed_grid_types = ("physical", "numerical")


class DiagnosticComponent(sympl.DiagnosticComponent):
    """
    Customized version of :class:`sympl.DiagnosticComponent` which keeps track
    of the grid over which the component is instantiated.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, domain, grid_type="numerical"):
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
        self._hb = domain.horizontal_boundary
        super().__init__()

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
    def horizontal_boundary(self):
        """
        Returns
        -------
        tasmania.HorizontalBoundary :
            The object handling the lateral boundary conditions.
        """
        return self._hb


class ImplicitTendencyComponent(sympl.ImplicitTendencyComponent):
    """
    Customized version of :class:`sympl.ImplicitTendencyComponent` which keeps track
    of the grid over which the component is instantiated.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self, domain, grid_type="numerical", tendencies_in_diagnostics=False, name=None
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

        tendencies_in_diagnostics : `bool`, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : `str`, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
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
        self._hb = domain.horizontal_boundary
        super().__init__(tendencies_in_diagnostics, name)

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
    def horizontal_boundary(self):
        """
        Returns
        -------
        tasmania.HorizontalBoundary :
            The object handling the lateral boundary conditions.
        """
        return self._hb


class Stepper(sympl.Stepper):
    """
    Customized version of :class:`sympl.Stepper` which keeps track
    of the grid over which the component is instantiated.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self, domain, grid_type="numerical", tendencies_in_diagnostics=False, name=None
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

        tendencies_in_diagnostics : `bool`, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : `str`, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
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
        self._hb = domain.horizontal_boundary
        super().__init__(tendencies_in_diagnostics, name)

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
    def horizontal_boundary(self):
        """
        Returns
        -------
        tasmania.HorizontalBoundary :
            The object handling the lateral boundary conditions.
        """
        return self._hb


class TendencyComponent(sympl.TendencyComponent):
    """
    Customized version of :class:`sympl.TendencyComponent` which keeps track
    of the grid over which the component is instantiated.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(
        self, domain, grid_type="numerical", tendencies_in_diagnostics=False, name=None
    ):
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The underlying domain.
        grid_type : `str`, optional
            The type of grid over which instantiating the class. Either:

                * 'physical';
                * 'numerical' (default).

        tendencies_in_diagnostics : `bool`, optional
            A boolean indicating whether this object will put tendencies of
            quantities in its diagnostic output.
        name : `str`, optional
            A label to be used for this object, for example as would be used for
            Y in the name "X_tendency_from_Y". By default the class name in
            lowercase is used.
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
        self._hb = domain.horizontal_boundary
        super().__init__(tendencies_in_diagnostics, name)

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
    def horizontal_boundary(self):
        """
        Returns
        -------
        tasmania.HorizontalBoundary :
            The object handling the lateral boundary conditions.
        """
        return self._hb


class TendencyPromoter(abc.ABC):
    """ Promote a tendency to a state variable. """

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

        # needed by sympl's TendencyChecker
        self.input_properties = {}
        for name, props in self.tendency_properties.items():
            assert "dims" in props
            assert "units" in props
            self.input_properties[name] = {
                "dims": props["dims"],
                "units": clean_units(props["units"] + " s"),
            }

        self._tendency_checker = SupersetTendencyChecker(self)

    @property
    @abc.abstractmethod
    def tendency_properties(self):
        pass

    def __call__(self, state, tendencies):
        """
        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The dictionary of model variables.
        tendencies : dict[str, sympl.DataArray]
            The dictionary of tendencies.
        """
        self._tendency_checker.check_tendencies(tendencies)

        for name, props in self.tendency_properties.items():
            dims = props["dims"]
            units = props["units"]

            if any(src != trg for src, trg in zip(tendencies[name].dims, dims)):
                state[name] = tendencies[name].transpose(*dims).to_units(units)
            else:
                state[name] = tendencies[name].to_units(units)
