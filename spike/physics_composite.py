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
class PhysicsComponentComposite:
    """
    Abstract base class whose derived classes automate the
    execution of a set of physical parameterizations, pursuing
    a specific coupling strategy.
    """

    __metaclass__ = abc.ABCMeta

    def __init__(self, *args):
        """
		Constructor.

		Parameters
		----------
		*args : obj
			Instances of :class:`sympl.DiagnosticComponent` or
			:class:`sympl.PrognosticComponent`, representing the
			parameterizations to execute.

		References
		----------
		Donahue, A. S., and P. M. Caldwell. (2018). \
		Impact of physics parameterization ordering in a global atmosphere model. \
		*Journal of Advances in Modeling earth Systems*, *10*:481-499.
		"""
        assert_sequence(args, reftype=(Diagnostic, Tendency))
        self._components_list = self._initialize_components_list(args)

    @property
    def components_list(self):
        """
        Returns
        -------
        tuple :
                Tuple of instances of :class:`sympl.DiagnosticComponent` or
                :class:`sympl.PrognosticComponent`, representing the
                parameterizations to execute.
                As this method is marked as abstract, its implementation is
                delegated to the derived classes.
        """
        return self._components_list

    @property
    @abc.abstractmethod
    def current_state_input_properties(self):
        """
        Return
        ------
        dict :
                Dictionary whose keys are strings denoting model variables
                which must be present in the input dictionary representing
                the state at the current time level, and whose values are
                dictionaries specifying fundamental properties (dims, units)
                for those variables.
        """

    @property
    @abc.abstractmethod
    def provisional_state_input_properties(self):
        """
        Return
        ------
        dict :
                Dictionary whose keys are strings denoting model variables
                which must be present in the input dictionary representing
                any arbitrary provisional state (e.g., the state updated
                by the dynamical core),	and whose values are dictionaries
                specifying fundamental properties (dims, units) for those
                variables.
        """

    @property
    @abc.abstractmethod
    def current_state_output_properties(self):
        """
            Return
            ------
            dict :
        Dictionary whose keys are strings denoting model variables
        which will be surely present in the input dictionary representing
        the state at the current time level when the call operator
        returns, and whose values are dictionaries specifying
        fundamental properties (dims, units) for those variables.
        Note that the state could include some other variables (not used
        by any physical package).
        """

    @property
    @abc.abstractmethod
    def provisional_state_output_properties(self):
        """
            Return
            ------
            dict :
        Dictionary whose keys are strings denoting model variables
        which will be surely present in the input state representing
        the provisional state when the call operator returns, and whose
        values are dictionaries specifying fundamental properties
        (dims, units) for those variables.
        Note that the state could include some other variables (not used
        by any physical package).
        """

    @property
    @abc.abstractmethod
    def tendency_properties(self):
        """
            Return
            ------
            dict :
        Dictionary whose keys are strings denoting tendencies
        which are computed by this object, and whose values are
        dictionaries specifying fundamental properties (dims, units)
        for those tendencies.
        """


class DiagnosticComponentComposite:
    """
        Callable class wrapping and chaining a set of :class:`sympl.DiagnosticComponent`.

        Attributes
        ----------
        input_properties : dict
                Dictionary whose keys are strings denoting model variables
                which should be present in the input state dictionary, and
                whose values are dictionaries specifying fundamental properties
                (dims, units) of those variables.
    diagnostic_properties : dict
        Dictionary whose keys are strings denoting model variables
        retrieved from the input state dictionary, and whose values
        are dictionaries specifying fundamental properties (dims, units)
        of those variables.
        output_properties : dict
        Dictionary whose keys are strings denoting model variables which
        will be present in the input state dictionary when the call operator
        returns, and whose values are dictionaries specifying fundamental
        properties (dims, units) for those variables.
    """

    def __init__(self, *args):
        """
        Parameters
        ----------
        *args :
                The :class:`sympl.Diagnostic`\s to wrap and chain.
        """
        assert_sequence(args, reftype=Diagnostic)
        self._components_list = args

        self.input_properties = get_input_properties(self._components_list)
        self.diagnostic_properties = combine_component_properties(
            self._components_list, "diagnostic_properties"
        )
        self.output_properties = get_output_properties(self._components_list)

    def __call__(self, state):
        """
        Retrieve diagnostics from the input state by sequentially calling
        the wrapped :class:`sympl.DiagnosticComponent`\s, and incrementally
        update the input state with those diagnostics.

        Parameters
        ----------
        state : dict
                The input model state as a dictionary whose keys are strings denoting
                model variables, and whose values are :class:`sympl.DataArray`\s storing
                data for those variables.
        """
        return_dict = {}

        for component in self._components_list:
            diagnostics = component(state)
            state.update(diagnostics)
            return_dict.update(diagnostics)

        return return_dict
