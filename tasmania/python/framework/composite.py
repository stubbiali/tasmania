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
"""
This module contains:
	DiagnosticComponentComposite
"""
from sympl import DiagnosticComponent, combine_component_properties

from tasmania.python.utils.framework_utils import get_input_properties
from tasmania.python.utils.utils import assert_sequence


class DiagnosticComponentComposite:
    """
	Callable class wrapping and chaining a set of :class:`sympl.DiagnosticComponent`\s.

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

    def __init__(self, *args, execution_policy="serial"):
        """
		Parameters
		----------
		*args :
			The :class:`sympl.Diagnostic`\s to wrap and chain.
		execution_policy : `str`, optional
			String specifying the runtime policy according to which parameterizations
			should be invoked. Either:

				* 'serial', to call the physical packages sequentially,
					and add diagnostics to the current state incrementally;
				* 'as_parallel', to call the physical packages *as* we
					were in a parallel region where each package may be
					assigned to a different thread/process; in other terms,
					we do not rely on the order in which parameterizations
					are passed to this object, and diagnostics are not added
					to the current state before returning.

		"""
        assert_sequence(args, reftype=DiagnosticComponent)
        self._components_list = args

        self.input_properties = get_input_properties(
            self._components_list, consider_diagnostics=execution_policy == "serial"
        )
        self.diagnostic_properties = combine_component_properties(
            self._components_list, "diagnostic_properties"
        )

        self._call = (
            self._call_serial if execution_policy == "serial" else self._call_asparallel
        )

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

		Return
		------
		dict :
			Dictionary whose keys are strings denoting diagnostic variables,
			and whose values are :class:`sympl.DataArray`\s storing data for
			those variables.
		"""
        return self._call(state)

    def _call_serial(self, state):
        return_dict = {}

        tmp_state = {}
        tmp_state.update(state)

        for component in self._components_list:
            diagnostics = component(tmp_state)
            tmp_state.update(diagnostics)
            return_dict.update(diagnostics)

        return return_dict

    def _call_asparallel(self, state):
        return_dict = {}

        for component in self._components_list:
            diagnostics = component(state)
            return_dict.update(diagnostics)

        return return_dict
