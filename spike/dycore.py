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
class SequentialSplittingDynamicalCore:
    """
    Abstract base class whose derived classes implement different dynamical cores
    pursuing the (symmetrized) sequential-update splitting method.
    """

    # Make the class abstract
    __metaclass__ = abc.ABCMeta

    def __init__(self, grid, moist):
        """
        Constructor.

        Parameters
        ----------
        grid : obj
                The underlying grid, as an instance of
                :class:`~tasmania.grids.grid_xyz.GridXYZ`,
                or one of its derived classes.
        moist : bool
                :obj:`True` for a moist dynamical core, :obj:`False` otherwise.
        """
        self._grid, self._moist = grid, moist

        self._input_checker = InputChecker(self)
        self._output_checker = OutputChecker(self)

    @property
    @abc.abstractmethod
    def input_properties(self):
        """
        Return
        ------
        dict :
                Dictionary whose keys are strings denoting variables which
                should be included in any input state, and whose values
                are fundamental properties (dims, units) of those variables.
        """

    @property
    @abc.abstractmethod
    def output_properties(self):
        """
        Return
        ------
        dict :
                Dictionary whose keys are strings denoting variables which are
                included in the output state, and whose values are fundamental
                properties (dims, units) of those variables.
        """

    def __call__(self, state, timestep):
        """
        Call operator advancing the input state one timestep forward.

        Parameters
        ----------
        state : dict
                Dictionary whose keys are strings denoting input model variables,
                and whose values are :class:`sympl.DataArray`\s storing values
                for those variables.
        timestep : timedelta
                :class:`datetime.timedelta` representing the timestep size, i.e.,
                the amount of time to step forward.

        Return
        ------
        dict :
                Dictionary whose keys are strings denoting output model variables,
                and whose values are :class:`sympl.DataArray`\s storing values
                for those variables.
        """
        # Ensure the input state contains all the required variables
        # in proper units and dimensions
        self._input_checker.check_inputs(state)

        # Initialize the output state
        out_state = {}
        out_state.update(state)

        # Extract Numpy arrays from current state
        in_state_units = {
            name: self.input_properties[name]["units"]
            for name in self.input_properties.keys()
        }
        raw_in_state = make_raw_state(out_state, units=in_state_units)

        # Stepped the model raw state
        raw_out_state = self.array_call(raw_in_state, timestep)

        # Create DataArrays out of the Numpy arrays contained in the stepped state
        out_state_units = {
            name: self.output_properties[name]["units"]
            for name in self.output_properties.keys()
        }
        out_state = make_state(
            raw_out_state, self._grid, units=out_state_units
        )

        # Ensure the time specified in the output state is correct
        out_state["time"] = state["time"] + timestep

        # Ensure the state contains all the required variables
        # in the right dimensions and units
        self._output_checker.check_outputs(
            {name: out_state[name] for name in out_state if name != "time"}
        )

        return out_state

    @abc.abstractmethod
    def array_call(self, raw_state, timestep):
        """
        Parameters
        ----------
        raw_state : dict
                Dictionary whose keys are strings denoting input model
                variables, and whose values are :class:`numpy.ndarray`\s
                storing values for those variables.
        timestep : timedelta
                The timestep size.

        Return
        ------
        dict :
                Dictionary whose keys are strings denoting output model
                variables, and whose values are :class:`numpy.ndarray`\s
                storing values for those variables.
        """

    def update_topography(self, time):
        """
        Update the underlying (time-dependent) topography.

        Parameters
        ----------
        time : obj
                :class:`datetime.timedelta` representing the elapsed simulation time.
        """
        self._grid.update_topography(time)
