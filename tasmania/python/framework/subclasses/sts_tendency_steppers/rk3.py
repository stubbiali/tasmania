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
import numpy as np

from tasmania.python.framework.register import register
from tasmania.python.framework.sts_tendency_stepper import STSTendencyStepper
from tasmania.python.utils.framework_utils import get_increment

# @register(name="rk3")
# class RK3(STSTendencyStepper):
#     """
#     The three-stages, third-order Runge-Kutta scheme.
#
#     References
#     ----------
#     Gear, C. W. (1971). *Numerical initial value problems in \
#         ordinary differential equations.* Prentice Hall PTR.
#     """
#
#     def __init__(self, *args, **kwargs):
#         super().__init__(*args, **kwargs)
#
#         # free parameters for RK3
#         self._alpha1 = 1.0 / 2.0
#         self._alpha2 = 3.0 / 4.0
#
#         # set the other parameters yielding a third-order method
#         self._gamma1 = (3.0 * self._alpha2 - 2.0) / (
#             6.0 * self._alpha1 * (self._alpha2 - self._alpha1)
#         )
#         self._gamma2 = (3.0 * self._alpha1 - 2.0) / (
#             6.0 * self._alpha2 * (self._alpha1 - self._alpha2)
#         )
#         self._gamma0 = 1.0 - self._gamma1 - self._gamma2
#         self._beta21 = self._alpha2 - 1.0 / (6.0 * self._alpha1 * self._gamma2)
#
#     def _call(self, state, prv_state, timestep):
#         # shortcuts
#         out_units = {
#             name: properties["units"]
#             for name, properties in self.output_properties.items()
#         }
#         a1, a2 = self._alpha1, self._alpha2
#         b21 = self._beta21
#         g0, g1, g2 = self._gamma0, self._gamma1, self._gamma2
#         dt = timestep.total_seconds()
#
#         # first stage
#         k0, diagnostics = get_increment(state, timestep, self.prognostic)
#         dtk0 = multiply(dt, k0)
#         state_1 = add(
#             multiply(1 - a1, state),
#             multiply(a1, add(prv_state, dtk0, unshared_variables_in_output=False)),
#             units=out_units,
#             unshared_variables_in_output=True,
#         )
#         state_1["time"] = state["time"] + a1 * timestep
#
#         if self._enforce_hb:
#             # enforce the boundary conditions on each prognostic variable
#             self._hb.enforce(
#                 state_1, field_names=self.output_properties.keys(), grid=self._grid
#             )
#
#         # second stage
#         k1, _ = get_increment(state_1, timestep, self.prognostic)
#         dtk1 = multiply(dt, k1)
#         state_2 = add(
#             multiply(1 - a2, state),
#             add(
#                 multiply(a2, add(prv_state, dtk1, unshared_variables_in_output=False)),
#                 multiply(b21, subtract(dtk0, dtk1, unshared_variables_in_output=False)),
#                 unshared_variables_in_output=False,
#             ),
#             units=out_units,
#             unshared_variables_in_output=True,
#         )
#         state_2["time"] = state["time"] + a2 * timestep
#
#         if self._enforce_hb:
#             # enforce the boundary conditions on each prognostic variable
#             self._hb.enforce(
#                 state_2, field_names=self.output_properties.keys(), grid=self._grid
#             )
#
#         # third stage
#         k2, _ = get_increment(state_2, timestep, self.prognostic)
#         dtk2 = multiply(dt, k2)
#         dtk1k2 = add(multiply(g1, dtk1), multiply(g2, dtk2))
#         dtk0k1k2 = add(multiply(g0, dtk0), dtk1k2)
#         out_state = add(
#             prv_state, dtk0k1k2, units=out_units, unshared_variables_in_output=False
#         )
#         out_state["time"] = state["time"] + timestep
#
#         if self._enforce_hb:
#             # enforce the boundary conditions on each prognostic variable
#             self._hb.enforce(
#                 out_state, field_names=self.output_properties.keys(), grid=self._grid
#             )
#
#         # restore original units of the tendencies
#         restore_tendency_units(k0)
#         restore_tendency_units(k1)
#         restore_tendency_units(k2)
#
#         return diagnostics, out_state
