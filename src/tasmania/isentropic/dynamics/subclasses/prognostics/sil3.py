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


# # convenient aliases
# mfwv = "mass_fraction_of_water_vapor_in_air"
# mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
# mfpw = "mass_fraction_of_precipitation_water_in_air"


# @register(name="sil3")
# class SIL3(IsentropicPrognostic):
#     """ The semi-implicit Lorenz three cycle scheme. """
#
#     def __init__(
#         self,
#         horizontal_flux_scheme,
#         grid,
#         hb,
#         moist,
#         backend,
#         backend_opts,
#         dtype,
#             build_info,
#         exec_info,
#         default_origin,
#         rebuild,
#         managed_memory,
#         **kwargs
#     ):
#         # call parent's constructor
#         super().__init__(
#             IsentropicMinimalHorizontalFlux,
#             horizontal_flux_scheme,
#             grid,
#             hb,
#             moist,
#             backend,
#             backend_opts,
#             dtype,
#             build_info,
#             exec_info,
#             default_origin,
#             rebuild,
#             managed_memory,
#         )
#
#         # extract the upper boundary conditions on the pressure field and
#         # the free coefficients of the scheme
#         self._pt = (
#             kwargs["pt"].to_units("Pa").values.item()
#             if "pt" in kwargs
#             else 0.0
#         )
#         self._a = kwargs.get("a", 0.375)
#         self._b = kwargs.get("b", 0.375)
#         self._c = kwargs.get("c", 0.25)
#
#         # instantiate the component retrieving the diagnostic variables
#         self._diagnostics = IsentropicDiagnostics(
#             grid,
#             backend=backend,
#             backend_opts=backend_opts,
#             dtype=dtype,
#             build_info=build_info,
#             exec_info=exec_info,
#             default_origin=default_origin,
#             rebuild=rebuild,
#             managed_memory=managed_memory,
#         )
#
#         # initialize the pointers to the stencils
#         self._stencil_first_stage_slow = None
#         self._stencil_first_stage_fast = None
#         self._stencil_second_stage_slow = None
#         self._stencil_second_stage_fast = None
#         self._stencil_third_stage_slow = None
#         self._stencil_third_stage_fast = None
#
#     @property
#     def stages(self):
#         return 3
#
#     @property
#     def substep_fractions(self):
#         return 1.0 / 3.0, 0.5, 1.0
#
#     def stage_call(self, stage, timestep, state, tendencies=None):
#         tendencies = {} if tendencies is None else tendencies
#
#         if self._stencil_first_stage_slow is None:
#             # initialize the stencils
#             self._stencils_initialize(tendencies)
#
#         # set stencils' inputs
#         self._stencils_set_inputs(stage, timestep, state, tendencies)
#
#         # step the isentropic density and the water species
#         if stage == 0:
#             self._stencil_first_stage_slow.compute()
#         elif stage == 1:
#             self._stencil_second_stage_slow.compute()
#         else:
#             self._stencil_third_stage_slow.compute()
#
#         # apply the boundary conditions on the stepped isentropic density
#         dt = timestep / 3.0
#         if stage == 0:
#             try:
#                 self._hb.dmn_enforce_field(
#                     self._s1,
#                     "air_isentropic_density",
#                     "kg m^-2 K^-1",
#                     time=state["time"] + dt,
#                 )
#             except AttributeError:
#                 self._hb.enforce_field(
#                     self._s1,
#                     "air_isentropic_density",
#                     "kg m^-2 K^-1",
#                     time=state["time"] + dt,
#                 )
#         elif stage == 1:
#             try:
#                 self._hb.dmn_enforce_field(
#                     self._s2,
#                     "air_isentropic_density",
#                     "kg m^-2 K^-1",
#                     time=state["time"] + dt,
#                 )
#             except AttributeError:
#                 self._hb.enforce_field(
#                     self._s2,
#                     "air_isentropic_density",
#                     "kg m^-2 K^-1",
#                     time=state["time"] + dt,
#                 )
#         else:
#             try:
#                 self._hb.dmn_enforce_field(
#                     self._s_new,
#                     "air_isentropic_density",
#                     "kg m^-2 K^-1",
#                     time=state["time"] + dt,
#                 )
#             except AttributeError:
#                 self._hb.enforce_field(
#                     self._s_new,
#                     "air_isentropic_density",
#                     "kg m^-2 K^-1",
#                     time=state["time"] + dt,
#                 )
#
#         # diagnose the Montgomery potential from the stepped isentropic density,
#         # then step the momenta
#         if stage == 0:
#             self._diagnostics.get_montgomery_potential(
#                 self._s1, self._pt, self._mtg1
#             )
#             self._stencil_first_stage_fast.compute()
#         elif stage == 1:
#             self._diagnostics.get_montgomery_potential(
#                 self._s2, self._pt, self._mtg2
#             )
#             self._stencil_second_stage_fast.compute()
#         else:
#             self._diagnostics.get_montgomery_potential(
#                 self._s_new, self._pt, self._mtg_new
#             )
#             self._stencil_third_stage_fast.compute()
#
#         # collect the outputs
#         out_state = {"time": state["time"] + dt}
#         if stage == 0:
#             out_state["air_isentropic_density"] = self._s1
#             out_state["x_momentum_isentropic"] = self._su1
#             out_state["y_momentum_isentropic"] = self._sv1
#             if self._moist:
#                 out_state["isentropic_density_of_water_vapor"] = self._sqv1
#                 out_state[
#                     "isentropic_density_of_cloud_liquid_water"
#                 ] = self._sqc1
#                 out_state[
#                     "isentropic_density_of_precipitation_water"
#                 ] = self._sqr1
#         elif stage == 1:
#             out_state["air_isentropic_density"] = self._s2
#             out_state["x_momentum_isentropic"] = self._su2
#             out_state["y_momentum_isentropic"] = self._sv2
#             if self._moist:
#                 out_state["isentropic_density_of_water_vapor"] = self._sqv2
#                 out_state[
#                     "isentropic_density_of_cloud_liquid_water"
#                 ] = self._sqc2
#                 out_state[
#                     "isentropic_density_of_precipitation_water"
#                 ] = self._sqr2
#         else:
#             out_state["air_isentropic_density"] = self._s_new
#             out_state["x_momentum_isentropic"] = self._su_new
#             out_state["y_momentum_isentropic"] = self._sv_new
#             if self._moist:
#                 out_state["isentropic_density_of_water_vapor"] = self._sqv_new
#                 out_state[
#                     "isentropic_density_of_cloud_liquid_water"
#                 ] = self._sqc_new
#                 out_state[
#                     "isentropic_density_of_precipitation_water"
#                 ] = self._sqr_new
#
#         return out_state
#
#     def _stencils_allocate(self, tendencies):
#         super()._stencils_allocate(tendencies)
#
#         # allocate the arrays which will store the intermediate values
#         # of the model variables
#         nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
#         dtype = self._dtype
#         self._s1 = np.zeros((nx, ny, nz), dtype=dtype)
#         self._s2 = np.zeros((nx, ny, nz), dtype=dtype)
#         self._u1 = np.zeros((nx + 1, ny, nz), dtype=dtype)
#         self._u2 = np.zeros((nx + 1, ny, nz), dtype=dtype)
#         self._v1 = np.zeros((nx, ny + 1, nz), dtype=dtype)
#         self._v2 = np.zeros((nx, ny + 1, nz), dtype=dtype)
#         self._mtg1 = np.zeros((nx, ny, nz), dtype=dtype)
#         self._mtg2 = self._mtg1
#         self._mtg_new = np.zeros((nx, ny, nz), dtype=dtype)
#         self._su1 = np.zeros((nx, ny, nz), dtype=dtype)
#         self._su2 = np.zeros((nx, ny, nz), dtype=dtype)
#         self._sv1 = np.zeros((nx, ny, nz), dtype=dtype)
#         self._sv2 = np.zeros((nx, ny, nz), dtype=dtype)
#         if self._moist:
#             self._sqv1 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._sqv2 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._sqc1 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._sqc2 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._sqr1 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._sqr2 = np.zeros((nx, ny, nz), dtype=dtype)
#
#         # allocate the arrays which will store the physical tendencies
#         if hasattr(self, "_s_tnd"):
#             self._s_tnd_1 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._s_tnd_2 = np.zeros((nx, ny, nz), dtype=dtype)
#         if hasattr(self, "_su_tnd"):
#             self._su_tnd_1 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._su_tnd_2 = np.zeros((nx, ny, nz), dtype=dtype)
#         if hasattr(self, "_sv_tnd"):
#             self._sv_tnd_1 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._sv_tnd_2 = np.zeros((nx, ny, nz), dtype=dtype)
#         if hasattr(self, "_qv_tnd"):
#             self._qv_tnd_1 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._qv_tnd_2 = np.zeros((nx, ny, nz), dtype=dtype)
#         if hasattr(self, "_qc_tnd"):
#             self._qc_tnd_1 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._qc_tnd_2 = np.zeros((nx, ny, nz), dtype=dtype)
#         if hasattr(self, "_qr_tnd"):
#             self._qr_tnd_1 = np.zeros((nx, ny, nz), dtype=dtype)
#             self._qr_tnd_2 = np.zeros((nx, ny, nz), dtype=dtype)
#
#     def _stencils_initialize(self, tendencies):
#         nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
#         nb = self._hb.nb
#
#         s_tnd_on = "air_isentropic_density" in tendencies
#         su_tnd_on = "x_momentum_isentropic" in tendencies
#         sv_tnd_on = "y_momentum_isentropic" in tendencies
#         qv_tnd_on = mfwv in tendencies
#         qc_tnd_on = mfcw in tendencies
#         qr_tnd_on = mfpw in tendencies
#
#         # allocate inputs and outputs
#         self._stencils_allocate(tendencies)
#
#         # initialize the stencil performing the slow mode of the first stage
#         inputs = {
#             "s0": self._s_now,
#             "u0": self._u_now,
#             "v0": self._v_now,
#             "su0": self._su_now,
#             "sv0": self._sv_now,
#         }
#         if s_tnd_on:
#             inputs["s_tnd_0"] = self._s_tnd
#         outputs = {"s1": self._s1}
#         if self._moist:
#             inputs.update(
#                 {
#                     "sqv0": self._sqv_now,
#                     "sqc0": self._sqc_now,
#                     "sqr0": self._sqr_now,
#                 }
#             )
#             if qv_tnd_on:
#                 inputs["qv_tnd_0"] = self._qv_tnd
#             if qc_tnd_on:
#                 inputs["qc_tnd_0"] = self._qc_tnd
#             if qr_tnd_on:
#                 inputs["qr_tnd_0"] = self._qr_tnd
#             outputs.update(
#                 {"sqv1": self._sqv1, "sqc1": self._sqc1, "sqr1": self._sqr1}
#             )
#         self._stencil_first_stage_slow = gt.NGStencil(
#             definitions_func=self._stencil_first_stage_slow_defs,
#             inputs=inputs,
#             global_inputs={"dt": self._dt, "dx": self._dx, "dy": self._dy},
#             outputs=outputs,
#             domain=gt.domain.Rectangle(
#                 (nb, nb, 0), (nx - nb - 1, ny - nb - 1, nz - 1)
#             ),
#             mode=self._backend,
#         )
#
#         # initialize the stencil performing the fast mode of the first stage
#         inputs = {
#             "s0": self._s_now,
#             "s1": self._s1,
#             "u0": self._u_now,
#             "v0": self._v_now,
#             "mtg0": self._mtg_now,
#             "mtg1": self._mtg1,
#             "su0": self._su_now,
#             "sv0": self._sv_now,
#         }
#         if su_tnd_on:
#             inputs["su_tnd_0"] = self._su_tnd
#         if sv_tnd_on:
#             inputs["sv_tnd_0"] = self._sv_tnd
#         outputs = {"su1": self._su1, "sv1": self._sv1}
#         self._stencil_first_stage_fast = gt.NGStencil(
#             definitions_func=self._stencil_first_stage_fast_defs,
#             inputs=inputs,
#             global_inputs={"dt": self._dt, "dx": self._dx, "dy": self._dy},
#             outputs=outputs,
#             domain=gt.domain.Rectangle(
#                 (nb, nb, 0), (nx - nb - 1, ny - nb - 1, nz - 1)
#             ),
#             mode=self._backend,
#         )
#
#         # initialize the stencil performing the slow mode of the second stage
#         inputs = {
#             "s0": self._s_now,
#             "s1": self._s1,
#             "u0": self._u_now,
#             "u1": self._u1,
#             "v0": self._v_now,
#             "v1": self._v1,
#             "su0": self._su_now,
#             "su1": self._su1,
#             "sv0": self._sv_now,
#             "sv1": self._sv1,
#         }
#         if s_tnd_on:
#             inputs["s_tnd_0"] = self._s_tnd
#             inputs["s_tnd_1"] = self._s_tnd_1
#         outputs = {"s2": self._s2}
#         if self._moist:
#             inputs.update(
#                 {
#                     "sqv0": self._sqv_now,
#                     "sqv1": self._sqv1,
#                     "sqc0": self._sqc_now,
#                     "sqc1": self._sqc1,
#                     "sqr0": self._sqr_now,
#                     "sqr1": self._sqr1,
#                 }
#             )
#             if qv_tnd_on:
#                 inputs["qv_tnd_0"] = self._qv_tnd
#                 inputs["qv_tnd_1"] = self._qv_tnd_1
#             if qc_tnd_on:
#                 inputs["qc_tnd_0"] = self._qc_tnd
#                 inputs["qc_tnd_1"] = self._qc_tnd_1
#             if qr_tnd_on:
#                 inputs["qr_tnd_0"] = self._qr_tnd
#                 inputs["qr_tnd_1"] = self._qr_tnd_1
#             outputs.update(
#                 {"sqv2": self._sqv2, "sqc2": self._sqc2, "sqr2": self._sqr2}
#             )
#         self._stencil_second_stage_slow = gt.NGStencil(
#             definitions_func=self._stencil_second_stage_slow_defs,
#             inputs=inputs,
#             global_inputs={"dt": self._dt, "dx": self._dx, "dy": self._dy},
#             outputs=outputs,
#             domain=gt.domain.Rectangle(
#                 (nb, nb, 0), (nx - nb - 1, ny - nb - 1, nz - 1)
#             ),
#             mode=self._backend,
#         )
#
#         # initialize the stencil performing the fast mode of the second stage
#         inputs = {
#             "s0": self._s_now,
#             "s1": self._s1,
#             "s2": self._s2,
#             "u0": self._u_now,
#             "u1": self._u1,
#             "v0": self._v_now,
#             "v1": self._v1,
#             "mtg0": self._mtg_now,
#             "mtg2": self._mtg2,
#             "su0": self._su_now,
#             "su1": self._su1,
#             "sv0": self._sv_now,
#             "sv1": self._sv1,
#         }
#         if su_tnd_on:
#             inputs["su_tnd_0"] = self._su_tnd
#             inputs["su_tnd_1"] = self._su_tnd_1
#         if sv_tnd_on:
#             inputs["sv_tnd_0"] = self._sv_tnd
#             inputs["sv_tnd_1"] = self._sv_tnd_1
#         outputs = {"su2": self._su2, "sv2": self._sv2}
#         self._stencil_second_stage_fast = gt.NGStencil(
#             definitions_func=self._stencil_second_stage_fast_defs,
#             inputs=inputs,
#             global_inputs={"dt": self._dt, "dx": self._dx, "dy": self._dy},
#             outputs=outputs,
#             domain=gt.domain.Rectangle(
#                 (nb, nb, 0), (nx - nb - 1, ny - nb - 1, nz - 1)
#             ),
#             mode=self._backend,
#         )
#
#         # initialize the stencil performing the slow mode of the third stage
#         inputs = {
#             "s0": self._s_now,
#             "s1": self._s1,
#             "s2": self._s2,
#             "u0": self._u_now,
#             "u1": self._u1,
#             "u2": self._u2,
#             "v0": self._v_now,
#             "v1": self._v1,
#             "v2": self._v2,
#             "su0": self._su_now,
#             "su1": self._su1,
#             "su2": self._su2,
#             "sv0": self._sv_now,
#             "sv1": self._sv1,
#             "sv2": self._sv2,
#         }
#         if s_tnd_on:
#             inputs["s_tnd_0"] = self._s_tnd
#             inputs["s_tnd_1"] = self._s_tnd_1
#             inputs["s_tnd_2"] = self._s_tnd_2
#         outputs = {"s3": self._s_new}
#         if self._moist:
#             inputs.update(
#                 {
#                     "sqv0": self._sqv_now,
#                     "sqv1": self._sqv1,
#                     "sqv2": self._sqv2,
#                     "sqc0": self._sqc_now,
#                     "sqc1": self._sqc1,
#                     "sqc2": self._sqc2,
#                     "sqr0": self._sqr_now,
#                     "sqr1": self._sqr1,
#                     "sqr2": self._sqr2,
#                 }
#             )
#             if qv_tnd_on:
#                 inputs["qv_tnd_0"] = self._qv_tnd
#                 inputs["qv_tnd_1"] = self._qv_tnd_1
#                 inputs["qv_tnd_2"] = self._qv_tnd_2
#             if qc_tnd_on:
#                 inputs["qc_tnd_0"] = self._qc_tnd
#                 inputs["qc_tnd_1"] = self._qc_tnd_1
#                 inputs["qc_tnd_2"] = self._qc_tnd_2
#             if qr_tnd_on:
#                 inputs["qr_tnd_0"] = self._qr_tnd
#                 inputs["qr_tnd_1"] = self._qr_tnd_1
#                 inputs["qr_tnd_2"] = self._qr_tnd_2
#             outputs.update(
#                 {
#                     "sqv3": self._sqv_new,
#                     "sqc3": self._sqc_new,
#                     "sqr3": self._sqr_new,
#                 }
#             )
#         self._stencil_third_stage_slow = gt.NGStencil(
#             definitions_func=self._stencil_third_stage_slow_defs,
#             inputs=inputs,
#             global_inputs={"dt": self._dt, "dx": self._dx, "dy": self._dy},
#             outputs=outputs,
#             domain=gt.domain.Rectangle(
#                 (nb, nb, 0), (nx - nb - 1, ny - nb - 1, nz - 1)
#             ),
#             mode=self._backend,
#         )
#
#         # initialize the stencil performing the fast mode of the third stage
#         inputs = {
#             "s0": self._s_now,
#             "s1": self._s1,
#             "s2": self._s2,
#             "s3": self._s_new,
#             "u0": self._u_now,
#             "u1": self._u1,
#             "u2": self._u2,
#             "v0": self._v_now,
#             "v1": self._v1,
#             "v2": self._v2,
#             "mtg0": self._mtg_now,
#             "mtg2": self._mtg2,
#             "mtg3": self._mtg_new,
#             "su0": self._su_now,
#             "su1": self._su1,
#             "su2": self._su2,
#             "sv0": self._sv_now,
#             "sv1": self._sv1,
#             "sv2": self._sv2,
#         }
#         if su_tnd_on:
#             inputs["su_tnd_0"] = self._su_tnd
#             inputs["su_tnd_1"] = self._su_tnd_1
#             inputs["su_tnd_2"] = self._su_tnd_2
#         if sv_tnd_on:
#             inputs["sv_tnd_0"] = self._sv_tnd
#             inputs["sv_tnd_1"] = self._sv_tnd_1
#             inputs["sv_tnd_2"] = self._sv_tnd_2
#         outputs = {"su3": self._su_new, "sv3": self._sv_new}
#         self._stencil_third_stage_fast = gt.NGStencil(
#             definitions_func=self._stencil_third_stage_fast_defs,
#             inputs=inputs,
#             global_inputs={
#                 "dt": self._dt,
#                 "dx": self._dx,
#                 "dy": self._dy,
#                 "a": self._a,
#                 "b": self._b,
#                 "c": self._c,
#             },
#             outputs=outputs,
#             domain=gt.domain.Rectangle(
#                 (nb, nb, 0), (nx - nb - 1, ny - nb - 1, nz - 1)
#             ),
#             mode=self._backend,
#         )
#
#     def _stencils_set_inputs(self, stage, timestep, state, tendencies):
#         # shortcuts
#         if tendencies is not None:
#             s_tnd_on = (
#                 tendencies.get("air_isentropic_density", None) is not None
#             )
#             su_tnd_on = (
#                 tendencies.get("x_momentum_isentropic", None) is not None
#             )
#             sv_tnd_on = (
#                 tendencies.get("y_momentum_isentropic", None) is not None
#             )
#             qv_tnd_on = tendencies.get(mfwv, None) is not None
#             qc_tnd_on = tendencies.get(mfcw, None) is not None
#             qr_tnd_on = tendencies.get(mfpw, None) is not None
#         else:
#             s_tnd_on = (
#                 su_tnd_on
#             ) = sv_tnd_on = qv_tnd_on = qc_tnd_on = qr_tnd_on = False
#
#         # update the local time step
#         self._dt.value = timestep.total_seconds()
#
#         # update the Numpy arrays which serve as inputs to the GT4Py stencils
#         if stage == 0:
#             self._s_now[...] = state["air_isentropic_density"][...]
#             self._u_now[...] = state["x_velocity_at_u_locations"][...]
#             self._v_now[...] = state["y_velocity_at_v_locations"][...]
#             self._mtg_now[...] = state["montgomery_potential"][...]
#             self._su_now[...] = state["x_momentum_isentropic"][...]
#             self._sv_now[...] = state["y_momentum_isentropic"][...]
#             if self._moist:
#                 self._sqv_now[...] = state[
#                     "isentropic_density_of_water_vapor"
#                 ][...]
#                 self._sqc_now[...] = state[
#                     "isentropic_density_of_cloud_liquid_water"
#                 ][...]
#                 self._sqr_now[...] = state[
#                     "isentropic_density_of_precipitation_water"
#                 ][...]
#             if s_tnd_on:
#                 self._s_tnd[...] = tendencies["air_isentropic_density"][...]
#             if su_tnd_on:
#                 self._su_tnd[...] = tendencies["x_momentum_isentropic"][...]
#             if sv_tnd_on:
#                 self._sv_tnd[...] = tendencies["y_momentum_isentropic"][...]
#             if qv_tnd_on:
#                 self._qv_tnd[...] = tendencies[mfwv][...]
#             if qc_tnd_on:
#                 self._qc_tnd[...] = tendencies[mfcw][...]
#             if qr_tnd_on:
#                 self._qr_tnd[...] = tendencies[mfpw][...]
#         elif stage == 1:
#             self._s1[...] = state["air_isentropic_density"][...]
#             self._u1[...] = state["x_velocity_at_u_locations"][...]
#             self._v1[...] = state["y_velocity_at_v_locations"][...]
#             if "montgomery_potential" in state:
#                 self._mtg1[...] = state["montgomery_potential"][...]
#             self._su1[...] = state["x_momentum_isentropic"][...]
#             self._sv1[...] = state["y_momentum_isentropic"][...]
#             if self._moist:
#                 self._sqv1[...] = state["isentropic_density_of_water_vapor"][
#                     ...
#                 ]
#                 self._sqc1[...] = state[
#                     "isentropic_density_of_cloud_liquid_water"
#                 ][...]
#                 self._sqr1[...] = state[
#                     "isentropic_density_of_precipitation_water"
#                 ][...]
#             if s_tnd_on:
#                 self._s_tnd_1[...] = tendencies["air_isentropic_density"][...]
#             if su_tnd_on:
#                 self._su_tnd_1[...] = tendencies["x_momentum_isentropic"][...]
#             if sv_tnd_on:
#                 self._sv_tnd_1[...] = tendencies["y_momentum_isentropic"][...]
#             if qv_tnd_on:
#                 self._qv_tnd_1[...] = tendencies[mfwv][...]
#             if qc_tnd_on:
#                 self._qc_tnd_1[...] = tendencies[mfcw][...]
#             if qr_tnd_on:
#                 self._qr_tnd_1[...] = tendencies[mfpw][...]
#         else:
#             self._s2[...] = state["air_isentropic_density"][...]
#             self._u2[...] = state["x_velocity_at_u_locations"][...]
#             self._v2[...] = state["y_velocity_at_v_locations"][...]
#             if "montgomery_potential" in state:
#                 self._mtg2[...] = state["montgomery_potential"][...]
#             self._su2[...] = state["x_momentum_isentropic"][...]
#             self._sv2[...] = state["y_momentum_isentropic"][...]
#             if self._moist:
#                 self._sqv2[...] = state["isentropic_density_of_water_vapor"][
#                     ...
#                 ]
#                 self._sqc2[...] = state[
#                     "isentropic_density_of_cloud_liquid_water"
#                 ][...]
#                 self._sqr2[...] = state[
#                     "isentropic_density_of_precipitation_water"
#                 ][...]
#             if s_tnd_on:
#                 self._s_tnd_2[...] = tendencies["air_isentropic_density"][...]
#             if su_tnd_on:
#                 self._su_tnd_2[...] = tendencies["x_momentum_isentropic"][...]
#             if sv_tnd_on:
#                 self._sv_tnd_2[...] = tendencies["y_momentum_isentropic"][...]
#             if qv_tnd_on:
#                 self._qv_tnd_2[...] = tendencies[mfwv][...]
#             if qc_tnd_on:
#                 self._qc_tnd_2[...] = tendencies[mfcw][...]
#             if qr_tnd_on:
#                 self._qr_tnd_2[...] = tendencies[mfpw][...]
#
#     def _stencil_first_stage_slow_defs(
#         self,
#         dt,
#         dx,
#         dy,
#         s0,
#         u0,
#         v0,
#         su0,
#         sv0,
#         sqv0=None,
#         sqc0=None,
#         sqr0=None,
#         s_tnd_0=None,
#         qv_tnd_0=None,
#         qc_tnd_0=None,
#         qr_tnd_0=None,
#     ):
#         i = gt.Index(axis=0)
#         j = gt.Index(axis=1)
#
#         s1 = gt.Equation()
#         sqv1 = gt.Equation() if sqv0 is not None else None
#         sqc1 = gt.Equation() if sqv0 is not None else None
#         sqr1 = gt.Equation() if sqv0 is not None else None
#
#         fluxes = self._hflux(
#             i,
#             j,
#             dt,
#             s0,
#             u0,
#             v0,
#             su0,
#             sv0,
#             sqv=sqv0,
#             sqc=sqc0,
#             sqr=sqr0,
#             s_tnd=s_tnd_0,
#             qv_tnd=qv_tnd_0,
#             qc_tnd=qc_tnd_0,
#             qr_tnd=qr_tnd_0,
#         )
#
#         flux_s_x, flux_s_y = fluxes[0], fluxes[1]
#         if sqv0 is not None:
#             flux_sqv_x, flux_sqv_y = fluxes[6], fluxes[7]
#             flux_sqc_x, flux_sqc_y = fluxes[8], fluxes[9]
#             flux_sqr_x, flux_sqr_y = fluxes[10], fluxes[11]
#
#         s1[i, j] = s0[i, j] - dt / 3.0 * (
#             (flux_s_x[i, j] - flux_s_x[i - 1, j]) / dx
#             + (flux_s_y[i, j] - flux_s_y[i, j - 1]) / dy
#             - (s_tnd_0[i, j] if s_tnd_0 is not None else 0.0)
#         )
#
#         if sqv0 is not None:
#             sqv1[i, j] = sqv0[i, j] - dt / 3.0 * (
#                 (flux_sqv_x[i, j] - flux_sqv_x[i - 1, j]) / dx
#                 + (flux_sqv_y[i, j] - flux_sqv_y[i, j - 1]) / dy
#                 - (s0[i, j] * qv_tnd_0[i, j] if qv_tnd_0 is not None else 0.0)
#             )
#
#             sqc1[i, j] = sqc0[i, j] - dt / 3.0 * (
#                 (flux_sqc_x[i, j] - flux_sqc_x[i - 1, j]) / dx
#                 + (flux_sqc_y[i, j] - flux_sqc_y[i, j - 1]) / dy
#                 - (s0[i, j] * qc_tnd_0[i, j] if qc_tnd_0 is not None else 0.0)
#             )
#
#             sqr1[i, j] = sqr0[i, j] - dt / 3.0 * (
#                 (flux_sqr_x[i, j] - flux_sqr_x[i - 1, j]) / dx
#                 + (flux_sqr_y[i, j] - flux_sqr_y[i, j - 1]) / dy
#                 - (s0[i, j] * qr_tnd_0[i, j] if qr_tnd_0 is not None else 0.0)
#             )
#
#         if sqv0 is None:
#             return s1
#         else:
#             return s1, sqv1, sqc1, sqr1
#
#     def _stencil_first_stage_fast_defs(
#         self,
#         dt,
#         dx,
#         dy,
#         s0,
#         s1,
#         u0,
#         v0,
#         mtg0,
#         mtg1,
#         su0,
#         sv0,
#         su_tnd_0=None,
#         sv_tnd_0=None,
#     ):
#         i = gt.Index(axis=0)
#         j = gt.Index(axis=1)
#
#         sqv = gt.Equation()
#         sqc = gt.Equation()
#         sqr = gt.Equation()
#         su1 = gt.Equation()
#         sv1 = gt.Equation()
#
#         fluxes = self._hflux(
#             i,
#             j,
#             dt,
#             s0,
#             u0,
#             v0,
#             su0,
#             sv0,
#             sqv=sqv,
#             sqc=sqc,
#             sqr=sqr,
#             su_tnd=su_tnd_0,
#             sv_tnd=sv_tnd_0,
#         )
#
#         flux_su_x, flux_su_y = fluxes[2], fluxes[3]
#         flux_sv_x, flux_sv_y = fluxes[4], fluxes[5]
#
#         su1[i, j] = su0[i, j] - dt / 3.0 * (
#             (flux_su_x[i, j] - flux_su_x[i - 1, j]) / dx
#             + (flux_su_y[i, j] - flux_su_y[i, j - 1]) / dy
#             + 0.5 * s0[i, j] * (mtg0[i + 1, j] - mtg0[i - 1, j]) / (2.0 * dx)
#             + 0.5 * s1[i, j] * (mtg1[i + 1, j] - mtg1[i - 1, j]) / (2.0 * dx)
#             - (su_tnd_0[i, j] if su_tnd_0 is not None else 0.0)
#         )
#         sv1[i, j] = sv0[i, j] - dt / 3.0 * (
#             (flux_sv_x[i, j] - flux_sv_x[i - 1, j]) / dx
#             + (flux_sv_y[i, j] - flux_sv_y[i, j - 1]) / dy
#             + 0.5 * s0[i, j] * (mtg0[i, j + 1] - mtg0[i, j - 1]) / (2.0 * dy)
#             + 0.5 * s1[i, j] * (mtg1[i, j + 1] - mtg1[i, j - 1]) / (2.0 * dy)
#             - (sv_tnd_0[i, j] if sv_tnd_0 is not None else 0.0)
#         )
#
#         return su1, sv1
#
#     def _stencil_second_stage_slow_defs(
#         self,
#         dt,
#         dx,
#         dy,
#         s0,
#         s1,
#         u0,
#         u1,
#         v0,
#         v1,
#         su0,
#         su1,
#         sv0,
#         sv1,
#         sqv0=None,
#         sqv1=None,
#         sqc0=None,
#         sqc1=None,
#         sqr0=None,
#         sqr1=None,
#         s_tnd_0=None,
#         s_tnd_1=None,
#         qv_tnd_0=None,
#         qv_tnd_1=None,
#         qc_tnd_0=None,
#         qc_tnd_1=None,
#         qr_tnd_0=None,
#         qr_tnd_1=None,
#     ):
#         i = gt.Index(axis=0)
#         j = gt.Index(axis=1)
#
#         s2 = gt.Equation()
#         sqv2 = gt.Equation() if sqv0 is not None else None
#         sqc2 = gt.Equation() if sqv0 is not None else None
#         sqr2 = gt.Equation() if sqv0 is not None else None
#
#         fluxes0 = self._hflux(
#             i,
#             j,
#             dt,
#             s0,
#             u0,
#             v0,
#             su0,
#             sv0,
#             sqv=sqv0,
#             sqc=sqc0,
#             sqr=sqr0,
#             s_tnd=s_tnd_0,
#             qv_tnd=qv_tnd_0,
#             qc_tnd=qc_tnd_0,
#             qr_tnd=qr_tnd_0,
#         )
#         fluxes1 = self._hflux(
#             i,
#             j,
#             dt,
#             s1,
#             u1,
#             v1,
#             su1,
#             sv1,
#             sqv=sqv1,
#             sqc=sqc1,
#             sqr=sqr1,
#             s_tnd=s_tnd_1,
#             qv_tnd=qv_tnd_1,
#             qc_tnd=qc_tnd_1,
#             qr_tnd=qr_tnd_1,
#         )
#
#         flux_s_x_0, flux_s_y_0 = fluxes0[0], fluxes0[1]
#         flux_s_x_1, flux_s_y_1 = fluxes1[0], fluxes1[1]
#         if sqv0 is not None:
#             flux_sqv_x_0, flux_sqv_y_0 = fluxes0[6], fluxes0[7]
#             flux_sqv_x_1, flux_sqv_y_1 = fluxes1[6], fluxes1[7]
#             flux_sqc_x_0, flux_sqc_y_0 = fluxes0[8], fluxes0[9]
#             flux_sqc_x_1, flux_sqc_y_1 = fluxes1[8], fluxes1[9]
#             flux_sqr_x_0, flux_sqr_y_0 = fluxes0[10], fluxes0[11]
#             flux_sqr_x_1, flux_sqr_y_1 = fluxes1[10], fluxes1[11]
#
#         s2[i, j] = s0[i, j] - dt * (
#             1.0
#             / 6.0
#             * (
#                 (flux_s_x_0[i, j] - flux_s_x_0[i - 1, j]) / dx
#                 + (flux_s_y_0[i, j] - flux_s_y_0[i, j - 1]) / dy
#                 - (s_tnd_0[i, j] if s_tnd_0 is not None else 0.0)
#             )
#             + 0.5
#             * (
#                 (flux_s_x_1[i, j] - flux_s_x_1[i - 1, j]) / dx
#                 + (flux_s_y_1[i, j] - flux_s_y_1[i, j - 1]) / dy
#                 - (s_tnd_1[i, j] if s_tnd_1 is not None else 0.0)
#             )
#         )
#
#         if sqv0 is not None:
#             sqv2[i, j] = sqv0[i, j] - dt * (
#                 1.0
#                 / 6.0
#                 * (
#                     (flux_sqv_x_0[i, j] - flux_sqv_x_0[i - 1, j]) / dx
#                     + (flux_sqv_y_0[i, j] - flux_sqv_y_0[i, j - 1]) / dy
#                     - (
#                         s0[i, j] * qv_tnd_0[i, j]
#                         if qv_tnd_0 is not None
#                         else 0.0
#                     )
#                 )
#                 + 0.5
#                 * (
#                     (flux_sqv_x_1[i, j] - flux_sqv_x_1[i - 1, j]) / dx
#                     + (flux_sqv_y_1[i, j] - flux_sqv_y_1[i, j - 1]) / dy
#                     - (
#                         s1[i, j] * qv_tnd_1[i, j]
#                         if qv_tnd_1 is not None
#                         else 0.0
#                     )
#                 )
#             )
#
#             sqc2[i, j] = sqc0[i, j] - dt * (
#                 1.0
#                 / 6.0
#                 * (
#                     (flux_sqc_x_0[i, j] - flux_sqc_x_0[i - 1, j]) / dx
#                     + (flux_sqc_y_0[i, j] - flux_sqc_y_0[i, j - 1]) / dy
#                     - (
#                         s0[i, j] * qc_tnd_0[i, j]
#                         if qc_tnd_0 is not None
#                         else 0.0
#                     )
#                 )
#                 + 0.5
#                 * (
#                     (flux_sqc_x_1[i, j] - flux_sqc_x_1[i - 1, j]) / dx
#                     + (flux_sqc_y_1[i, j] - flux_sqc_y_1[i, j - 1]) / dy
#                     - (
#                         s1[i, j] * qc_tnd_1[i, j]
#                         if qc_tnd_1 is not None
#                         else 0.0
#                     )
#                 )
#             )
#
#             sqr2[i, j] = sqr0[i, j] - dt * (
#                 1.0
#                 / 6.0
#                 * (
#                     (flux_sqr_x_0[i, j] - flux_sqr_x_0[i - 1, j]) / dx
#                     + (flux_sqr_y_0[i, j] - flux_sqr_y_0[i, j - 1]) / dy
#                     - (
#                         s0[i, j] * qr_tnd_0[i, j]
#                         if qr_tnd_0 is not None
#                         else 0.0
#                     )
#                 )
#                 + 0.5
#                 * (
#                     (flux_sqr_x_1[i, j] - flux_sqr_x_1[i - 1, j]) / dx
#                     + (flux_sqr_y_1[i, j] - flux_sqr_y_1[i, j - 1]) / dy
#                     - (
#                         s1[i, j] * qr_tnd_1[i, j]
#                         if qr_tnd_1 is not None
#                         else 0.0
#                     )
#                 )
#             )
#
#         if sqv0 is None:
#             return s2
#         else:
#             return s2, sqv2, sqc2, sqr2
#
#     def _stencil_second_stage_fast_defs(
#         self,
#         dt,
#         dx,
#         dy,
#         s0,
#         s1,
#         s2,
#         u0,
#         u1,
#         v0,
#         v1,
#         mtg0,
#         mtg2,
#         su0,
#         su1,
#         sv0,
#         sv1,
#         su_tnd_0=None,
#         su_tnd_1=None,
#         sv_tnd_0=None,
#         sv_tnd_1=None,
#     ):
#         i = gt.Index(axis=0)
#         j = gt.Index(axis=1)
#
#         sqv = gt.Equation()
#         sqc = gt.Equation()
#         sqr = gt.Equation()
#         su2 = gt.Equation()
#         sv2 = gt.Equation()
#
#         fluxes0 = self._hflux(
#             i,
#             j,
#             dt,
#             s0,
#             u0,
#             v0,
#             su0,
#             sv0,
#             sqv=sqv,
#             sqc=sqc,
#             sqr=sqr,
#             su_tnd=su_tnd_0,
#             sv_tnd=sv_tnd_0,
#         )
#         fluxes1 = self._hflux(
#             i,
#             j,
#             dt,
#             s1,
#             u1,
#             v1,
#             su1,
#             sv1,
#             sqv=sqv,
#             sqc=sqc,
#             sqr=sqr,
#             su_tnd=su_tnd_1,
#             sv_tnd=sv_tnd_1,
#         )
#
#         flux_su_x_0, flux_su_y_0 = fluxes0[2], fluxes0[3]
#         flux_su_x_1, flux_su_y_1 = fluxes1[2], fluxes1[3]
#         flux_sv_x_0, flux_sv_y_0 = fluxes0[4], fluxes0[5]
#         flux_sv_x_1, flux_sv_y_1 = fluxes1[4], fluxes1[5]
#
#         su2[i, j] = su0[i, j] - dt * (
#             1.0
#             / 6.0
#             * (
#                 (flux_su_x_0[i, j] - flux_su_x_0[i - 1, j]) / dx
#                 + (flux_su_y_0[i, j] - flux_su_y_0[i, j - 1]) / dy
#                 - (su_tnd_0[i, j] if su_tnd_0 is not None else 0.0)
#             )
#             + 0.5
#             * (
#                 (flux_su_x_1[i, j] - flux_su_x_1[i - 1, j]) / dx
#                 + (flux_su_y_1[i, j] - flux_su_y_1[i, j - 1]) / dy
#                 - (su_tnd_1[i, j] if su_tnd_1 is not None else 0.0)
#             )
#             + 1.0
#             / 3.0
#             * s0[i, j]
#             * (mtg0[i + 1, j] - mtg0[i - 1, j])
#             / (2.0 * dx)
#             + 1.0
#             / 3.0
#             * s2[i, j]
#             * (mtg2[i + 1, j] - mtg2[i - 1, j])
#             / (2.0 * dx)
#         )
#         sv2[i, j] = sv0[i, j] - dt * (
#             1.0
#             / 6.0
#             * (
#                 (flux_sv_x_0[i, j] - flux_sv_x_0[i - 1, j]) / dx
#                 + (flux_sv_y_0[i, j] - flux_sv_y_0[i, j - 1]) / dy
#                 - (sv_tnd_0[i, j] if sv_tnd_0 is not None else 0.0)
#             )
#             + 0.5
#             * (
#                 (flux_sv_x_1[i, j] - flux_sv_x_1[i - 1, j]) / dx
#                 + (flux_sv_y_1[i, j] - flux_sv_y_1[i, j - 1]) / dy
#                 - (sv_tnd_1[i, j] if sv_tnd_1 is not None else 0.0)
#             )
#             + 1.0
#             / 3.0
#             * s0[i, j]
#             * (mtg0[i, j + 1] - mtg0[i, j - 1])
#             / (2.0 * dy)
#             + 1.0
#             / 3.0
#             * s2[i, j]
#             * (mtg2[i, j + 1] - mtg2[i, j - 1])
#             / (2.0 * dy)
#         )
#
#         return su2, sv2
#
#     def _stencil_third_stage_slow_defs(
#         self,
#         dt,
#         dx,
#         dy,
#         s0,
#         s1,
#         s2,
#         u0,
#         u1,
#         u2,
#         v0,
#         v1,
#         v2,
#         su0,
#         su1,
#         su2,
#         sv0,
#         sv1,
#         sv2,
#         sqv0=None,
#         sqv1=None,
#         sqv2=None,
#         sqc0=None,
#         sqc1=None,
#         sqc2=None,
#         sqr0=None,
#         sqr1=None,
#         sqr2=None,
#         s_tnd_0=None,
#         s_tnd_1=None,
#         s_tnd_2=None,
#         qv_tnd_0=None,
#         qv_tnd_1=None,
#         qv_tnd_2=None,
#         qc_tnd_0=None,
#         qc_tnd_1=None,
#         qc_tnd_2=None,
#         qr_tnd_0=None,
#         qr_tnd_1=None,
#         qr_tnd_2=None,
#     ):
#         i = gt.Index(axis=0)
#         j = gt.Index(axis=1)
#
#         s3 = gt.Equation()
#         sqv3 = gt.Equation() if sqv0 is not None else None
#         sqc3 = gt.Equation() if sqv0 is not None else None
#         sqr3 = gt.Equation() if sqv0 is not None else None
#
#         fluxes0 = self._hflux(
#             i,
#             j,
#             dt,
#             s0,
#             u0,
#             v0,
#             su0,
#             sv0,
#             sqv=sqv0,
#             sqc=sqc0,
#             sqr=sqr0,
#             s_tnd=s_tnd_0,
#             qv_tnd=qv_tnd_0,
#             qc_tnd=qc_tnd_0,
#             qr_tnd=qr_tnd_0,
#         )
#         fluxes1 = self._hflux(
#             i,
#             j,
#             dt,
#             s1,
#             u1,
#             v1,
#             su1,
#             sv1,
#             sqv=sqv1,
#             sqc=sqc1,
#             sqr=sqr1,
#             s_tnd=s_tnd_1,
#             qv_tnd=qv_tnd_1,
#             qc_tnd=qc_tnd_1,
#             qr_tnd=qr_tnd_1,
#         )
#         fluxes2 = self._hflux(
#             i,
#             j,
#             dt,
#             s2,
#             u2,
#             v2,
#             su2,
#             sv2,
#             sqv=sqv2,
#             sqc=sqc2,
#             sqr=sqr2,
#             s_tnd=s_tnd_2,
#             qv_tnd=qv_tnd_2,
#             qc_tnd=qc_tnd_2,
#             qr_tnd=qr_tnd_2,
#         )
#
#         flux_s_x_0, flux_s_y_0 = fluxes0[0], fluxes0[1]
#         flux_s_x_1, flux_s_y_1 = fluxes1[0], fluxes1[1]
#         flux_s_x_2, flux_s_y_2 = fluxes2[0], fluxes2[1]
#         if sqv0 is not None:
#             flux_sqv_x_0, flux_sqv_y_0 = fluxes0[6], fluxes0[7]
#             flux_sqv_x_1, flux_sqv_y_1 = fluxes1[6], fluxes1[7]
#             flux_sqv_x_2, flux_sqv_y_2 = fluxes2[6], fluxes2[7]
#             flux_sqc_x_0, flux_sqc_y_0 = fluxes0[8], fluxes0[9]
#             flux_sqc_x_1, flux_sqc_y_1 = fluxes1[8], fluxes1[9]
#             flux_sqc_x_2, flux_sqc_y_2 = fluxes2[8], fluxes2[9]
#             flux_sqr_x_0, flux_sqr_y_0 = fluxes0[10], fluxes0[11]
#             flux_sqr_x_1, flux_sqr_y_1 = fluxes1[10], fluxes1[11]
#             flux_sqr_x_2, flux_sqr_y_2 = fluxes2[10], fluxes2[11]
#
#         s3[i, j] = s0[i, j] - dt * (
#             0.5
#             * (
#                 (flux_s_x_0[i, j] - flux_s_x_0[i - 1, j]) / dx
#                 + (flux_s_y_0[i, j] - flux_s_y_0[i, j - 1]) / dy
#                 - (s_tnd_0[i, j] if s_tnd_0 is not None else 0.0)
#             )
#             - 0.5
#             * (
#                 (flux_s_x_1[i, j] - flux_s_x_1[i - 1, j]) / dx
#                 + (flux_s_y_1[i, j] - flux_s_y_1[i, j - 1]) / dy
#                 - (s_tnd_1[i, j] if s_tnd_1 is not None else 0.0)
#             )
#             + (
#                 (flux_s_x_2[i, j] - flux_s_x_2[i - 1, j]) / dx
#                 + (flux_s_y_2[i, j] - flux_s_y_2[i, j - 1]) / dy
#                 - (s_tnd_2[i, j] if s_tnd_2 is not None else 0.0)
#             )
#         )
#
#         if sqv0 is not None:
#             sqv3[i, j] = sqv0[i, j] - dt * (
#                 0.5
#                 * (
#                     (flux_sqv_x_0[i, j] - flux_sqv_x_0[i - 1, j]) / dx
#                     + (flux_sqv_y_0[i, j] - flux_sqv_y_0[i, j - 1]) / dy
#                     - (
#                         s0[i, j] * qv_tnd_0[i, j]
#                         if qv_tnd_0 is not None
#                         else 0.0
#                     )
#                 )
#                 - 0.5
#                 * (
#                     (flux_sqv_x_1[i, j] - flux_sqv_x_1[i - 1, j]) / dx
#                     + (flux_sqv_y_1[i, j] - flux_sqv_y_1[i, j - 1]) / dy
#                     - (
#                         s1[i, j] * qv_tnd_1[i, j]
#                         if qv_tnd_1 is not None
#                         else 0.0
#                     )
#                 )
#                 + (
#                     (flux_sqv_x_2[i, j] - flux_sqv_x_2[i - 1, j]) / dx
#                     + (flux_sqv_y_2[i, j] - flux_sqv_y_2[i, j - 1]) / dy
#                     - (
#                         s2[i, j] * qv_tnd_2[i, j]
#                         if qv_tnd_2 is not None
#                         else 0.0
#                     )
#                 )
#             )
#
#             sqc3[i, j] = sqc0[i, j] - dt * (
#                 0.5
#                 * (
#                     (flux_sqc_x_0[i, j] - flux_sqc_x_0[i - 1, j]) / dx
#                     + (flux_sqc_y_0[i, j] - flux_sqc_y_0[i, j - 1]) / dy
#                     - (
#                         s0[i, j] * qc_tnd_0[i, j]
#                         if qc_tnd_0 is not None
#                         else 0.0
#                     )
#                 )
#                 - 0.5
#                 * (
#                     (flux_sqc_x_1[i, j] - flux_sqc_x_1[i - 1, j]) / dx
#                     + (flux_sqc_y_1[i, j] - flux_sqc_y_1[i, j - 1]) / dy
#                     - (
#                         s1[i, j] * qc_tnd_1[i, j]
#                         if qc_tnd_1 is not None
#                         else 0.0
#                     )
#                 )
#                 + (
#                     (flux_sqc_x_2[i, j] - flux_sqc_x_2[i - 1, j]) / dx
#                     + (flux_sqc_y_2[i, j] - flux_sqc_y_2[i, j - 1]) / dy
#                     - (
#                         s2[i, j] * qc_tnd_2[i, j]
#                         if qc_tnd_2 is not None
#                         else 0.0
#                     )
#                 )
#             )
#
#             sqr3[i, j] = sqr0[i, j] - dt * (
#                 0.5
#                 * (
#                     (flux_sqr_x_0[i, j] - flux_sqr_x_0[i - 1, j]) / dx
#                     + (flux_sqr_y_0[i, j] - flux_sqr_y_0[i, j - 1]) / dy
#                     - (
#                         s0[i, j] * qr_tnd_0[i, j]
#                         if qr_tnd_0 is not None
#                         else 0.0
#                     )
#                 )
#                 - 0.5
#                 * (
#                     (flux_sqr_x_1[i, j] - flux_sqr_x_1[i - 1, j]) / dx
#                     + (flux_sqr_y_1[i, j] - flux_sqr_y_1[i, j - 1]) / dy
#                     - (
#                         s1[i, j] * qr_tnd_1[i, j]
#                         if qr_tnd_1 is not None
#                         else 0.0
#                     )
#                 )
#                 + (
#                     (flux_sqr_x_2[i, j] - flux_sqr_x_2[i - 1, j]) / dx
#                     + (flux_sqr_y_2[i, j] - flux_sqr_y_2[i, j - 1]) / dy
#                     - (
#                         s2[i, j] * qr_tnd_2[i, j]
#                         if qr_tnd_2 is not None
#                         else 0.0
#                     )
#                 )
#             )
#
#         if sqv0 is None:
#             return s3
#         else:
#             return s3, sqv3, sqc3, sqr3
#
#     def _stencil_third_stage_fast_defs(
#         self,
#         dt,
#         dx,
#         dy,
#         a,
#         b,
#         c,
#         s0,
#         s1,
#         s2,
#         s3,
#         u0,
#         u1,
#         u2,
#         v0,
#         v1,
#         v2,
#         mtg0,
#         mtg2,
#         mtg3,
#         su0,
#         su1,
#         su2,
#         sv0,
#         sv1,
#         sv2,
#         su_tnd_0=None,
#         su_tnd_1=None,
#         su_tnd_2=None,
#         sv_tnd_0=None,
#         sv_tnd_1=None,
#         sv_tnd_2=None,
#     ):
#         i = gt.Index(axis=0)
#         j = gt.Index(axis=1)
#
#         sqv = gt.Equation()
#         sqc = gt.Equation()
#         sqr = gt.Equation()
#         su3 = gt.Equation()
#         sv3 = gt.Equation()
#
#         fluxes0 = self._hflux(
#             i,
#             j,
#             dt,
#             s0,
#             u0,
#             v0,
#             su0,
#             sv0,
#             sqv=sqv,
#             sqc=sqc,
#             sqr=sqr,
#             su_tnd=su_tnd_0,
#             sv_tnd=sv_tnd_0,
#         )
#         fluxes1 = self._hflux(
#             i,
#             j,
#             dt,
#             s1,
#             u1,
#             v1,
#             su1,
#             sv1,
#             sqv=sqv,
#             sqc=sqc,
#             sqr=sqr,
#             su_tnd=su_tnd_1,
#             sv_tnd=sv_tnd_1,
#         )
#         fluxes2 = self._hflux(
#             i,
#             j,
#             dt,
#             s2,
#             u2,
#             v2,
#             su2,
#             sv2,
#             sqv=sqv,
#             sqc=sqc,
#             sqr=sqr,
#             su_tnd=su_tnd_2,
#             sv_tnd=sv_tnd_2,
#         )
#
#         flux_su_x_0, flux_su_y_0 = fluxes0[2], fluxes0[3]
#         flux_su_x_1, flux_su_y_1 = fluxes1[2], fluxes1[3]
#         flux_su_x_2, flux_su_y_2 = fluxes2[2], fluxes2[3]
#         flux_sv_x_0, flux_sv_y_0 = fluxes0[4], fluxes0[5]
#         flux_sv_x_1, flux_sv_y_1 = fluxes1[4], fluxes1[5]
#         flux_sv_x_2, flux_sv_y_2 = fluxes2[4], fluxes2[5]
#
#         su3[i, j] = su0[i, j] - dt * (
#             0.5
#             * (
#                 (flux_su_x_0[i, j] - flux_su_x_0[i - 1, j]) / dx
#                 + (flux_su_y_0[i, j] - flux_su_y_0[i, j - 1]) / dy
#                 - (su_tnd_0[i, j] if su_tnd_0 is not None else 0.0)
#             )
#             - 0.5
#             * (
#                 (flux_su_x_1[i, j] - flux_su_x_1[i - 1, j]) / dx
#                 + (flux_su_y_1[i, j] - flux_su_y_1[i, j - 1]) / dy
#                 - (su_tnd_1[i, j] if su_tnd_1 is not None else 0.0)
#             )
#             + (
#                 (flux_su_x_2[i, j] - flux_su_x_2[i - 1, j]) / dx
#                 + (flux_su_y_2[i, j] - flux_su_y_2[i, j - 1]) / dy
#                 - (su_tnd_2[i, j] if su_tnd_2 is not None else 0.0)
#             )
#             + a * s0[i, j] * (mtg0[i + 1, j] - mtg0[i - 1, j]) / (2.0 * dx)
#             + b * s2[i, j] * (mtg2[i + 1, j] - mtg2[i - 1, j]) / (2.0 * dx)
#             + c * s3[i, j] * (mtg3[i + 1, j] - mtg3[i - 1, j]) / (2.0 * dx)
#         )
#         sv3[i, j] = sv0[i, j] - dt * (
#             0.5
#             * (
#                 (flux_sv_x_0[i, j] - flux_sv_x_0[i - 1, j]) / dx
#                 + (flux_sv_y_0[i, j] - flux_sv_y_0[i, j - 1]) / dy
#                 - (sv_tnd_0[i, j] if sv_tnd_0 is not None else 0.0)
#             )
#             - 0.5
#             * (
#                 (flux_sv_x_1[i, j] - flux_sv_x_1[i - 1, j]) / dx
#                 + (flux_sv_y_1[i, j] - flux_sv_y_1[i, j - 1]) / dy
#                 - (sv_tnd_1[i, j] if sv_tnd_1 is not None else 0.0)
#             )
#             + (
#                 (flux_sv_x_2[i, j] - flux_sv_x_2[i - 1, j]) / dx
#                 + (flux_sv_y_2[i, j] - flux_sv_y_2[i, j - 1]) / dy
#                 - (sv_tnd_2[i, j] if sv_tnd_2 is not None else 0.0)
#             )
#             + a * s0[i, j] * (mtg0[i, j + 1] - mtg0[i, j - 1]) / (2.0 * dy)
#             + b * s2[i, j] * (mtg2[i, j + 1] - mtg2[i, j - 1]) / (2.0 * dy)
#             + c * s3[i, j] * (mtg3[i, j + 1] - mtg3[i, j - 1]) / (2.0 * dy)
#         )
#
#         return su3, sv3