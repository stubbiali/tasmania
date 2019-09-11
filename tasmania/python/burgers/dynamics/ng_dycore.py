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
	NGBurgersDynamicalCore(NGDynamicalCore)
"""
import gridtools as gt
from tasmania.python.burgers.dynamics.ng_stepper import NGBurgersStepper
from tasmania.python.framework.ng_dycore import NGDynamicalCore
from tasmania.python.utils.storage_utils import (
    get_storage_descriptor,
    make_dataarray_3d,
)

try:
    from tasmania.conf import datatype
except TypeError:
    from numpy import float32 as datatype


class NGBurgersDynamicalCore(NGDynamicalCore):
    """
	The dynamical core for the inviscid 2-D Burgers equations.
	"""

    def __init__(
        self,
        domain,
        intermediate_tendencies=None,
        time_integration_scheme="forward_euler",
        flux_scheme="upwind",
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        halo=None,
        rebuild=None
    ):
        """
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		intermediate_tendencies : `obj`, optional
			An instance of either

				* :class:`sympl.TendencyComponent`,
				* :class:`sympl.TendencyComponentComposite`,
				* :class:`sympl.ImplicitTendencyComponent`,
				* :class:`sympl.ImplicitTendencyComponentComposite`, or
				* :class:`tasmania.ConcurrentCoupling`

			calculating the intermediate physical tendencies.
			Here, *intermediate* refers to the fact that these physical
			packages are called *before* each stage of the dynamical core
			to calculate the physical tendencies.
		time_integration_scheme : `str`, optional
			String specifying the time integration scheme to be used.
			Defaults to 'forward_euler'. See :class:`tasmania.BurgersStepper`
			for all available options.
		flux_scheme : `str`, optional
			String specifying the advective flux scheme to be used.
			Defaults to 'upwind'. See :class:`tasmania.BurgersAdvection`
			for all available options.
		backend : `str`, optional
			TODO
		backend_opts : `dict`, optional
			TODO
		build_info : `dict`, optional
			TODO
		dtype : `numpy.dtype`, optional
			TODO
		exec_info : `dict`, optional
			TODO
		halo : `tuple`, optional
			TODO
		rebuild : `bool`, optional
			TODO
		"""
        grid = domain.numerical_grid
        storage_shape = (grid.nx, grid.ny, 1)
        self._descriptor = get_storage_descriptor(storage_shape, dtype, halo=halo)
        self._backend = backend

        super().__init__(
            domain,
            grid_type="numerical",
            time_units="s",
            intermediate_tendencies=intermediate_tendencies,
            intermediate_diagnostics=None,
            substeps=0,
            fast_tendencies=None,
            fast_diagnostics=None,
            dtype=dtype,
        )

        assert (
            self.grid.nz == 1
        ), "The number grid points along the vertical dimension must be 1."

        self._stepper = NGBurgersStepper.factory(
            time_integration_scheme,
            self.grid.grid_xy,
            self.horizontal_boundary.nb,
            flux_scheme,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            exec_info=exec_info,
            halo=halo,
            rebuild=rebuild,
        )

    @property
    def _input_properties(self):
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-1"},
            "y_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def _substep_input_properties(self):
        return {}

    @property
    def _tendency_properties(self):
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-2"},
            "y_velocity": {"dims": dims, "units": "m s^-2"},
        }

    @property
    def _substep_tendency_properties(self):
        return {}

    @property
    def _output_properties(self):
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-1"},
            "y_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def _substep_output_properties(self):
        return {}

    @property
    def stages(self):
        return self._stepper.stages

    def substep_fractions(self):
        return 1

    def _allocate_output_state(self):
        u_gt = gt.storage.zeros(self._descriptor, backend=self._backend)
        u = make_dataarray_3d(u_gt.data, self.grid, "m s^-1", "x_velocity")
        u.attrs["gt_storage"] = u_gt

        v_gt = gt.storage.zeros(self._descriptor, backend=self._backend)
        v = make_dataarray_3d(v_gt.data, self.grid, "m s^-1", "y_velocity")
        v.attrs["gt_storage"] = v_gt

        return {"x_velocity": u, "y_velocity": v}

    def array_call(self, stage, raw_state, raw_tendencies, timestep):
        out_state = self._stepper(stage, raw_state, raw_tendencies, timestep)

        np_out_state = {"time": out_state["time"]}
        np_out_state.update(
            {key: out_state[key].data for key in out_state if key != "time"}
        )
        self.horizontal_boundary.dmn_enforce_raw(
            np_out_state,
            field_properties={
                "x_velocity": {"units": "m s^-1"},
                "y_velocity": {"units": "m s^-1"},
            },
        )

        return out_state

    def substep_array_call(
        self,
        stage,
        substep,
        raw_state,
        raw_stage_state,
        raw_tmp_state,
        raw_tendencies,
        timestep,
    ):
        raise NotImplementedError()
