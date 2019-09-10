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
    BurgersDynamicalCore(DynamicalCore)
"""
from tasmania.python.burgers.dynamics.stepper import BurgersStepper
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.utils.storage_utils import get_dataarray_3d, zeros

try:
    from tasmania.conf import datatype
except TypeError:
    from numpy import float32 as datatype


class BurgersDynamicalCore(DynamicalCore):
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
        self._backend = backend
        self._dtype = dtype
        self._halo = halo

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

        self._stepper = BurgersStepper.factory(
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
        grid = self.grid
        nx, ny = grid.nx, grid.ny
        backend = self._backend
        dtype = self._dtype
        halo = self._halo

        u = zeros((nx, ny, 1), backend, dtype, halo=halo)
        u_da = get_dataarray_3d(u, grid, "m s^-1", name="x_velocity")
        v = zeros((nx, ny, 1), backend, dtype, halo=halo)
        v_da = get_dataarray_3d(v, grid, "m s^-1", name="y_velocity")

        return {"x_velocity": u_da, "y_velocity": v_da}

    def array_call(self, stage, raw_state, raw_tendencies, timestep):
        out_state = self._stepper(stage, raw_state, raw_tendencies, timestep)

        self.horizontal_boundary.dmn_enforce_raw(
            out_state,
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
