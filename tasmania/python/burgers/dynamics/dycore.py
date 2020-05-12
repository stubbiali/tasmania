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
from typing import Optional, TYPE_CHECKING

from tasmania.python.burgers.dynamics.stepper import BurgersStepper
from tasmania.python.framework.dycore import DynamicalCore
from tasmania.python.utils import taz_types
from tasmania.python.utils.storage_utils import get_dataarray_3d, zeros

if TYPE_CHECKING:
    from tasmania.python.domain.domain import Domain


class BurgersDynamicalCore(DynamicalCore):
    """ The dynamical core for the inviscid 2-D Burgers equations. """

    def __init__(
        self,
        domain: "Domain",
        intermediate_tendency_component: Optional[taz_types.tendency_component_t] = None,
        time_integration_scheme: str = "forward_euler",
        flux_scheme: str = "upwind",
        gt_powered: bool = False,
        *,
        backend: str = "numpy",
        backend_opts: Optional[taz_types.options_dict_t] = None,
        build_info: Optional[taz_types.options_dict_t] = None,
        dtype: taz_types.dtype_t = np.float64,
        exec_info: Optional[taz_types.mutable_options_dict_t] = None,
        default_origin: Optional[taz_types.triplet_int_t] = None,
        rebuild: bool = False,
        managed_memory: bool = False
    ) -> None:
        """
        Parameters
        ----------
        domain : tasmania.Domain
            The :class:`~tasmania.Domain` holding the grid underneath.
        intermediate_tendency_component : `obj`, optional
            An instance of either

            * :class:`~sympl.TendencyComponent`,
            * :class:`~sympl.TendencyComponentComposite`,
            * :class:`~sympl.ImplicitTendencyComponent`,
            * :class:`~sympl.ImplicitTendencyComponentComposite`, or
            * :class:`~tasmania.ConcurrentCoupling`

            prescribing physics tendencies and retrieving diagnostic quantities.
            This object is called at the beginning of each stage on the latest
            provisional state.
        time_integration_scheme : `str`, optional
            String specifying the time integration scheme to be used.
            Defaults to "forward_euler". See :class:`~tasmania.BurgersStepper`
            for all available options.
        flux_scheme : `str`, optional
            String specifying the advective flux scheme to be used.
            Defaults to "upwind". See :class:`~tasmania.BurgersAdvection`
            for all available options.
        gt_powered : `bool`, optional
            ``True`` to harness GT4Py, ``False`` for a vanilla NumPy implementation.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        default_origin : `tuple[int]`, optional
            Default origin of the storages.
        rebuild : `bool`, optional
            ``True`` to trigger the stencils compilation at any class instantiation,
            ``False`` to rely on the caching mechanism implemented by GT4Py.
        managed_memory : `bool`, optional
            ``True`` to allocate the storages as managed memory, ``False`` otherwise.
        """
        self._backend = backend
        self._dtype = dtype
        self._default_origin = default_origin
        self._managed_memory = managed_memory

        super().__init__(
            domain,
            grid_type="numerical",
            intermediate_tendency_component=intermediate_tendency_component,
            intermediate_diagnostic_component=None,
            substeps=0,
            fast_tendency_component=None,
            fast_diagnostic_component=None,
            gt_powered=gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            rebuild=rebuild,
        )

        assert (
            self.grid.nz == 1
        ), "The number grid points along the vertical dimension must be 1."

        self._stepper = BurgersStepper.factory(
            time_integration_scheme,
            self.grid.grid_xy,
            self.horizontal_boundary.nb,
            flux_scheme,
            gt_powered=gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            exec_info=exec_info,
            default_origin=default_origin,
            rebuild=rebuild,
            managed_memory=managed_memory,
        )

    @property
    def stage_input_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-1"},
            "y_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def substep_input_properties(self) -> taz_types.properties_dict_t:
        return {}

    @property
    def stage_tendency_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-2"},
            "y_velocity": {"dims": dims, "units": "m s^-2"},
        }

    @property
    def substep_tendency_properties(self) -> taz_types.properties_dict_t:
        return {}

    @property
    def stage_output_properties(self) -> taz_types.properties_dict_t:
        g = self.grid
        dims = (g.grid_xy.x.dims[0], g.grid_xy.y.dims[0], g.z.dims[0])
        return {
            "x_velocity": {"dims": dims, "units": "m s^-1"},
            "y_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def substep_output_properties(self) -> taz_types.properties_dict_t:
        return {}

    @property
    def stages(self) -> int:
        return self._stepper.stages

    def substep_fractions(self) -> int:
        return 1

    def allocate_output_state(self):
        grid = self.grid
        nx, ny = grid.nx, grid.ny
        gt_powered = self._gt_powered
        backend = self._backend
        dtype = self._dtype
        default_origin = self._default_origin
        managed_memory = self._managed_memory

        u = zeros(
            (nx, ny, 1),
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        u_da = get_dataarray_3d(
            u, grid, "m s^-1", name="x_velocity", set_coordinates=False
        )
        v = zeros(
            (nx, ny, 1),
            gt_powered=gt_powered,
            backend=backend,
            dtype=dtype,
            default_origin=default_origin,
            managed_memory=managed_memory,
        )
        v_da = get_dataarray_3d(
            v, grid, "m s^-1", name="y_velocity", set_coordinates=False
        )

        return {"x_velocity": u_da, "y_velocity": v_da}

    def stage_array_call(
        self,
        stage: int,
        raw_state: taz_types.array_dict_t,
        raw_tendencies: taz_types.array_dict_t,
        timestep: taz_types.timedelta_t,
    ) -> taz_types.array_dict_t:
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
        stage: int,
        substep: int,
        raw_state: taz_types.array_dict_t,
        raw_stage_state: taz_types.array_dict_t,
        raw_tmp_state: taz_types.array_dict_t,
        raw_tendencies: taz_types.array_dict_t,
        timestep: taz_types.timedelta_t,
    ):
        raise NotImplementedError()
