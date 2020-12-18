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
        intermediate_tendency_component: Optional[
            taz_types.tendency_component_t
        ] = None,
        time_integration_scheme: str = "forward_euler",
        flux_scheme: str = "upwind",
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None
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
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        super().__init__(
            domain,
            intermediate_tendency_component=intermediate_tendency_component,
            intermediate_diagnostic_component=None,
            substeps=0,
            fast_tendency_component=None,
            fast_diagnostic_component=None,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options,
        )

        assert (
            self.grid.nz == 1
        ), "The number grid points along the vertical dimension must be 1."

        self._stepper = BurgersStepper.factory(
            time_integration_scheme,
            self.grid.grid_xy,
            self.horizontal_boundary.nb,
            flux_scheme,
            backend=self.backend,
            backend_options=self.backend_options,
            storage_options=self.storage_options,
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

        u = self.zeros(shape=(grid.nx, grid.ny, 1))
        u_da = get_dataarray_3d(
            u, grid, "m s^-1", name="x_velocity", set_coordinates=False
        )
        v = self.zeros(shape=(grid.nx, grid.ny, 1))
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
