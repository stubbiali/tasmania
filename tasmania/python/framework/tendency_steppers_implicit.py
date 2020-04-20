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

from tasmania.python.framework.tendency_stepper import TendencyStepper, registry
from tasmania.python.utils.framework_utils import get_increment


@registry(scheme_name="implicit")
class Implicit(TendencyStepper):
    """ Interpret the diagnostics as the new values for the prognostic variables. """

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        gt_powered=False,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=np.float64,
        rebuild=False,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary,
            gt_powered=gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            rebuild=rebuild
        )

        # consistency check
        assert len(self.prognostic.tendency_properties) == 0

        # ovewrite property dictionaries
        self.diagnostic_properties = {}
        self.output_properties = self.prognostic.diagnostic_properties.copy()

    def _call(self, state, timestep):
        # calculate the diagnostics
        _, diagnostics = get_increment(state, timestep, self.prognostic)

        return {}, diagnostics
