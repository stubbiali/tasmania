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
from tasmania.python.framework.register import register
from tasmania.python.framework.tendency_stepper import TendencyStepper
from tasmania.python.utils.framework import get_increment


@register(name="implicit")
class Implicit(TendencyStepper):
    """ Interpret the diagnostics as the new values for the prognostic variables. """

    def __init__(
        self,
        *args,
        execution_policy="serial",
        enforce_horizontal_boundary=False,
        backend="numpy",
        backend_options=None,
        storage_options=None,
        **kwargs
    ):
        super().__init__(
            *args,
            execution_policy=execution_policy,
            enforce_horizontal_boundary=enforce_horizontal_boundary,
            backend=backend,
            backend_options=backend_options,
            storage_options=storage_options
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
