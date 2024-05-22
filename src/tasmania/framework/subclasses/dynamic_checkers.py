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
from sympl._core.dynamic_checkers import (
    InflowComponentChecker,
    OutflowComponentChecker,
)


class StageInputComponentChecker(InflowComponentChecker):
    name = "stage_input_properties"
    properties_name = "stage_input_properties"


class StageTendencyComponentChecker(InflowComponentChecker):
    name = "stage_tendency_properties"
    properties_name = "stage_tendency_properties"


class InputTendencyInflowComponentChecker(InflowComponentChecker):
    name = "input_tendency_properties"
    properties_name = "input_tendency_properties"

    # def check(self, dataarray_dict, input_dataarray_dict=None) -> None:
    #     self.check_missing_fields(dataarray_dict)


class StageOutputOutflowComponentChecker(OutflowComponentChecker):
    name = "stage_output_properties"
    properties_name = "stage_output_properties"
