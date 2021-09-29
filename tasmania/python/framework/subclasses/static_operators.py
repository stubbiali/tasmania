# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
from sympl._core.static_operators import StaticComponentOperator


class StageInputStaticComponentOperator(StaticComponentOperator):
    name = "stage_input_properties"
    properties_name = "stage_input_properties"

    @classmethod
    def get_allocator(cls, component):
        return None

    @classmethod
    def get_dims(cls, component):
        return super()._get_dims(component, "stage_input_properties")


class SubstepInputStaticComponentOperator(StaticComponentOperator):
    name = "substep_input_properties"
    properties_name = "substep_input_properties"

    @classmethod
    def get_allocator(cls, component):
        return None

    @classmethod
    def get_dims(cls, component):
        return super()._get_dims(component, "substep_input_properties")


class InputTendencyStaticComponentOperator(StaticComponentOperator):
    name = "input_tendency_properties"
    properties_name = "input_tendency_properties"

    @classmethod
    def get_allocator(cls, component):
        return None

    @classmethod
    def get_dims(cls, component):
        return super()._get_dims(component, "input_properties")


class StageTendencyStaticComponentOperator(StaticComponentOperator):
    name = "stage_tendency_properties"
    properties_name = "stage_tendency_properties"

    @classmethod
    def get_allocator(cls, component):
        return None

    @classmethod
    def get_dims(cls, component):
        return super()._get_dims(component, "stage_input_properties")


class SubstepTendencyStaticComponentOperator(StaticComponentOperator):
    name = "substep_tendency_properties"
    properties_name = "substep_tendency_properties"

    @classmethod
    def get_allocator(cls, component):
        return None

    @classmethod
    def get_dims(cls, component):
        return super()._get_dims(component, "substep_input_properties")


class StageOutputStaticComponentOperator(StaticComponentOperator):
    name = "stage_output_properties"
    properties_name = "stage_output_properties"

    @classmethod
    def get_allocator(cls, component):
        return getattr(component, "allocate_stage_output", None)

    @classmethod
    def get_dims(cls, component):
        return super()._get_dims(component, "stage_input_properties")


class SubstepOutputStaticComponentOperator(StaticComponentOperator):
    name = "substep_output_properties"
    properties_name = "substep_output_properties"

    @classmethod
    def get_allocator(cls, component):
        return getattr(component, "allocate_substep_output", None)

    @classmethod
    def get_dims(cls, component):
        return super()._get_dims(component, "substep_input_properties")


class FusedOutputStaticComponentOperator(StaticComponentOperator):
    name = "fused_output_properties"
    properties_name = "fused_output_properties"

    @classmethod
    def get_allocator(cls, component):
        return None

    @classmethod
    def get_dims(cls, component):
        sco1 = StaticComponentOperator.factory("stage_output_properties")
        dims1 = sco1.get_dims(component)
        sco2 = StaticComponentOperator.factory("substep_output_properties")
        dims2 = sco2.get_dims(component)
        dims1.update(dims2)
        return dims1

    @classmethod
    def get_properties(cls, component):
        sco1 = StaticComponentOperator.factory("stage_output_properties")
        properties1 = sco1.get_properties(component)
        sco2 = StaticComponentOperator.factory("substep_output_properties")
        properties2 = sco2.get_properties(component)
        properties1.update(properties2)
        return properties1

    @classmethod
    def get_properties_with_dims(cls, component):
        sco1 = StaticComponentOperator.factory("stage_output_properties")
        properties1 = sco1.get_properties_with_dims(component)
        sco2 = StaticComponentOperator.factory("substep_output_properties")
        properties2 = sco2.get_properties_with_dims(component)
        properties1.update(properties2)
        return properties1
