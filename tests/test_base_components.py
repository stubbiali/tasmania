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
from hypothesis import \
	assume, given, HealthCheck, settings, strategies as hyp_st, reproduce_failure
import pytest

import os
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import utils

from tasmania.python.framework.base_components import \
	DiagnosticComponent, ImplicitTendencyComponent, TendencyComponent


class FakeDiagnosticComponent(DiagnosticComponent):
	def __init__(self, domain, grid_type, **kwargs):
		super().__init__(domain, grid_type)

	@property
	def input_properties(self):
		return {}

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		return {}


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_diagnostic_component(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(utils.st_domain(), label="domain")

	# ========================================
	# test bed
	# ========================================
	obj = FakeDiagnosticComponent(domain, 'physical')
	assert isinstance(obj, DiagnosticComponent)

	obj = FakeDiagnosticComponent(domain, 'numerical')
	assert isinstance(obj, DiagnosticComponent)


class FakeImplicitTendencyComponent(ImplicitTendencyComponent):
	def __init__(self, domain, grid_type, **kwargs):
		super().__init__(domain, grid_type, **kwargs)

	@property
	def input_properties(self):
		return {}

	@property
	def tendency_properties(self):
		return {}

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state, timestep):
		return {}, {}


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_implicit_tendency_component(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(utils.st_domain(), label="domain")

	# ========================================
	# test bed
	# ========================================
	obj = FakeImplicitTendencyComponent(domain, 'physical')
	assert isinstance(obj, ImplicitTendencyComponent)

	obj = FakeImplicitTendencyComponent(domain, 'numerical')
	assert isinstance(obj, ImplicitTendencyComponent)


class FakeTendencyComponent(TendencyComponent):
	def __init__(self, domain, grid_type, **kwargs):
		super().__init__(domain, grid_type, **kwargs)

	@property
	def input_properties(self):
		return {}

	@property
	def tendency_properties(self):
		return {}

	@property
	def diagnostic_properties(self):
		return {}

	def array_call(self, state):
		return {}, {}


@settings(
	suppress_health_check=(HealthCheck.too_slow, HealthCheck.data_too_large),
	deadline=None
)
@given(hyp_st.data())
def test_tendency_component(data):
	# ========================================
	# random data generation
	# ========================================
	domain = data.draw(utils.st_domain(), label="domain")

	# ========================================
	# test bed
	# ========================================
	obj = FakeTendencyComponent(domain, 'physical')
	assert isinstance(obj, TendencyComponent)

	obj = FakeTendencyComponent(domain, 'numerical')
	assert isinstance(obj, TendencyComponent)


if __name__ == '__main__':
	pytest.main([__file__])
