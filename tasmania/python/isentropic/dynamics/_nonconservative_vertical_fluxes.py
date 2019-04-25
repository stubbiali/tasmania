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
# -*- coing: utf-8 -*-
"""
This module contains:
	Centered(IsentropicNonconservativeVerticalFlux)
"""
import gridtools as gt
from tasmania.python.isentropic.dynamics.fluxes import \
	IsentropicNonconservativeVerticalFlux


class Centered(IsentropicNonconservativeVerticalFlux):
	"""
	A centered scheme to compute the horizontal
	numerical fluxes for the prognostic model variables.
	The nonconservative form of the governing equations,
	expressed using isentropic coordinates, is used.
	"""
	def __init__(self, grid, moist_on):
		super().__init__(grid, moist_on)
		self.nb = 1
		self.order = 2

	def __call__(
		self, i, j, k, dt, w, s, s_prv, u, u_prv, v, v_prv,
		qv=None, qv_prv=None, qc=None, qc_prv=None, qr=None, qr_prv=None
	):
		raise NotImplementedError()

