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
from tasmania.python.isentropic.dynamics.vertical_fluxes import (
    IsentropicVerticalFlux,
)
from tasmania.python.utils.framework_utils import register


@register(name="maccormack")
class MacCormack(IsentropicVerticalFlux):
    @staticmethod
    def __call__(
        dt,
        dz,
        w,
        s,
        s_prv,
        su,
        su_prv,
        sv,
        sv_prv,
        sqv=None,
        sqv_prv=None,
        sqc=None,
        sqc_prv=None,
        sqr=None,
        sqr_prv=None,
    ):
        raise NotImplementedError()
