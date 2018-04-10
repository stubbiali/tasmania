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
import os
import pickle

#
# Mandatory settings
#
filename1 = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_clipping_maccormack.pickle')
filename2 = os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_kessler_wrf_clipping_sedimentation_maccormack.pickle')
field = 'mass_fraction_of_precipitation_water_in_air'

#
# Load
#
with open(filename1, 'rb') as data:
	state_save1 = pickle.load(data)
	#field1 = state_save[field]
with open(filename2, 'rb') as data:
	state_save2 = pickle.load(data)
	#field2 = state_save[field]

print('here we are')

