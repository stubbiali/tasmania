#!/bin/bash
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
rm -rf baseline_images
pytest --mpl-generate-path=baseline_images/test_contour         plot/test_contour.py
pytest --mpl-generate-path=baseline_images/test_contourf 	      plot/test_contourf.py
pytest --mpl-generate-path=baseline_images/test_hovmoller 	    plot/test_hovmoller.py
pytest --mpl-generate-path=baseline_images/test_patches			    plot/test_patches.py
pytest --mpl-generate-path=baseline_images/test_plot				    plot/test_plot.py
pytest --mpl-generate-path=baseline_images/test_plot_composite	plot/test_plot_composite.py
pytest --mpl-generate-path=baseline_images/test_profile 			  plot/test_profile.py
pytest --mpl-generate-path=baseline_images/test_quiver 			    plot/test_quiver.py
pytest --mpl-generate-path=baseline_images/test_timeseries 		  plot/test_timeseries.py
