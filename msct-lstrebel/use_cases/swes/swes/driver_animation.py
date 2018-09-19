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
"""
Driver to create a Matplotlib animation starting
from Earth system science snapshots.

Usage:

	python driver_movie.py -i <infile> -a <argument> [-p <projection> -f <format> -o <outfile>]

where

    <infile> is the input file,
    <argument> specifies what should be plotted,
    <projection> specifies the projection to use,
    <format> specifies the format for the movie,
    <outfile> is the output file name, without the extension.

All the supported options for <argument>, <projection> and <format>
are listed in animation.py.
"""
import sys
import getopt
from animation import ess_movie_maker

#
# Read arguments from command line
#
try:
	opts, args = getopt.getopt(sys.argv[1:], "hi:a:p:f:o:")
except getopt.GetoptError:
	print('Usage: driver_movie.py -i <infile> -a <argument> -p <projection> -f <format> -o <outfile>')
	sys.exit(2)

infile = field_to_plot = projection = movie_format = outfile = None
do_help = False
for opt, arg in opts:
	if opt == '-i':
		infile = str(arg)
	elif opt == '-a':
		field_to_plot = str(arg)
	elif opt == '-p':
		projection = str(arg)
	elif opt == '-f':	
		movie_format = str(arg)
	elif opt == '-o':
		outfile = str(arg)
	elif opt == '-h':
		do_help = True

if do_help or len(sys.argv) < 2:
	print(__doc__)
	sys.exit()

assert (infile is not None and field_to_plot is not None), \
	   "Missing mandatory command line argument"

#
# Make movie
#
kwarg = {"filename": infile, "field_to_plot": field_to_plot}
if projection is not None:
	kwarg["projection"] = projection
if movie_format is not None:
	kwarg["movie_format"] = movie_format
if outfile is not None:
	kwarg["movie_name"] = outfile
ess_movie_maker(**kwarg)

