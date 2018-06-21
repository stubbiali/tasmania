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
#!/bin/sh

# 2D Example

#python domain_decomposition.py 

#/home/lst/Documents/masterthesis/scotch_6.0.4/src/scotch/gpart 5 subdomains_scotch.src subdomains_scotch.map

#/home/lst/Documents/masterthesis/parmetis/metis-5.1.0/build/Linux-x86_64/programs/gpmetis subdomains_metis.dat 5 -contig

#python visualize_domain_decomposition.py -i "subdomains_metis.dat.part.5" -gp metis -d 16 8 -o metis

#python visualize_domain_decomposition.py -i "subdomains_scotch.map" -gp scotch -d 16 8 -o scotch

#python visualize_domain_decomposition.py -i "pymetis_test.dat.part.5" -gp metis -d 16 8 -o pymetis


# 3D Example

# SETUP:
#    domain = np.array([2048, 1024, 40])
#    slices = np.array([16, 8, 1])
#    stencil = np.array([1, 1, 1, 1, 0, 0])
#    periodic = np.array([1, 0, 0])

python domain_decomposition.py 

/home/lst/Documents/masterthesis/scotch_6.0.4/src/scotch/gpart 5 subdomains_scotch.src subdomains_scotch.map

/home/lst/Documents/masterthesis/parmetis/metis-5.1.0/build/Linux-x86_64/programs/gpmetis subdomains_metis.dat 5 -contig

python visualize_domain_decomposition.py -i "subdomains_metis.dat.part.5" -gp metis -d 16 8 1 -o metis

python visualize_domain_decomposition.py -i "subdomains_scotch.map" -gp scotch -d 16 8 1 -o scotch

python visualize_domain_decomposition.py -i "subdomains_pymetis.dat.part.5" -gp metis -d 16 8 1 -o pymetis
