# -*- Makefile -*-
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
all: parser_1d parser_2d

parser_1d: parser_1d_cpp.hpp parser_1d_cpp.cpp parser_2d.pyx setup_2d.py
	python setup_1d.py build_ext --inplace	

parser_2d: parser_2d_cpp.hpp parser_2d_cpp.cpp parser_2d.pyx setup_2d.py
	python setup_2d.py build_ext --inplace	

.PHONY: clean

clean:
	$(RM) -r *.so build/
