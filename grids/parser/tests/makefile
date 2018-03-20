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
all: test1d

test1d: ../parser_1d_cpp.hpp ../parser_1d_cpp.cpp test_parser_1d_cpp.cpp
	@echo "Compiling test for parser_1d module ..."
	@g++ -std=c++11 -I ../ -c -o parser_1d_cpp.o ../parser_1d_cpp.cpp
	@g++ -std=c++11 -I ../ -c -o test_parser_1d_cpp.o test_parser_1d_cpp.cpp
	@g++ -std=c++11 -I ../ -o test_parser_1d_cpp parser_1d_cpp.o test_parser_1d_cpp.o

.PHONY: clean

clean:
	$(RM) *.o test_parser_1d
