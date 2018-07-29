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
import os
import sys
from setuptools import setup, Extension


version = '0.2.0'


if sys.version_info.major < 3:
	print('Python 3.x is required.')
	sys.exit(1)


def read_file(fname):
	"""
	Read file into string.

	Parameters
	----------
	fname : str
		Full path to the file.

	Return
	------
	str :
		File content as a string.
	"""
	return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
	name			 = 'tasmania',
	description		 = 'A Python library for building Earth system models.',
	long_description = read_file('README.md'),
	version			 = version,
	keywords		 = 'tasmania',
	author			 = 'Stefano Ubbiali',
	author_email	 = 'subbiali@phys.ethz.ch',
	url				 = 'https://github.com/eth-cscs/tasmania',
	license			 = '',
	package_dir		 = {'': 'tasmania'},
	packages		 = ['grids', 'dynamics', 'physics', 'plot'],
	#package_data	 = {'': ['tests/*', '*.pickle']},
	setup_requires	 = ['setuptools_scm', 'pytest-runner'],
	tests_require	 = ['pytest'],
	install_requires = read_file('requirements.txt').split('\n'),
	ext_package		 = 'grids.parser',
	ext_modules		 = [Extension('parser_1d', ['tasmania/grids/parser/parser_1d_cpp.cpp'],
								  include_dirs=['tasmania/grids/parser']),
						Extension('parser_2d', ['tasmania/grids/parser/parser_2d_cpp.cpp'],
								  include_dirs=['tasmania/grids/parser'])]
)
