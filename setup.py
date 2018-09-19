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
from distutils.core import Extension
import os
import sys
from setuptools import setup


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
	name='tasmania',
	version='0.2.0',
	author='Stefano Ubbiali',
	author_email='subbiali@phys.ethz.ch',
	description='A Python library to ease the composition, configuration, ' 
			    'and execution of Earth system models.',
	long_description=read_file('README.md'),
	long_description_content_type='text/markdown',
	keywords='tasmania',
	url='https://github.com/eth-cscs/tasmania',
	license='',
	package_dir={'': 'tasmania'},
	packages=['grids', 'dynamics', 'physics', 'plot'],
	#package_data={'': ['tests/*', '*.pickle']},
	setup_requires=['setuptools_scm', 'pytest-runner'],
	tests_require=['pytest'],
	install_requires=read_file('requirements.txt').split('\n'),
	ext_package='grids.parser',
	ext_modules=[Extension('parser_1d',
						   sources=['tasmania/grids/parser/parser_1d_cpp.cpp',
									'tasmania/grids/parser/parser_1d.cpp'],
						   include_dirs=['tasmania/grids/parser']),
				 Extension('parser_2d',
						   sources=['tasmania/grids/parser/parser_2d_cpp.cpp',
						   			'tasmania/grids/parser/parser_2d.cpp'],
				 		   include_dirs=['tasmania/grids/parser'])],
	classifiers=(
		'Development Status :: 3 - Alpha',
		'Programming Language :: Python :: 3',
		'Operating System :: OS Independent',
	),
)
