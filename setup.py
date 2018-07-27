import os
import sys
from setuptools import setup, find_packages, Extension


version = '0.2'


#
# This interface is Python 3.x only
#
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
	description		 = 'Python library for Earth system science',
	long_description = read_file('README.md'),
	version			 = version,
	keywords		 = '',
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
