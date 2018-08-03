import os
import sys
from setuptools import setup, Extension
from tasmania import __version__


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
	version			 = __version__,
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
