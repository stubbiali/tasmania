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
