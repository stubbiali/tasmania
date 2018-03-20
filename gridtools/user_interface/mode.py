"""
This module contains the constants used by the user to specify the computation mode (e.g. python, C++, etc.).
"""
from enum import Enum


class Mode(Enum):
	PYTHON = 0
	CPP = 1
	ALPHA = 2
	DEBUG = 3
	NUMPY = 4
