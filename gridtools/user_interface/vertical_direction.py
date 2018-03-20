"""
This module contains the constants used by the user to specify the direction which should be followed
to iterate over the vertical axis (e.g., forward, backward).
"""
from enum import Enum


class VerticalDirection(Enum):
	FORWARD = 0
	BACKWARD = 1
	PARALLEL = 2
