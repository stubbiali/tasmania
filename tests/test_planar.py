## @package gt4ess
#  Test Planar class.

import math
import numpy as np

import config as cfg
from grids.planar import Planar as Grid

#
# Data
#
kwargs = dict(domain_x = [0.,10.],
			  nx = 101,
			  units_x = "km",
			  dims_x = "x",
			  domain_y = [-5.,7.],
			  ny = 241,
			  units_y = "km",
			  dims_y = "y")

#
# Test
#
g = Grid(**kwargs)
