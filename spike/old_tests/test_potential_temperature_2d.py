## @package gt4ess
#  Test PotentialTemperature2d class.

import numpy as np

from grids.potential_temperature import PotentialTemperature2d as Grid

# Set domain and number of grid points
domain_x = (0., 100.)
n_x = 1001
domain_theta = (5., 10.)
n_theta = 51

# Specify terrain surface profile
h_s = '500 * exp((- x*x / (10.*10.)))'

# Instantiate grid
grid = Grid(domain_x, n_x, domain_theta, n_theta, h_s)
