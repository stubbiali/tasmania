""" 
Test Topography1d class. 
"""
import numpy as np

from grids.axis import Axis
from grids.topography import Topography1d

x = Axis(np.linspace(0, 1e5, 1e5+1), dims = 'x')
hs = Topography1d(x, 'gaussian')

print('Test passed!')
