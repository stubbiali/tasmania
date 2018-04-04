import math
import matplotlib.pyplot as plt
import numpy as np

from tasmania.grids.axis import Axis
from tasmania.namelist import datatype

class GridXY:
	"""
	Rectangular and regular two-dimensional grid embedded in a reference system whose coordinates are, 
	in the order, :math:`x` and :math:`y`. No assumption is made on the nature of the coordinates. For 
	instance, :math:`x` may be the longitude, in which case :math:`x \equiv \lambda`, and :math:`y` may 
	be the latitude, in which case :math:`y \equiv \phi`.

	Attributes
	----------
	x : obj
		:class:`~grids.axis.Axis` representing the :math:`x` main levels.
	x_half_levels : obj
		:class:`~grids.axis.Axis` representing the :math:`x` half levels.
	nx : int
		Number of grid points along :math:`x`.
	dx : float
		The :math:`x`-spacing.
	y : obj
		:class:`~grids.axis.Axis` representing the :math:`y` main levels.
	y_half_levels : obj
		:class:`~grids.axis.Axis` representing the :math:`y` half levels.
	ny : int
		Number of grid points along :math:`y`.
	dy : float
		The :math:`y`-spacing.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, 
				 units_x = 'degrees_east', dims_x = 'longitude', units_y = 'degrees_north', dims_y = 'latitude'):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			Tuple in the form :math:`(x_{start}, ~ x_{stop})`.
		nx : int
			Number of grid points along :math:`x`.
		domain_y : tuple
			Tuple in the form :math:`(y_{start}, ~ y_{stop})`.
		ny : int
			Number of grid points along :math:`y`.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate.
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate.
		units_y : `str`, optional
			Units for the :math:`y`-coordinate. 
		dims_y : `str`, optional
			Label for the :math:`y`-coordinate.

		Note
		----
		Axes labels should use the `CF Conventions <http://cfconventions.org>`_.
		"""
		xv = np.array([.5 * (domain_x[0] + domain_x[1])], dtype = datatype) if nx == 1 \
			 else np.linspace(domain_x[0], domain_x[1], nx, dtype = datatype)
		self.x = Axis(xv, dims_x, attrs = {'units': units_x}) 
		self.nx = int(nx)
		self.dx = 1. if nx == 1 else (domain_x[1] - domain_x[0]) / (nx - 1.)
		self.x_half_levels = Axis(
			np.linspace(domain_x[0] - 0.5 * self.dx, domain_x[1] + 0.5 * self.dx, nx + 1, dtype = datatype),
			dims_x, attrs = {'units': units_x})

		yv = np.array([.5 * (domain_y[0] + domain_y[1])], dtype = datatype) if ny == 1 \
			 else np.linspace(domain_y[0], domain_y[1], ny, dtype = datatype)
		self.y = Axis(yv, dims_y, attrs = {'units': units_y})
		self.ny = int(ny)
		self.dy = 1. if ny == 1 else (domain_y[1] - domain_y[0]) / (ny - 1.)
		self.y_half_levels = Axis(
			np.linspace(domain_y[0] - 0.5 * self.dy, domain_y[1] + 0.5 * self.dy, ny + 1, dtype = datatype),
			dims_y, attrs = {'units': units_y})
