import numpy as np
import sympl

from tasmania.namelist import datatype


class GridXY:
	"""
	This class represents a rectangular and regular two-dimensional grid
	embedded in a reference system whose coordinates are, in the order,
	:math:`x` and :math:`y`. No assumption is made on the nature of the
	coordinates. For instance, :math:`x` may be the longitude, in which
	case :math:`x \equiv \lambda`, and :math:`y` may be the latitude,
	in which case :math:`y \equiv \phi`.

	Attributes
	----------
	x : dataarray_like
		:class:`sympl.DataArray` storing the :math:`x`-coordinates of
		the mass points.
	x_at_u_locations : dataarray_like
		:class:`sympl.DataArray` storing the :math:`x`-coordinates of
		the :math:`staggered` points.
	nx : int
		Number of mass points in the :math:`x`-direction.
	dx : float
		The :math:`x`-spacing.
	y : dataarray_like
		:class:`sympl.DataArray` storing the :math:`y`-coordinates of
		the mass points.
	y_at_v_locations : dataarray_like
		:class:`sympl.DataArray` storing the :math:`y`-coordinates of
		the :math:`y`-staggered points.
	ny : int
		Number of mass points in the :math:`y`-direction.
	dy : float
		The :math:`y`-spacing.
	"""
	def __init__(self, domain_x, nx, domain_y, ny, 
				 dims_x='longitude', units_x='degrees_east',
				 dims_y='latitude', units_y='degrees_north',
				 dtype=datatype):
		""" 
		Constructor.

		Parameters
		----------
		domain_x : tuple
			The interval which the domain includes along the :math:`x`-axis.
		nx : int
			Number of mass points in the :math:`x`-direction.
		domain_y : tuple
			The interval which the domain includes along the :math:`y`-axis.
		ny : int
			Number of mass points along :math:`y`.
		dims_x : `str`, optional
			Label for the :math:`x`-coordinate. Defaults to 'longitude'.
		units_x : `str`, optional
			Units for the :math:`x`-coordinate. Defaults to 'degrees_east'.
		dims_y : `str`, optional
			Label for the :math:`y`-coordinate. Defaults to 'latitude'.
		units_y : `str`, optional
			Units for the :math:`y`-coordinate. Defaults to 'degrees_north'.
		dtype : `obj`, optional
			Instance of :class:`numpy.dtype` specifying the data type for
			any :class:`numpy.ndarray` used within this class.
			Defaults to :obj:`~tasmania.namelist.datatype`.

		Note
		----
		Axes labels should use the `CF Conventions <http://cfconventions.org>`_.
		"""
		# x-coordinates of the mass points
		xv = np.array([0.5 * (domain_x[0]+domain_x[1])], dtype=dtype) if nx == 1 \
			 else np.linspace(domain_x[0], domain_x[1], nx, dtype=dtype)
		self.x = sympl.DataArray(xv, coords=[xv], dims=dims_x, name='x',
								 attrs={'units': units_x})

		# Number of mass points along the x-axis
		self.nx = int(nx)

		# x-spacing
		self.dx = 1. if nx == 1 else (domain_x[1]-domain_x[0]) / (nx-1.)

		# x-coordinates of the x-staggered points
		xv_u = np.linspace(domain_x[0] - 0.5*self.dx, domain_x[1] + 0.5*self.dx,
						   nx+1, dtype=dtype)
		self.x_at_u_locations = sympl.DataArray(xv_u, coords=[xv_u], dims=dims_x,
												name='x_at_u_locations',
												attrs={'units': units_x})

		# y-coordinates of the mass points
		yv = np.array([0.5 * (domain_y[0]+domain_y[1])], dtype=dtype) if ny == 1 \
			 else np.linspace(domain_y[0], domain_y[1], ny, dtype=dtype)
		self.y = sympl.DataArray(yv, coords=[yv], dims=dims_y, name='y',
								 attrs={'units': units_y})

		# Number of mass points along the y-axis
		self.ny = int(ny)

		# y-spacing
		self.dy = 1. if ny == 1 else (domain_y[1]-domain_y[0]) / (ny-1.)

		# y-coordinates of the y-staggered points
		yv_v = np.linspace(domain_y[0] - 0.5*self.dy, domain_y[1] + 0.5*self.dy,
						   ny+1, dtype=dtype)
		self.y_at_v_locations = sympl.DataArray(yv_v, coords=[yv_v], dims=dims_y,
												name='y_at_v_locations',
												attrs={'units': units_y})
