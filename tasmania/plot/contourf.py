"""
This module contains:
	Contourf(Drawer)
	make_contourf_xy
	make_contourf_xz
	make_contourf_xh
	make_contourf_yz
	make_contourf_yh
"""
import numpy as np

from tasmania.plot.drawer import Drawer
from tasmania.plot.plot_utils import make_contour, make_contourf, make_lineplot
from tasmania.plot.retrievers import DataRetriever
from tasmania.plot.utils import to_units


class Contourf(Drawer):
	"""
	Drawer which generates a contourf plot of a state quantity
	at a cross-section parallel to one coordinate plane.
	"""
	def __init__(self, grid, field_name, field_units,
				 x=None, y=None, z=None,
				 xaxis_name=None, xaxis_units=None, xaxis_y=None, xaxis_z=None,
				 yaxis_name=None, yaxis_units=None, yaxis_x=None, yaxis_z=None,
				 zaxis_name=None, zaxis_units=None, zaxis_x=None, zaxis_y=None,
				 topography_units=None, topography_x=None, topography_y=None,
				 properties=None):
		"""
		Parameters
		----------
		grid : grid
			Instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
			or one of its derived classes representing the underlying grid.
		field_name : str
			The state quantity to visualize.
		field_units : str
			The units for the quantity to visualize.
		x : `int`, optional
			Index along the first dimension of the field array identifying
			the cross-section to visualize. To be specified only if both :obj:`y`
			and :obj:`z` are not given.
		y : `int`, optional
			Index along the second dimension of the field array identifying
			the cross-section to visualize. To be specified only if both :obj:`x`
			and :obj:`z` are not given.
		z : `int`, optional
			Index along the third dimension of the field array identifying
			the cross-section to visualize. To be specified only if both :obj:`x`
			and :obj:`y` are not given.
		xaxis_name : `str`, optional
			If either :obj:`y` or :obj:`z` is given, the name of the computational
			axis to place on the plot x-axis. Options are:

				* 'x' (default).

		xaxis_units : `str`, optional
			If either :obj:`y` or :obj:`z` is given, units for the :obj:`xaxis_name`
			computational axis. If not specified, the native units of the
			computational axis are used.
		xaxis_y : `int`, optional
			Index along the second dimension of the :obj:`xaxis_name` computational
			axis array identifying the cross-section to visualize. Defaults to :obj:`y`.
			Only effective if :obj:`xaxis_name` is not 'x' and :obj:`y` is given.
		xaxis_z : `int`, optional
			Index along the third dimension of the :obj:`xaxis_name` computational
			axis array identifying the cross-section to visualize. Defaults to :obj:`z`.
			Only effective if :obj:`xaxis_name` is not 'x' and :obj:`z` is given.
		yaxis_name : `str`, optional
			The name of the computational axis to place either on the plot x-axis
			if :obj:`x` is given, or on the plot y-axis if :obj:`z` is given.
			Options are:

				* 'y' (default).

		yaxis_units : `str`, optional
			If either :obj:`x` or :obj:`z` is given, units for the :obj:`yaxis_name`
			computational axis. If not specified, the native units of the
			computational axis are used.
		yaxis_x : `int`, optional
			Index along the first dimension of the :obj:`yaxis_name` computational
			axis array identifying the cross-section to visualize. Defaults to :obj:`x`.
			Only effective if :obj:`yaxis_name` is not 'y' and :obj:`x` is given.
		yaxis_z : `int`, optional
			Index along the third dimension of the :obj:`yaxis_name` computational
			axis array identifying the cross-section to visualize. Defaults to :obj:`z`.
			Only effective if :obj:`yaxis_name` is not 'y' and :obj:`z` is given.
		zaxis_name : `str`, optional
			If either :obj:`x` or :obj:`y` is given, the name of the computational
			axis to place either on the plot y-axis. Options are:

				* 'z' (default);
				* 'height';
				* 'height_on_interface_levels';
				* 'air_pressure';
				* 'air_pressure_on_interface_levels'.

		zaxis_units : `str`, optional
			If either :obj:`x` or :obj:`y` is given, units for the :obj:`zaxis_name`
			computational axis. If not specified, the native units of the
			computational axis are used.
		zaxis_x : `int`, optional
			Index along the first dimension of the :obj:`zaxis_name` computational
			axis array identifying the cross-section to visualize. Defaults to :obj:`x`.
			Only effective if :obj:`zaxis_name` is not 'z' and :obj:`x` is given.
		zaxis_y : `int`, optional
			Index along the second dimension of the :obj:`zaxis_name` computational
			axis array identifying the cross-section to visualize. Defaults to :obj:`y`.
			Only effective if :obj:`zaxis_name` is not 'z' and :obj:`y` is given.
		topography_units : `str`, optional
			Units for the topography. If not specified, the native units for the
			topography are used. Only effective if :obj:`z` is given.
		topography_x : `int`, optional
			Index along the first dimension of the topography array identifying
			the cross-section to visualize. Defaults to :obj:`x`.
			Only effective if :obj:`zaxis_name` is either 'height' or
			'height_on_interface_levels', and :obj:`x` is given.
		topography_y : `int`, optional
			Index along the second dimension of the topography array identifying
			the cross-section to visualize. Defaults to :obj:`y`.
			Only effective if :obj:`zaxis_name` is either 'height' or
			'height_on_interface_levels', and :obj:`y` is given.
		properties : `dict`, optional
			Dictionary whose keys are strings denoting plot-specific
			properties, and whose values specify values for those properties.
			:func:`tasmania.plot.utils.make_contour` and
			:func:`tasmania.plot.utils.make_lineplot`.
			The latter two utilities are leveraged to draw the topography.
		"""
		super().__init__(properties)

		flag_x = 0 if x is None else 1
		flag_y = 0 if y is None else 1
		flag_z = 0 if z is None else 1
		if flag_x + flag_y + flag_z != 1:
			raise ValueError('A plane is uniquely identified by one index, but here '
				'x is{}given, y is{}given and z is{}given.'.format(
					' ' if flag_x else ' not ', ' ' if flag_y else ' not ',
					' ' if flag_z else ' not ',
				)
			)

		slice_x = slice(x, x+1 if x != -1 else None, None) if flag_x else None
		slice_y = slice(y, y+1 if y != -1 else None, None) if flag_y else None
		slice_z = slice(z, z+1 if z != -1 else None, None) if flag_z else None

		retriever = DataRetriever(grid, field_name, field_units,
								  slice_x, slice_y, slice_z)

		if flag_z:
			topo_retriever = DataRetriever(grid, 'topography', topography_units)
			self._slave = lambda state, fig, ax: make_contourf_xy(
				grid, xaxis_units, yaxis_units, topo_retriever, retriever,
				state, fig, ax, **self.properties)
		else:
			if zaxis_name != 'z':
				zax = zaxis_x if zaxis_x is not None else x
				zay = zaxis_y if zaxis_y is not None else y
				zaslice_x = None if zax is None else \
							slice(zax, zax+1 if zax != -1 else None, None)
				zaslice_y = None if zay is None else \
							slice(zay, zay+1 if zay != -1 else None, None)
				zaxis_retriever = DataRetriever(grid, zaxis_name, zaxis_units,
											    zaslice_x, zaslice_y)

				if zaxis_name in ['height', 'height_on_interface_levels']:
					tx = topography_x if topography_x is not None else x
					ty = topography_y if topography_y is not None else y
					tslice_x = slice(tx, tx+1 if tx != -1 else None, None) if tx is not None \
							   else None
					tslice_y = slice(ty, ty+1 if ty != -1 else None, None) if ty is not None \
							   else None
					topo_retriever = DataRetriever(grid, 'topography', zaxis_units,
												   tslice_x, tslice_y)
				else:
					topo_retriever = None

				if flag_x:
					self._slave = lambda state, fig, ax: make_contourf_yh(
						grid, yaxis_units, zaxis_retriever, topo_retriever, retriever,
						state, fig, ax, **self.properties
					)
				else:
					self._slave = lambda state, fig, ax: make_contourf_xh(
						grid, xaxis_units, zaxis_retriever, topo_retriever, retriever,
						state, fig, ax, **self.properties
					)
			else:
				if flag_x:
					self._slave = lambda state, fig, ax: make_contourf_yz(
						grid, yaxis_units, zaxis_units, retriever, state, fig, ax, **self.properties
					)
				else:
					self._slave = lambda state, fig, ax: make_contourf_xz(
						grid, xaxis_units, zaxis_units, retriever, state, fig, ax, **self.properties
					)

	def __call__(self, state, fig, ax):
		"""
		Call operator generating the contourf plot.
		"""
		self._slave(state, fig, ax)


def make_contourf_xy(grid, xaxis_units, yaxis_units, topo_retriever, field_retriever,
					 state, fig, ax, **kwargs):
	field = np.squeeze(field_retriever(state))

	xv = to_units(grid.x, xaxis_units).values if field.shape[0] == grid.nx \
		 else to_units(grid.x_at_u_locations, xaxis_units).values
	yv = to_units(grid.y, yaxis_units).values if field.shape[1] == grid.ny \
		 else to_units(grid.y_at_v_locations, yaxis_units).values
	x  = np.repeat(xv[:, np.newaxis], yv.shape[0], axis=1)
	y  = np.repeat(yv[np.newaxis, :], xv.shape[0], axis=0)

	make_contourf(x, y, field, fig, ax, **kwargs)

	topo = topo_retriever(state)
	topo = 0.5 * (topo[:-1, :] + topo[1:, :]) if topo.shape[0] > x.shape[0] else topo
	topo = 0.5 * (topo[:, :-1] + topo[:, 1:]) if topo.shape[1] > y.shape[1] else topo

	make_contour(x, y, topo, ax, colors='black', alpha=0.5)


def make_contourf_xz(grid, xaxis_units, zaxis_units, field_retriever,
					 state, fig, ax, **kwargs):
	field = np.squeeze(field_retriever(state))

	xv = to_units(grid.x, xaxis_units).values if field.shape[0] == grid.nx \
		 else to_units(grid.x_at_u_locations, xaxis_units).values
	zv = to_units(grid.z, zaxis_units).values if field.shape[1] == grid.nz \
		 else to_units(grid.z_on_interface_levels, zaxis_units).values
	x  = np.repeat(xv[:, np.newaxis], zv.shape[0], axis=1)
	z  = np.repeat(zv[np.newaxis, :], xv.shape[0], axis=0)

	make_contourf(x, z, field, fig, ax, **kwargs)


def make_contourf_xh(grid, xaxis_units, zaxis_retriever, topo_retriever,
					 field_retriever, state, fig, ax, **kwargs):
	field = np.squeeze(field_retriever(state))

	zv    = np.squeeze(zaxis_retriever(state))
	field = 0.5 * (field[:-1, :] + field[1:, :]) if field.shape[0] > zv.shape[0] else field
	zv    = 0.5 * (zv[:-1, :] + zv[1:, :]) if zv.shape[0] > field.shape[0] else zv
	field = 0.5 * (field[:, :-1] + field[:, 1:]) if field.shape[1] > zv.shape[1] else field
	z     = 0.5 * (zv[:, :-1] + zv[:, 1:]) if zv.shape[1] > field.shape[1] else zv

	xv = to_units(grid.x, xaxis_units).values if field.shape[0] == grid.nx \
		 else to_units(grid.x_at_u_locations, xaxis_units).values
	x  = np.repeat(xv[:, np.newaxis], z.shape[1], axis=1)

	make_contourf(x, z, field, fig, ax, **kwargs)

	if topo_retriever is not None:
		topo  = np.squeeze(topo_retriever(state))
		topo  = 0.5 * (topo[:-1] + topo[1:]) if topo.shape[0] > xv.shape[0] else topo
		topox = 0.5 * (xv[:-1] + xv[1:]) if xv.shape[0] > topo.shape[0] else xv

		make_lineplot(topox, topo, ax, **kwargs)


def make_contourf_yz(grid, yaxis_units, zaxis_units, field_retriever,
					 state, fig, ax, **kwargs):
	field = np.squeeze(field_retriever(state))

	yv = to_units(grid.y, yaxis_units).values if field.shape[0] == grid.ny \
		 else to_units(grid.y_at_v_locations, yaxis_units).values
	zv = to_units(grid.z, zaxis_units).values if field.shape[1] == grid.nz \
		 else to_units(grid.z_on_interface_levels, zaxis_units).values
	y  = np.repeat(yv[:, np.newaxis], zv.shape[0], axis=1)
	z  = np.repeat(zv[np.newaxis, :], yv.shape[0], axis=0)

	make_contourf(y, z, field, fig, ax, **kwargs)


def make_contourf_yh(grid, yaxis_units, zaxis_retriever, topo_retriever,
					 field_retriever, state, fig, ax, **kwargs):
	field = np.squeeze(field_retriever(state))

	zv    = np.squeeze(zaxis_retriever(state))
	field = 0.5 * (field[:-1, :] + field[1:, :]) if field.shape[0] > zv.shape[0] else field
	zv    = 0.5 * (zv[:-1, :] + zv[1:, :]) if zv.shape[0] > field.shape[0] else zv
	field = 0.5 * (field[:, :-1] + field[:, 1:]) if field.shape[1] > zv.shape[1] else field
	z     = 0.5 * (zv[:, :-1] + zv[:, 1:]) if zv.shape[1] > field.shape[1] else zv

	yv  = to_units(grid.y, yaxis_units).values if field.shape[0] == grid.ny \
		  else to_units(grid.y_at_v_locations, yaxis_units).values
	y  = np.repeat(yv[:, np.newaxis], z.shape[1], axis=1)

	make_contourf(y, z, field, fig, ax, **kwargs)

	if topo_retriever is not None:
		topo  = np.squeeze(topo_retriever(state))
		topo  = 0.5 * (topo[:-1] + topo[1:]) if topo.shape[0] > yv.shape[0] else topo
		topoy = 0.5 * (yv[:-1] + yv[1:]) if yv.shape[0] > topo.shape[0] else yv

		make_lineplot(topoy, topo, ax, **kwargs)
