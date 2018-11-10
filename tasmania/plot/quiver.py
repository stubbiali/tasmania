# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version. 
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
"""
This module contains:
	Quiver(Drawer)
	make_quiver_xy
	make_quiver_xz
	make_quiver_xh
	make_quiver_yz
	make_quiver_yh
"""
import numpy as np

from tasmania.plot.drawer import Drawer
from tasmania.plot.plot_utils import make_contour, make_lineplot, make_quiver
from tasmania.plot.retrievers import DataRetriever
from tasmania.plot.utils import to_units


class Quiver(Drawer):
	"""
	Drawer which generates a quiver plot of a vector-valued state quantity
	at a cross-section parallel to one coordinate plane.
	"""
	def __init__(self, grid, x=None, y=None, z=None,
				 xcomp_name=None, xcomp_units=None,
				 ycomp_name=None, ycomp_units=None,
				 zcomp_name=None, zcomp_units=None,
				 scalar_name=None, scalar_units=None,
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
		x : `int`, optional
			Index along the first dimension of the components arrays identifying
			the cross-section to visualize. To be specified only if both :obj:`y`
			and :obj:`z` are not given.
		y : `int`, optional
			Index along the second dimension of the components arrays identifying
			the cross-section to visualize. To be specified only if both :obj:`x`
			and :obj:`z` are not given.
		z : `int`, optional
			Index along the third dimension of the components arrays identifying
			the cross-section to visualize. To be specified only if both :obj:`x`
			and :obj:`y` are not given.
		xcomp_name : `str`, optional
			The vector-valued field component along the first computational axis.
			Required if either :obj:`y` or :obj:`z` is given.
		xcomp_units : `str`, optional
			The units for the :obj:`xcomp_name` component.
		ycomp_name : `str`, optional
			The vector-valued field component along the second computational axis.
			Required if either :obj:`x` or :obj:`z` is given.
		ycomp_units : `str`, optional
			The units for the :obj:`ycomp_name` component.
		zcomp_name : `str`, optional
			The vector-valued field component along the third computational axis.
			Required if either :obj:`x` or :obj:`y` is given.
		zcomp_units : `str`, optional
			The units for the :obj:`zcomp_name` component.
		scalar_name : `str`, optional
			The name of the scalar quantity associated with the vector-valued field
			and used to generate the colormap. If not specified, the Euclidean norm
			of the vector-valued field is used.
		scalar_units : `str`, optional
			The units for the field :obj:`scalar_name`.
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
			:func:`tasmania.plot.utils.make_quiver`,
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

		if not flag_x:
			assert xcomp_name is not None, 'Please specify the x-component.'
			xcomp_retriever = DataRetriever(grid, xcomp_name, xcomp_units,
											slice_x, slice_y, slice_z)
		else:
			xcomp_retriever = None

		if not flag_y:
			assert ycomp_name is not None, 'Please specify the y-component.'
			ycomp_retriever = DataRetriever(grid, ycomp_name, ycomp_units,
											slice_x, slice_y, slice_z)
		else:
			ycomp_retriever = None

		if not flag_z:
			assert zcomp_name is not None, 'Please specify the z-component.'
			zcomp_retriever = DataRetriever(grid, zcomp_name, zcomp_units,
											slice_x, slice_y, slice_z)
		else:
			zcomp_retriever = None

		if scalar_name is not None:
			scalar_retriever = DataRetriever(grid, scalar_name, scalar_units,
											 slice_x, slice_y, slice_z)
		else:
			scalar_retriever = None

		if flag_z:
			topo_retriever   = DataRetriever(grid, 'topography', topography_units)

			self._slave = lambda state, fig, ax: make_quiver_xy(
				grid, xaxis_units, yaxis_units, xcomp_retriever, ycomp_retriever,
				scalar_retriever, topo_retriever, state, fig, ax, **self.properties)
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
					self._slave = lambda state, fig, ax: make_quiver_yh(
						grid, yaxis_units, zaxis_retriever, ycomp_retriever, zcomp_retriever,
						scalar_retriever, topo_retriever, state, fig, ax, **self.properties
					)
				else:
					self._slave = lambda state, fig, ax: make_quiver_xh(
						grid, xaxis_units, zaxis_retriever, xcomp_retriever, zcomp_retriever,
						scalar_retriever, topo_retriever, state, fig, ax, **self.properties
					)
			else:
				if flag_x:
					self._slave = lambda state, fig, ax: make_quiver_yz(
						grid, yaxis_units, zaxis_units, ycomp_retriever, zcomp_retriever,
						scalar_retriever, state, fig, ax, **self.properties
					)
				else:
					self._slave = lambda state, fig, ax: make_quiver_xz(
						grid, xaxis_units, zaxis_units, xcomp_retriever, zcomp_retriever,
						scalar_retriever, state, fig, ax, **self.properties
					)

	def __call__(self, state, fig, ax):
		"""
		Call operator generating the quiver plot.
		"""
		self._slave(state, fig, ax)


def make_quiver_xy(grid, xaxis_units, yaxis_units, xcomp_retriever, ycomp_retriever,
				   scalar_retriever, topo_retriever, state, fig, ax, **kwargs):
	vx = np.squeeze(xcomp_retriever(state))
	vy = np.squeeze(ycomp_retriever(state))

	if scalar_retriever is None:
		vx = 0.5 * (vx[:-1, :] + vx[1:, :]) if vx.shape[0] > vy.shape[0] else vx
		vy = 0.5 * (vy[:-1, :] + vy[1:, :]) if vy.shape[0] > vx.shape[0] else vy

		vx = 0.5 * (vx[:, :-1] + vx[:, 1:]) if vx.shape[1] > vy.shape[1] else vx
		vy = 0.5 * (vy[:, :-1] + vy[:, 1:]) if vy.shape[1] > vx.shape[1] else vy

		sc = np.sqrt(vx**2 + vy**2)
	else:
		sc = np.squeeze(scalar_retriever(state))

		vx = 0.5 * (vx[:-1, :] + vx[1:, :]) \
			 if vx.shape[0] > vy.shape[0] or vx.shape[0] > sc.shape[0] else vx
		vy = 0.5 * (vy[:-1, :] + vy[1:, :]) \
			 if vy.shape[0] > vx.shape[0] or vy.shape[0] > sc.shape[0] else vy
		sc = 0.5 * (sc[:-1, :] + sc[1:, :]) \
			 if sc.shape[0] > vx.shape[0] or sc.shape[0] > vy.shape[0] else sc

		vx = 0.5 * (vx[:, :-1] + vx[:, 1:]) \
			 if vx.shape[1] > vy.shape[1] or vx.shape[1] > sc.shape[1] else vx
		vy = 0.5 * (vy[:, :-1] + vy[:, 1:]) \
			 if vy.shape[1] > vx.shape[1] or vy.shape[1] > sc.shape[1] else vy
		sc = 0.5 * (sc[:, :-1] + sc[:, 1:]) \
			 if sc.shape[1] > vx.shape[1] or sc.shape[1] > vy.shape[1] else sc

	xv = to_units(grid.x, xaxis_units).values if vx.shape[0] == grid.nx \
		else to_units(grid.x_at_u_locations, xaxis_units).values
	yv = to_units(grid.y, yaxis_units).values if vx.shape[1] == grid.ny \
		else to_units(grid.y_at_v_locations, yaxis_units).values
	x  = np.repeat(xv[:, np.newaxis], yv.shape[0], axis=1)
	y  = np.repeat(yv[np.newaxis, :], xv.shape[0], axis=0)

	make_quiver(x, y, vx, vy, sc, fig, ax, **kwargs)

	topo = topo_retriever(state)
	topo = 0.5 * (topo[:-1, :] + topo[1:, :]) if topo.shape[0] > x.shape[0] else topo
	topo = 0.5 * (topo[:, :-1] + topo[:, 1:]) if topo.shape[1] > y.shape[1] else topo

	make_contour(x, y, topo, ax, **kwargs)


def make_quiver_xz(grid, xaxis_units, zaxis_units, xcomp_retriever, zcomp_retriever,
				   scalar_retriever, state, fig, ax, **kwargs):
	vx = np.squeeze(xcomp_retriever(state))
	vy = np.squeeze(zcomp_retriever(state))

	if scalar_retriever is None:
		vx = 0.5 * (vx[:-1, :] + vx[1:, :]) if vx.shape[0] > vy.shape[0] else vx
		vy = 0.5 * (vy[:-1, :] + vy[1:, :]) if vy.shape[0] > vx.shape[0] else vy

		vx = 0.5 * (vx[:, :-1] + vx[:, 1:]) if vx.shape[1] > vy.shape[1] else vx
		vy = 0.5 * (vy[:, :-1] + vy[:, 1:]) if vy.shape[1] > vx.shape[1] else vy

		sc = np.sqrt(vx**2 + vy**2)
	else:
		sc = np.squeeze(scalar_retriever(state))

		vx = 0.5 * (vx[:-1, :] + vx[1:, :]) \
			 if vx.shape[0] > vy.shape[0] or vx.shape[0] > sc.shape[0] else vx
		vy = 0.5 * (vy[:-1, :] + vy[1:, :]) \
			 if vy.shape[0] > vx.shape[0] or vy.shape[0] > sc.shape[0] else vy
		sc = 0.5 * (sc[:-1, :] + sc[1:, :]) \
			 if sc.shape[0] > vx.shape[0] or sc.shape[0] > vy.shape[0] else sc

		vx = 0.5 * (vx[:, :-1] + vx[:, 1:]) \
			 if vx.shape[1] > vy.shape[1] or vx.shape[1] > sc.shape[1] else vx
		vy = 0.5 * (vy[:, :-1] + vy[:, 1:]) \
			 if vy.shape[1] > vx.shape[1] or vy.shape[1] > sc.shape[1] else vy
		sc = 0.5 * (sc[:, :-1] + sc[:, 1:]) \
			 if sc.shape[1] > vx.shape[1] or sc.shape[1] > vy.shape[1] else sc

	xv = to_units(grid.x, xaxis_units).values if vx.shape[0] == grid.nx \
		 else to_units(grid.x_at_u_locations, xaxis_units).values
	yv = to_units(grid.z, zaxis_units).values if vx.shape[1] == grid.nz \
		 else to_units(grid.z_on_interface_levels, zaxis_units).values
	x  = np.repeat(xv[:, np.newaxis], yv.shape[0], axis=1)
	y  = np.repeat(yv[np.newaxis, :], xv.shape[0], axis=0)

	make_quiver(x, y, vx, vy, sc, fig, ax, **kwargs)


def make_quiver_xh(grid, xaxis_units, zaxis_retriever, xcomp_retriever, zcomp_retriever,
				   scalar_retriever, topo_retriever, state, fig, ax, **kwargs):
	raise NotImplementedError()


def make_quiver_yz(grid, yaxis_units, zaxis_units, ycomp_retriever, zcomp_retriever,
				   scalar_retriever, state, fig, ax, **kwargs):
	vx = np.squeeze(ycomp_retriever(state))
	vy = np.squeeze(zcomp_retriever(state))

	if scalar_retriever is None:
		vx = 0.5 * (vx[:-1, :] + vx[1:, :]) if vx.shape[0] > vy.shape[0] else vx
		vy = 0.5 * (vy[:-1, :] + vy[1:, :]) if vy.shape[0] > vx.shape[0] else vy

		vx = 0.5 * (vx[:, :-1] + vx[:, 1:]) if vx.shape[1] > vy.shape[1] else vx
		vy = 0.5 * (vy[:, :-1] + vy[:, 1:]) if vy.shape[1] > vx.shape[1] else vy

		sc = np.sqrt(vx**2 + vy**2)
	else:
		sc = np.squeeze(scalar_retriever(state))

		vx = 0.5 * (vx[:-1, :] + vx[1:, :]) \
			 if vx.shape[0] > vy.shape[0] or vx.shape[0] > sc.shape[0] else vx
		vy = 0.5 * (vy[:-1, :] + vy[1:, :]) \
			 if vy.shape[0] > vx.shape[0] or vy.shape[0] > sc.shape[0] else vy
		sc = 0.5 * (sc[:-1, :] + sc[1:, :]) \
			 if sc.shape[0] > vx.shape[0] or sc.shape[0] > vy.shape[0] else sc

		vx = 0.5 * (vx[:, :-1] + vx[:, 1:]) \
			 if vx.shape[1] > vy.shape[1] or vx.shape[1] > sc.shape[1] else vx
		vy = 0.5 * (vy[:, :-1] + vy[:, 1:]) \
			 if vy.shape[1] > vx.shape[1] or vy.shape[1] > sc.shape[1] else vy
		sc = 0.5 * (sc[:, :-1] + sc[:, 1:]) \
			 if sc.shape[1] > vx.shape[1] or sc.shape[1] > vy.shape[1] else sc

	xv = to_units(grid.y, yaxis_units).values if vx.shape[0] == grid.ny \
		 else to_units(grid.y_at_v_locations, yaxis_units).values
	yv = to_units(grid.z, zaxis_units).values if vx.shape[1] == grid.nz \
		 else to_units(grid.z_on_interface_levels, zaxis_units).values
	x  = np.repeat(xv[:, np.newaxis], yv.shape[0], axis=1)
	y  = np.repeat(yv[np.newaxis, :], xv.shape[0], axis=0)

	make_quiver(x, y, vx, vy, sc, fig, ax, **kwargs)


def make_quiver_yh(grid, yaxis_units, zaxis_retriever, xcomp_retriever, zcomp_retriever,
				   scalar_retriever, topo_retriever, state, fig, ax, **kwargs):
	raise NotImplementedError()
