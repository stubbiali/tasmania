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
	get_isothermal_isentropic_analytical_solution
	convert_relative_humidity_to_water_vapor
	_apply_teten_formula
	_apply_goff_gratch_formula
"""
import numpy as np
from sympl import DataArray

from tasmania.python.utils.data_utils import \
	get_physical_constants, make_dataarray_3d

try:
	from tasmania.conf import datatype
except ImportError:
	datatype = np.float32


_d_physical_constants = {
	'gas_constant_of_dry_air':
		DataArray(287.05, attrs={'units': 'J K^-1 kg^-1'}),
	'gravitational_acceleration':
		DataArray(9.81, attrs={'units': 'm s^-2'}),
	'reference_air_pressure':
		DataArray(1.0e5, attrs={'units': 'Pa'}),
	'specific_heat_of_dry_air_at_constant_pressure':
		DataArray(1004.0, attrs={'units': 'J K^-1 kg^-1'}),
}


def get_isothermal_isentropic_analytical_solution(
	grid, x_velocity_initial, temperature, mountain_height, mountain_width,
	x_staggered=True, z_staggered=False, physical_constants=None
):
	"""
	Get the analytical expression of a two-dimensional, hydrostatic, isentropic
	and isothermal flow over an isolated 'Switch of Agnesi' mountain.

	Parameters
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` representing the underlying grid.
		It must consist of only one points in the :math:`y`-direction.
	x_velocity_initial : dataarray_like
		One-item :class:`sympl.DataArray` representing the initial :math:`x`-velocity.
	temperature : dataarray_like
		One-item :class:`sympl.DataArray` representing the uniform air temperature.
	mountain_height : dataarray_like
		One-item :class:`sympl.DataArray` representing the maximum mountain height.
	mountain_width : dataarray_like
		One-item :class:`sympl.DataArray` representing the mountain half-width
		at half-height.
	x_staggered : `bool`, optional
		:obj:`True` if the solution should be staggered in the :math:`x`-direction,
		:obj:`False` otherwise. Default is :obj:`True`.
	z_staggered : `bool`, optional
		:obj:`True` if the solution should be staggered in the vertical direction,
		:obj:`False` otherwise. Default is :obj:`False`.
	physical_constants : `dict_like`, optional
		Dictionary whose keys are strings indicating physical constants used
		within this object, and whose values are :class:`sympl.DataArray`\s
		storing the values and units of those constants. The constants might be:

            * 'gas_constant_of_dry_air', in units compatible with \
                [J K^-1 kg^-1];
            * 'gravitational acceleration', in units compatible with [m s^-2];
            * 'reference_air_pressure', in units compatible with [Pa];
            * 'specific_heat_of_dry_air_at_constant_pressure', in units compatible \
                with [J K^-1 kg^-1].

        Please refer to
        :func:`tasmania.utils.data_utils.get_physical_constants` and
        :obj:`tasmania.utils.meteo_utils._d_physical_constants`
        for the default values.

	Returns
	-------
	u : dataarray_like
		:class:`sympl.DataArray` representing the :math:`x`-velocity.
	w : dataarray_like
		:class:`sympl.DataArray` representing the vertical velocity.

	References
	----------
	Durran, D. R. (1981). _The effects of moisture on mountain lee waves_. \
		Doctoral dissertation, Massachussets Institute of Technology.
	"""
	# Ensure the computational domain consists of only one grid-point in y-direction
	assert grid.ny == 1

	# Shortcuts
	u_bar = x_velocity_initial.to_units('m s^-1').values.item()
	T = temperature.to_units('K').values.item()
	h = mountain_height.to_units('m').values.item()
	a = mountain_width.to_units(grid.x.attrs['units']).values.item()

	# Get physical constants
	pcs   = get_physical_constants(_d_physical_constants, physical_constants)
	Rd    = pcs['gas_constant_of_dry_air']
	g     = pcs['gravitational_acceleration']
	p_ref = pcs['reference_air_pressure']
	cp    = pcs['specific_heat_of_dry_air_at_constant_pressure']

	# Compute Scorer parameter
	scpam = np.sqrt((g**2) / (cp * T  * (u_bar**2)) - (g**2) / (4. * (Rd**2) * (T**2)))

	# Build the underlying x-z grid
	xv = grid.x_at_u_locations.values if x_staggered else grid.x.values
	zv = grid.z_on_interface_levels.values if z_staggered else grid.z.values
	x, theta = np.meshgrid(xv, zv, indexing='ij')
	
	# The topography
	zs = h * (a**2) / ((x**2) + (a**2))

	# The geometric height
	theta_s = grid.z_on_interface_levels.to_units('K').values[-1]
	z = zs + cp * T / g * np.log(theta / theta_s)
	dz_dx = - 2. * h * (a**2) * x / (((x**2) + (a**2))**2)
	dz_dtheta = cp * T / (g * theta)

	# Compute mean pressure
	p_bar = p_ref * (T / theta) ** (cp / Rd)

	# Base and mean density
	rho_ref = p_ref / (Rd * T)
	rho_bar = p_bar / (Rd * T)
	drho_bar_dtheta = - cp * p_ref / ((Rd**2) * (T**2)) * ((T / theta)**(cp / Rd + 1.))

	# Compute the streamlines displacement and its derivative
	d = ((rho_bar / rho_ref)**(-0.5)) * h * a * \
		(a * np.cos(scpam*z) - x * np.sin(scpam*z)) / ((x**2) + (a**2))
	dd_dx = - ((rho_bar / rho_ref)**(-0.5)) * h * a / (((x**2) + (a**2))**2) * \
			(((a * np.sin(scpam*z) + x * np.cos(scpam*z)) * scpam * dz_dx + np.sin(scpam*z)) *
			 ((x**2) + (a**2)) + 2. * x * (a * np.cos(scpam*z) - x * np.sin(scpam*z)))
	dd_dtheta = 0.5 * cp / (Rd * T) * ((theta / T)**(0.5 * cp / Rd - 1.)) * \
				h * a * (a * np.cos(scpam*z) - x * np.sin(scpam*z)) / ((x**2) + (a**2)) - \
				((theta / T)**(0.5 * cp / Rd)) * h * a * \
				(a * np.sin(scpam*z) + x * np.cos(scpam*z)) * scpam * dz_dtheta / \
				((x**2) + (a**2))
	dd_dz = dd_dtheta / dz_dtheta

	# Compute the horizontal and vertical velocity
	u_ = u_bar * (1. - drho_bar_dtheta * d / (dz_dtheta * rho_bar) - dd_dz)
	u  = make_dataarray_3d(u_[:, np.newaxis, :], grid, 'm s^-1')
	w_ = u_bar * dd_dx
	w  = make_dataarray_3d(w_[:, np.newaxis, :], grid, 'm s^-1')

	return u, w


def convert_relative_humidity_to_water_vapor(method, p, T, rh):
	"""
	Convert relative humidity to water vapor mixing ratio.

	Parameters
	----------
	method : str
		String specifying the formula to be used to compute the
		saturation water vapor pressure. Either:

			* 'teten', for the Teten's formula;
			* 'goff_gratch', for the Goff-Gratch formula.

	p : dataarray_like
		:class:`sympl.DataArray` representing the pressure.
	T : dataarray_like
		:class:`sympl.DataArray` representing the temperature.
	rh : dataarray_like
		:class:`sympl.DataArray` representing the relative humidity.
	
	Return
	------
	array_like :
		:class:`numpy.ndarray` representing the mass fraction of water vapor,
		in units of ([:math:`g \, g^{-1}`]).

	References
	----------
	Vaisala, O. (2013). _Humidity conversion formulas: Calculation formulas for humidity_. \
		Retrieved from `<https://www.vaisala.com>`_.
	"""
	# Extract the raw arrays
	p_  = p.to_units('Pa').values
	T_  = T.to_units('K').values
	rh_ = rh.to_units('1').values

	# Get the saturation water vapor pressure
	if method == 'teten':
		p_sat = apply_teten_formula(T_)
	elif method == 'goff_gratch':
		p_sat = apply_goff_gratch_formula(T_)
	else:
		raise ValueError("""Unknown formula to compute the saturation water vapor pressure.\n"""
						 """Available options are: ''teten'', ''goff_gratch''.""")

	# Compute the water vapor pressure
	pw = rh_ * p_sat

	# Compute the mixing ratio of water vapor
	B = 0.62198
	qv = np.where(p_sat >= 0.616 * p_, 0., B * pw / (p_ - pw))

	return qv


def apply_teten_formula(T):
	"""
	Compute the saturation vapor pressure over water at a given temperature,
	relying upon the Teten's formula.

	Parameters
	----------
	T : array_like
		:class:`numpy.ndarray` representing the temperature ([:math:`K`]).

	Return
	------
	array_like :
		:class:`numpy.ndarray` representing the saturation water
		vapor pressure ([:math:`Pa`]).
	"""
	# Constants occurring in the Teten's formula
	pw = 610.78
	aw = 17.27
	Tr = 273.16
	bw = 35.86

	# Apply the Teten's formula to compute the saturation water vapor pressure
	e = pw * np.exp(aw * (T - Tr) / (T - bw))

	return e


def apply_goff_gratch_formula(T):
	"""
	Compute the saturation vapor pressure over water at a given temperature,
	relying upon the Goff-Gratch formula.

	Parameters
	----------
	T : array_like
		:class:`numpy.ndarray` representing the temperature ([:math:`K`]).

	Return
	------
	array_like :
		:class:`numpy.ndarray` representing the saturation
		water vapor pressure ([:math:`Pa`]).

	References
	----------
	Goff, J. A., and S. Gratch. (1946). `Low-pressure properties of water from -160 to 212 F`. \
		*Transactions of the American Society of Heating and Ventilating Engineers*, 95-122.
	"""
	# Constants occurring in the Goff-Gratch formula
	C1 = 7.90298
	C2 = 5.02808
	C3 = 1.3816e-7
	C4 = 11.344
	C5 = 8.1328e-3
	C6 = 3.49149

	# The steam-point (i.e., boiling point at 1 atm) temperature,
	# and the saturation water vapor pressure at the steam-point
	T_st = 373.15
	e_st = 1013.25e2
	
	# Apply the Goff-Gratch formula to compute the saturation water vapor pressure
	e = e_st * 10 ** (- C1 * (T_st / T - 1.) 
					  + C2 * np.log10(T_st / T)
					  - C3 * (10. ** (C4 * (1. - T / T_st)) - 1.)
					  + C5 * (10 ** (- C6 * (T_st / T - 1.)) - 1.))

	return e
