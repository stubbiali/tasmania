"""
Meteo-oriented utilities.
"""
import numpy as np

from namelist import cp, datatype, g, p_ref, Rd
from utils.utils import smaller_than as lt

def get_isentropic_isothermal_analytical_solution(grid, x_velocity_initial, temperature, mountain_height, mountain_width,
									   			  x_staggered = True, z_staggered = False):
	"""
	Get the analytical expression of a two-dimensional, hydrostatic, isentropic and isothermal flow over an isolated
	`Switch of Agnesi` mountain.

	Parameters
	----------
	grid : obj
		:class:`~grids.grid_xyz.GridXYZ` representing the underlying grid. It must consist of only one points in :math:`y`-direction.
	x_velocity_initial : float
		The initial :math:`x`-velocity, in units of [:math:`m \, s^{-1}`].
	temperature : float
		The temperature, in units of [:math:`K`].
	mountain_height : float
		The maximum mountain height, in units of [:math:`m`].
	mountain_width : float
		The mountain half-width at half-height, in units of [:math:`m`].
	x_staggered : `bool`, optional
		:obj:`True` if the solution should be staggered in the :math:`x`-direction, :obj:`False` otherwise.
		Default is :obj:`True`.
	z_staggered : `bool`, optional
		:obj:`True` if the solution should be staggered in the vertical direction, :obj:`False` otherwise.
		Default is :obj:`False`.

	Returns
	-------
	u : array_like
		:class:`numpy.ndarray` representing the :math:`x`-velocity.
	w : array_like
		:class:`numpy.ndarray` representing the vertical velocity.

	References
	----------
	Durran, D. R. (1981). `The effects of moisture on mountain lee waves`. \
		Doctoral dissertation, Massachussets Institute of Technology.
	"""
	# Ensure the computational domain consists of only one grid-point in y-direction
	assert grid.ny == 1

	# Shortcuts
	u_bar, T, h, a = x_velocity_initial, temperature, mountain_height, mountain_width
	nx, nz = grid.nx, grid.nz

	# Compute Scorer parameter
	l = np.sqrt((g ** 2) / (cp * T  * (u_bar ** 2)) - (g ** 2) / (4. * (Rd ** 2) * (T ** 2)))

	# Build the underlying x-z grid
	xv = grid.x_half_levels.values if x_staggered else grid.x.values
	zv = grid.z_half_levels.values if z_staggered else grid.z.values
	x, theta = np.meshgrid(xv, zv, indexing = 'ij')
	
	# The topography
	zs = h * (a ** 2) / ((x ** 2) + (a ** 2))

	# The geometric height
	theta_s = grid.z_half_levels.values[-1]
	z = zs + cp * T / g * np.log(theta / theta_s)
	dz_dx = - 2. * h * (a ** 2) * x / (((x ** 2) + (a ** 2)) ** 2)
	dz_dtheta = cp * T / (g * theta)

	# Compute mean pressure
	p_bar = p_ref * (T / theta) ** (cp / Rd)

	# Base and mean density
	rho_ref = p_ref / (Rd * T)
	rho_bar = p_bar / (Rd * T)
	drho_bar_dtheta = - cp * p_ref / ((Rd ** 2) * (T ** 2)) * ((T / theta) ** (cp / Rd + 1.))

	# Compute the streamlines displacement and its derivative
	d = ((rho_bar / rho_ref) ** (-0.5)) * h * a * (a * np.cos(l * z) - x * np.sin(l * z)) / ((x ** 2) + (a ** 2))
	dd_dx = - ((rho_bar / rho_ref) ** (-0.5)) * h * a / (((x ** 2) + (a ** 2)) ** 2) * \
			(((a * np.sin(l * z) + x * np.cos(l * z)) * l * dz_dx + np.sin(l * z)) * ((x ** 2) + (a ** 2)) +
			 2. * x * (a * np.cos(l * z) - x * np.sin(l * z)))
	dd_dtheta = 0.5 * cp / (Rd * T) * ((theta / T) ** (0.5 * cp / Rd - 1.)) * \
				h * a * (a * np.cos(l * z) - x * np.sin(l * z)) / ((x ** 2) + (a ** 2)) - \
				((theta / T) ** (0.5 * cp / Rd)) * h * a * (a * np.sin(l * z) + x * np.cos(l * z)) * l * dz_dtheta / \
				((x ** 2) + (a ** 2))
	dd_dz = dd_dtheta / dz_dtheta

	# Compute the horizontal and vertical velocity
	u = u_bar * (1. - drho_bar_dtheta * d / (dz_dtheta * rho_bar) - dd_dz)
	w = u_bar * dd_dx

	return u, w

def convert_relative_humidity_to_water_vapor(method, p, T, rh):
	"""
	Convert relative humidity to water vapor mixing ratio.

	Parameters
	----------
	method : str
		String specifying the formula to be used to compute the saturation water vapor pressure. Either:

		* 'teten', for the Teten's formula;
		* 'goff_gratch', for the Goff-Gratch formula.

	p : array_like
		:class:`numpy.ndarray` representing the pressure ([:math:`Pa`]).
	T : array_like
		:class:`numpy.ndarray` representing the temperature ([:math:`K`]).
	rh : array_like
		:class:`numpy.ndarray` representing the relative humidity ([:math:`-`]).
	
	Return
	------
	array_like :
		:class:`numpy.ndarray` representing the fraction of water vapor ([:math:`g \, g^{-1}`]).

	References
	----------
	Vaisala, O. (2013). `Humidity conversion formulas: Calculation formulas for humidity`. Retrieved from \
		`<https://www.vaisala.com>`_.
	"""
	# Get the saturation water vapor pressure
	if method == 'teten':
		p_sat = apply_teten_formula(T)
	elif method == 'goff_gratch':
		p_sat = apply_goff_gratch_formula(T)
	else:
		raise ValueError("""Unknown formula to compute the saturation water vapor pressure.\n"""
						 """Available options are: ''teten'', ''goff_gratch''.""")

	# Compute the water vapor presure
	pw = rh * p_sat

	# Compute the mixing ratio of water vapor
	B = 0.62198
	qv = np.where(p_sat >= 0.616 * p, 0., B * pw / (p - pw))

	return qv

def apply_teten_formula(T):
	"""
	Compute the saturation vapor pressure over water at a given temperature, relying upon the Teten's formula.

	Parameters
	----------
	T : array_like
		:class:`numpy.ndarray` representing the temperature ([:math:`K`]).

	Return
	------
	array_like :
		:class:`numpy.ndarray` representing the saturation water vapor pressure ([:math:`Pa`]).
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
	Compute the saturation vapor pressure over water at a given temperature, relying upon the Goff-Gratch formula.

	Parameters
	----------
	T : array_like
		:class:`numpy.ndarray` representing the temperature ([:math:`K`]).

	Return
	------
	array_like :
		:class:`numpy.ndarray` representing the saturation water vapor pressure ([:math:`Pa`]).

	References
	----------
	Goff, J. A., and S. Gratch. (1946). `Low-pressure properties of water from -160 to 212 F`. \
		Transactions of the American Society of Heating and Ventilating Engineers, 95-122.
	"""
	# Constants occurring in the Goff-Gratch formula
	C1 = 7.90298
	C2 = 5.02808
	C3 = 1.3816e-7
	C4 = 11.344
	C5 = 8.1328e-3
	C6 = 3.49149

	# The steam-point (i.e., boiling point at 1 atm) temperature, and the saturation water vapor pressure at the steam-point
	T_st = 373.15
	e_st = 1013.25e2
	
	# Apply the Goff-Gratch formula to compute the saturation water vapor pressure
	e = e_st * 10 ** (- C1 * (T_st / T - 1.) 
					  + C2 * np.log10(T_st / T)
					  - C3 * (10. ** (C4 * (1. - T / T_st)) - 1.)
					  + C5 * (10 ** (- C6 * (T_st / T - 1.)) - 1.))

	return e
