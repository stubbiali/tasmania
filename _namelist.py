""" 
Configuration and global variables used throughout the package. 

Physical constants:
	* :data:`namelist.p_ref`: Reference pressure ([:math:`Pa`]).
	* :data:`namelist.p_sl`: Reference pressure at sea level ([:math:`Pa`]).
	* :data:`namelist.T_sl`: Reference temperature at sea level ([:math:`K`]).
	* :data:`namelist.beta`: Rate of increase in reference temperature with the logarithm \
		of reference pressure ([:math:`K ~ Pa^{-1}`]).
	* :data:`namelist.Rd`: Gas constant for dry airi ([:math:`J ~ K^{-1} ~ Kg^{-1}`]).
	* :data:`namelist.cp`: Specific heat of dry air at constant pressure ([:math:`J ~ K^{-1} ~ Kg^{-1}`]).
	* :data:`namelist.g`: Mean gravitational acceleration ([:math:`m ~ s^{-2}`]). 

Grid settings:
	* :data:`namelist.domain_x`: Tuple storing the boundaries of the domain in the :math:`x`-direction \
		in the form (:math:`x_{west}`, :math:`x_{east}`).
	* :data:`namelist.nx`: Number of grid points in the :math:`x`-direction.
	* :data:`namelist.domain_y`: Tuple storing the boundaries of the domain in the :math:`y`-direction \
		in the form (:math:`y_{south}`, :math:`y_{north}`).
	* :data:`namelist.ny`: Number of grid points in the :math:`y`-direction.
	* :data:`namelist.domain_z`: Tuple storing the boundaries of the domain in the :math:`z`-direction \
		in the form (:math:`z_{top}`, :math:`z_{bottom}`).
	* :data:`namelist.nz`: Number of grid points in the :math:`z`-direction.
	* :data:`namelist.z_interface`: For a hybrid coordinate system, interface level at which terrain-following \
		:math:`z`-coordinate lines get back to horizontal lines.
	* :data:`namelist.topo_type`: Topography type. Available options are:

			- 'flat_terrain';
			- 'gaussian';
			- 'schaer';
			- 'user_defined'.

	* :data:`namelist.topo_time`: :class:`datetime.timedelta` object representing the elapsed simulation time \
				after which the topography should stop increasing.
	* :data:`namelist.topo_max_height`: When :data:`~namelist.topo_type` is 'gaussian', maximum mountain height ([:math:`m`]).
	* :data:`namelist.topo_width_x`: When :data:`~namelist.topo_type` is 'gaussian', mountain half-width in :math:`x`-direction \
		([:math:`m`]).
	* :data:`namelist.topo_width_y`: When :data:`~namelist.topo_type` is 'gaussian', mountain half-width in :math:`y`-direction \
		([:math:`m`]).
	* :data:`namelist.topo_str`: When :data:`~namelist.topo_type` is 'user_defined', terrain profile expression in the independent \
		variables :math:`x` and :math:`y`. Must be fully C++-compliant.

Model settings:
	* :data:`namelist.model_name`: Name of the model to implement. Available options are:

			- 'isentropic', for the isentropic model.

	* :data:`namelist.imoist`: :data:`True` if water constituents should be taken into account, :data:`False` otherwise.
	* :data:`namelist.horizontal_boundary_type`: Horizontal boundary conditions. Available options are:

			- 'periodic', for periodic boundary conditions;
			- 'relaxed', for relaxed boundary conditions. 

Numerical settings:
	* :data:`namelist.scheme`: Numerical scheme to implement. For the isentropic model, available options are:

			- 'upwind', for the first-order upwind scheme;
			- 'leapfrog', for the second-order leapfrog scheme;
			- 'maccormack', for the second-order maccormack scheme.

	* :data:`namelist.idamp`: :data:`True` if (explicit) vertical damping should be applied, :data:`False` otherwise. \
		Note that when vertical damping is switched off, the numerical diffusion is monotonically increased towards \
		the top of the model, so to act as a diffusive wave absorber.
	* :data:`namelist.damp_type`: Type of vertical damping to apply. Available options are:
		
			- 'rayleigh', for Rayleigh vertical damping.

	* :data:`namelist.damp_depth`: Number of levels (either main levels or half levels) in the absorbing region.
	* :data:`namelist.damp_max`: Maximum value which should be assumed by the damping coefficient.
	* :data:`namelist.idiff`: :data:`True` to add numerical horizontal diffusion, :data:`False` otherwise.
	* :data:`namelist.diff_coeff`: The diffusion coefficient, i.e., the diffusivity.
	* :data:`namelist.diff_coeff_moist`: The diffusion coefficient, i.e., the diffusivity, for the moisture components.
	* :data:`namelist.diff_max`: Maximum value which should be assumed by the diffusivity when diffusive vertical damping is applied.

Simulation settings:
	* :data:`namelist.dt`: :class:`datetime.timedelta` object representing the timestep.
	* :data:`namelist.initial_time`: :class:`datetime.datetime` representing the initial simulation time.
	* :data:`namelist.simulation_time`: :class:`datetime.timedelta` object representing the simulation time.
	* :data:`namelist.x_velocity_initial`: The initial, uniform :math:`x`-velocity ([:math:`m s^{-1}`]).
	* :data:`namelist.y_velocity_initial`: The initial, uniform :math:`y`-velocity ([:math:`m s^{-1}`]).
	* :data:`namelist.brunt_vaisala_initial`: The initial, uniform Brunt-Vaisala frequency.
	* :data:`namelist.backend`: GT4Py's backend to use. Available options are:
		
			- :data:`gridtools.mode.NUMPY`: Numpy (i.e., vectorized) backend.

	* :data:`namelist.save_freq`: Save state every :data:`~namelist.freq` iterations.
	* :data:`namelist.save_dest`: Path to the location where results should be saved. 
	* :data:`namelist.tol`: Tolerance used to compare floats (see :mod:`utils`).
	* :data:`namelist.datatype`: Datatype for :class:`numpy.ndarray`. Either :data:`np.float32` \
		or :data:`np.float64`.
"""
from datetime import datetime, timedelta
import numpy as np
import os

import gridtools as gt

#
# Physical constants
#
p_ref = 1.e5
p_sl  = 1.e5
T_sl  = 288.15
beta  = 42.
Rd    = 287.
cp    = 1004.
g     = 9.81

#
# Grid settings
#
domain_x        = [-100.e3, 200.e3]
nx              = 151
domain_y        = [-100.e3, 100.e3]
ny              = 101
domain_z        = [300. + 60., 300.]
nz              = 27
z_interface     = None
topo_type       = 'schaer'
topo_time       = timedelta(seconds = 1800.)
topo_max_height = 1500.
topo_width_x    = 10.e3
topo_width_y    = 10.e3
topo_str        = '3000 / pow(1. + (x * x) / (10000. * 10000.) + (y * y) / (10000. * 10000.), 1.5)'

#
# Model settings
#
model                    = 'isentropic'
imoist					 = False
horizontal_boundary_type = 'relaxed-symmetric-xz'

#
# Numerical settings
#
scheme           = 'maccormack'
idamp	         = True
damp_type        = 'rayleigh'
damp_depth       = 13
damp_max         = .0002
idiff            = True
diff_coeff       = .001
diff_coeff_moist = .03
diff_max         = .249

#
# Simulation settings
#
dt                    = timedelta(seconds = 3.33)
initial_time          = datetime(year = 1992, month = 2, day = 20)
simulation_time       = timedelta(seconds = 40000)
x_velocity_initial    = 10.
y_velocity_initial    = 0.
brunt_vaisala_initial = .01
backend  		      = gt.mode.NUMPY
save_freq		      = 50000
save_dest		      = os.path.join(os.environ['GT4ESS_ROOT'], 'post_processing/data/LR15_maccormack_symmetric.pickle')
tol      		      = 1.e-8		
datatype 		      = np.float32		
