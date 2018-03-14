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
	* :data:`namelist.topo_str`: When :data:`~namelist.topo_type` is 'user_defined', terrain profile expression in the \
		independent variables :math:`x` and :math:`y`. Must be fully C++-compliant.
	* :data:`namelist.topo_kwargs`: Dictionary storing :data:`~namelist.topo_max_height`, :data:`~namelist.topo_width_x`, \
		:data:`~namelist.topo_width_y` and :data:`~namelist.topo_str`.

Model settings:
	* :data:`namelist.model_name`: Name of the model to implement. Available options are:

			- 'isentropic', for the isentropic model.

	* :data:`namelist.moist_on`: :data:`True` if water constituents should be taken into account, :data:`False` otherwise.
	* :data:`namelist.horizontal_boundary_type`: Horizontal boundary conditions. Available options are:

			- 'periodic', for periodic boundary conditions;
			- 'relaxed', for relaxed boundary conditions. 

Numerical settings:
	* :data:`namelist.time_scheme`: Time integration scheme to implement. Available options are:
			
			- 'forward_euler', for the forward Euler scheme;
			- 'centered', for a centered time-integration scheme.

	* :data:`namelist.flux_scheme`: Numerical flux to use. Available options are:

			- 'upwind', for the upwind scheme;
			- 'centered', for a second-order centered scheme;
			- 'maccormack', for the MacCormack scheme.

	* :data:`namelist.damp_on`: :data:`True` if (explicit) vertical damping should be applied, :data:`False` otherwise. \
		Note that when vertical damping is switched off, the numerical diffusion is monotonically increased towards \
		the top of the model, so to act as a diffusive wave absorber.
	* :data:`namelist.damp_type`: Type of vertical damping to apply. Available options are:
		
			- 'rayleigh', for Rayleigh vertical damping.

	* :data:`namelist.damp_depth`: Number of levels (either main levels or half levels) in the absorbing region.
	* :data:`namelist.damp_max`: Maximum value which should be assumed by the damping coefficient.
	* :data:`namelist.smooth_on`: :data:`True` to enable numerical horizontal smoothing, :data:`False` otherwise.
	* :data:`namelist.smooth_type`: Type of smoothing technique to implement. Available options are:

			- 'first_order', for first-order smoothing;
			- 'second_order', for second-order smoothing.

	* :data:`namelist.smooth_depth`: Number of levels (either main levels or half levels) in the smoothing absorbing region.
	* :data:`namelist.smooth_coeff`: The smoothing coefficient.
	* :data:`namelist.smooth_coeff_max`: Maximum value for the smoothing coefficient when smoothing vertical damping is enabled.
	* :data:`namelist.smooth_moist_on`: :data:`True` to enable numerical horizontal smoothing on the moisture constituents, \
		:data:`False` otherwise.
	* :data:`namelist.smooth_moist_type`: Type of smoothing technique to apply on the moisture constituents. Available options are:

			- 'first_order', for first-order smoothing;
			- 'second_order', for second-order smoothing.

	* :data:`namelist.smooth_moist_depth`: Number of levels (either main levels or half levels) in the smoothing absorbing region \
		for the moisture constituents.
	* :data:`namelist.smooth_coeff_moist`: The smoothing coefficient for the moisture components.
	* :data:`namelist.smooth_coeff_moist_max`: Maximum value for the smoothing coefficient for the moisture components when \
		smoothing vertical damping is enabled.

Simulation settings:
	* :data:`namelist.dt`: :class:`datetime.timedelta` object representing the timestep.
	* :data:`namelist.initial_time`: :class:`datetime.datetime` representing the initial simulation time.
	* :data:`namelist.simulation_time`: :class:`datetime.timedelta` object representing the simulation time.
	* :data:`namelist.initial_state_type`: Integer identifying the initial state. See the documentation for the method \
		:meth:`~dycore.dycore_isentropic.DycoreIsentropic.get_initial_state()` of \
		:class:`~dycore.dycore_isentropic.DycoreIsentropic`.
	* :data:`namelist.x_velocity_initial`: The initial, uniform :math:`x`-velocity ([:math:`m s^{-1}`]).
	* :data:`namelist.y_velocity_initial`: The initial, uniform :math:`y`-velocity ([:math:`m s^{-1}`]).
	* :data:`namelist.brunt_vaisala_initial`: The initial, uniform Brunt-Vaisala frequency.
	* :data:`namelist.temperature_initial`: The initial, uniform temperature ([:math:`K`]).
	* :data:`namelist.initial_state_kwargs`: Dictionary storing :data:`~namelist.x_velocity_initial`, \
		:data:`~namelist.y_velocity_initial`, :data:`~namelist.brunt_vaisala_initial` and :data:`~namelist.temperature_initial`.
	* :data:`namelist.backend`: GT4Py's backend to use. Available options are:
		
			- :data:`gridtools.mode.NUMPY`: Numpy (i.e., vectorized) backend.

	* :data:`namelist.save_iterations`: List of the iterations at which the state should be saved.
	* :data:`namelist.save_dest`: Path to the location where results should be saved. 
	* :data:`namelist.tol`: Tolerance used to compare floats (see :mod:`utils`).
	* :data:`namelist.datatype`: Datatype for :class:`numpy.ndarray`. Either :data:`np.float32` or :data:`np.float64`.
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
domain_x        = [0, 500.e3]
nx              = 51
domain_y        = [-250.e3, 250.e3]
ny              = 51
domain_z        = [300. + 100., 300.]
nz              = 51
z_interface     = None
topo_type       = 'gaussian'
topo_time       = timedelta(seconds = 1800.)
topo_kwargs     = {
				   'topo_max_height': 1000.,
				   'topo_width_x': 50.e3,
				   'topo_width_y': 50.e3,
				   'topo_str': '1. * 10000. * 10000. / (x * x + 10000. * 10000.)',
				  }

#
# Model settings
#
model                    = 'isentropic' 
moist_on					 = False
horizontal_boundary_type = 'relaxed'

#
# Numerical settings
#
time_scheme             = 'forward_euler'
flux_scheme             = 'upwind'
damp_on	                = True
damp_type               = 'rayleigh'
damp_depth              = 15
damp_max                = .0002
smooth_on                 = True
smooth_type             = 'first_order'
smooth_damp_depth       = 0
smooth_coeff            = .03
smooth_coeff_max        = .25
smooth_moist_on		    = False
smooth_moist_type       = 'first_order'
smooth_moist_damp_depth = 30
smooth_moist_coeff      = .05
smooth_moist_coeff_max  = .25

#
# Simulation settings
#
dt                    = timedelta(seconds = 24)
initial_time          = datetime(year = 1992, month = 2, day = 20)
simulation_time       = timedelta(hours = 12)
initial_state_type    = 0
initial_state_kwargs  = {
						 'x_velocity_initial': 15.,
						 'y_velocity_initial': 0.,
						 'brunt_vaisala_initial': .01,
						 'temperature': 250.,
						}
backend  		      = gt.mode.NUMPY
save_iterations		  = []
save_dest		      = None #os.path.join(os.environ['TASMANIA_ROOT'], 'data/verification_moist_leapfrog_newinterface.pickle')
tol      		      = 1.e-8		
datatype 		      = np.float32
