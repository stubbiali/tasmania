""" 
Configuration and global variables used throughout the package. 

Physical constants:
	* :data:`~tasmania.namelist.p_ref`: Reference pressure ([:math:`Pa`]).
	* :data:`~tasmania.namelist.p_sl`: Reference pressure at sea level ([:math:`Pa`]).
	* :data:`~tasmania.namelist.T_sl`: Reference temperature at sea level ([:math:`K`]).
	* :data:`~tasmania.namelist.beta`: Rate of increase in reference temperature with the logarithm \
		of reference pressure ([:math:`K ~ Pa^{-1}`]).
	* :data:`~tasmania.namelist.Rd`: Gas constant for dry air ([:math:`J ~ K^{-1} ~ Kg^{-1}`]).
	* :data:`~tasmania.namelist.Rv`: Gas constant for water vapor ([:math:`J ~ K^{-1} ~ Kg^{-1}`]).
	* :data:`~tasmania.namelist.cp`: Specific heat of dry air at constant pressure ([:math:`J ~ K^{-1} ~ Kg^{-1}`]).
	* :data:`~tasmania.namelist.g`: Mean gravitational acceleration ([:math:`m ~ s^{-2}`]). 
	* :data:`~tasmania.namelist.L`: Specific latent heat of condensation of water ([:math:`J ~ kg^{-1}`]).
	* :data:`~tasmania.namelist.rho_water`: Water density ([:math:`kg ~ m^{-3}`]).

Grid settings:
	* :data:`~tasmania.namelist.domain_x`: Tuple storing the boundaries of the domain in the \
		:math:`x`-direction in the form (:math:`x_{west}`, :math:`x_{east}`).
	* :data:`~tasmania.namelist.nx`: Number of grid points in the :math:`x`-direction.
	* :data:`~tasmania.namelist.domain_y`: Tuple storing the boundaries of the domain in the \
		:math:`y`-direction in the form (:math:`y_{south}`, :math:`y_{north}`).
	* :data:`~tasmania.namelist.ny`: Number of grid points in the :math:`y`-direction.
	* :data:`~tasmania.namelist.domain_z`: Tuple storing the boundaries of the domain in the \
		:math:`z`-direction in the form (:math:`z_{top}`, :math:`z_{bottom}`).
	* :data:`~tasmania.namelist.nz`: Number of grid points in the :math:`z`-direction.
	* :data:`~tasmania.namelist.z_interface`: For a hybrid coordinate system, interface level \
		at which terrain-following :math:`z`-coordinate lines get back to horizontal lines.
	* :data:`~tasmania.namelist.topo_type`: Topography type. Available options are:

			- 'flat_terrain';
			- 'gaussian';
			- 'schaer';
			- 'user_defined'.

	* :data:`~tasmania.namelist.topo_time`: :class:`datetime.timedelta` object representing the elapsed \
		simulation time after which the topography should stop increasing.
	* :data:`~tasmania.namelist.topo_max_height`: When :data:`~tasmania.namelist.topo_type` is 'gaussian', \
		maximum mountain height ([:math:`m`]).
	* :data:`~tasmania.namelist.topo_width_x`: When :data:`~tasmania.namelist.topo_type` is 'gaussian', mountain \
		half-width in :math:`x`-direction ([:math:`m`]).
	* :data:`~tasmania.namelist.topo_width_y`: When :data:`~tasmania.namelist.topo_type` is 'gaussian', mountain \
		half-width in :math:`y`-direction ([:math:`m`]).
	* :data:`~tasmania.namelist.topo_str`: When :data:`~tasmania.namelist.topo_type` is 'user_defined', terrain profile \
		expression in the independent variables :math:`x` and :math:`y`. Must be fully C++-compliant.
	* :data:`~tasmania.namelist.topo_smooth`: :obj:`True` to smooth the topography out, :obj:`False` otherwise.
	* :data:`~tasmania.namelist.topo_kwargs`: Dictionary storing :data:`~tasmania.namelist.topo_max_height`, \
		:data:`~tasmania.namelist.topo_width_x`, :data:`~tasmania.namelist.topo_width_y`, :data:`~tasmania.namelist.topo_str`, \
		and :data:`~tasmania.namelist.topo_smooth`.

Model settings:
	* :data:`~tasmania.namelist.model_name`: Name of the model to implement. Available options are:

			- 'isentropic_conservative', for the isentropic model based on the conservative form \
				of the governing equations;
			- 'isentropic_nonconservative', for the isentropic model based on the nonconservative form \
				of the governing equations.

	* :data:`~tasmania.namelist.moist_on`: :data:`True` if water constituents should be taken into account, \
		:data:`False` otherwise.
	* :data:`~tasmania.namelist.horizontal_boundary_type`: Horizontal boundary conditions. Available options are:

			- 'periodic', for periodic boundary conditions;
			- 'relaxed', for relaxed boundary conditions. 

Numerical settings:
	* :data:`~tasmania.namelist.time_scheme`: Time integration scheme to implement. Available options are:
			
			- 'forward_euler', for the forward Euler scheme ('isentropic_conservative');
			- 'centered', for a centered time-integration scheme ('isentropic_conservative', 'isentropic_nonconservative').

	* :data:`~tasmania.namelist.flux_scheme`: Numerical flux to use. Available options are:

			- 'upwind', for the upwind scheme ('isentropic_conservative');
			- 'centered', for a second-order centered scheme ('isentropic_conservative', 'isentropic_nonconservative');
			- 'maccormack', for the MacCormack scheme ('isentropic_conservative').

	* :data:`~tasmania.namelist.damp_on`: :data:`True` if (explicit) vertical damping should be applied, 
		:data:`False` otherwise. Note that when vertical damping is switched off, the numerical diffusion \
		coefficient is monotonically increased towards the top of the model, so to act as a diffusive wave absorber.
	* :data:`~tasmania.namelist.damp_type`: Type of vertical damping to apply. Available options are:
		
			- 'rayleigh', for Rayleigh vertical damping.

	* :data:`~tasmania.namelist.damp_depth`: Number of levels (either main levels or half levels) in the absorbing region.
	* :data:`~tasmania.namelist.damp_max`: Maximum value which should be assumed by the damping coefficient.
	* :data:`~tasmania.namelist.smooth_on`: :data:`True` to enable numerical horizontal smoothing, :data:`False` otherwise.
	* :data:`~tasmania.namelist.smooth_type`: Type of smoothing technique to implement. Available options are:

			- 'first_order', for first-order smoothing;
			- 'second_order', for second-order smoothing.

	* :data:`~tasmania.namelist.smooth_depth`: Number of levels (either main levels or half levels) \
		in the smoothing absorbing region.
	* :data:`~tasmania.namelist.smooth_coeff`: The smoothing coefficient.
	* :data:`~tasmania.namelist.smooth_coeff_max`: Maximum value for the smoothing coefficient when \
		smoothing vertical damping is enabled.
	* :data:`~tasmania.namelist.smooth_moist_on`: :data:`True` to enable numerical horizontal smoothing \
		on the moisture constituents, :data:`False` otherwise.
	* :data:`~tasmania.namelist.smooth_moist_type`: Type of smoothing technique to apply on the moisture \
		constituents. Available options are:

			- 'first_order', for first-order smoothing;
			- 'second_order', for second-order smoothing.

	* :data:`~tasmania.namelist.smooth_moist_depth`: Number of levels (either main levels or half levels) \
		in the smoothing absorbing region for the moisture constituents.
	* :data:`~tasmania.namelist.smooth_coeff_moist`: The smoothing coefficient for the moisture components.
	* :data:`~tasmania.namelist.smooth_coeff_moist_max`: Maximum value for the smoothing coefficient for \
		the moisture components when smoothing vertical damping is enabled.

Microphysics settings:
	* :data:`~tasmania.namelist.physics_dynamics_coupling_on`: :obj:`True` to couple physics with dynamics, \
		i.e., to take the change over time in potential temperature into account, :obj:`False` otherwise.
	* :data:`~tasmania.namelist.sedimentation_on`: :obj:`True` to account for rain sedimentation, :obj:`False` otherwise.
	* :data:`~tasmania.namelist.sedimentation_flux_type`: String specifying the method used to compute the numerical \
		sedimentation flux. Available options are:

			- 'first_order_upwind', for the first-order upwind scheme;
			- 'second_order_upwind', for the second-order upwind scheme.

	* :data:`~tasmania.namelist.sedimentation_substeps`: If rain sedimentation is switched on, number of sub-timesteps \
	  	to perform in order to integrate the sedimentation flux. 
	* :data:`~tasmania.namelist.rain_evaporation_on`: :obj:`True` to account for rain evaporation, :obj:`False` otherwise.
	* :data:`~tasmania.namelist.slow_tendency_microphysics_on`: :obj:`True` to include a parameterization scheme \
		providing slow-varying cloud microphysical tendencies, :obj:`False` otherwise.
	* :data:`~tasmania.namelist.slow_tendency_microphysics_type`: The name of the parameterization scheme in charge of \
		providing slow-varying cloud microphysical tendencies. Available options are:

			- 'kessler_wrf', for the WRF version of the Kessler scheme.

	* :data:`~tasmania.namelist.slow_tendency_microphysics_kwargs`: Keyword arguments for the parameterization scheme \
		in charge of providing slow-varying cloud microphysical tendencies. Please see \
		:mod:`~tasmania.parameterizations.slow_tendencies` for many more details.
	* :data:`~tasmania.namelist.fast_tendency_microphysics_on`: :obj:`True` to include a parameterization scheme \
		providing fast-varying cloud microphysical tendencies, :obj:`False` otherwise.
	* :data:`~tasmania.namelist.fast_tendency_microphysics_type`: The name of the parameterization scheme in charge of \
		providing fast-varying cloud microphysical tendencies. Available options are:

			- 'kessler_wrf', for the WRF version of the Kessler scheme.

	* :data:`~tasmania.namelist.fast_tendency_microphysics_kwargs`: Keyword arguments for the parameterization scheme \
		in charge of providing fast-varying cloud microphysical tendencies. Please see \
		:mod:`~tasmania.parameterizations.fast_tendencies` for many more details.
	* :data:`~tasmania.namelist.adjustment_microphysics_on`: :obj:`True` to include a parameterization scheme \
		performing cloud microphysical adjustments, :obj:`False` otherwise.
	* :data:`~tasmania.namelist.adjustment_microphysics_type`: The name of the parameterization scheme in charge \
		of performing cloud microphysical adjustments. Available options are:

			- 'kessler_wrf', for the WRF version of the Kessler scheme;
			- 'kessler_wrf_saturation', for the WRF version of the Kessler scheme, \
				carrying out only the saturation adjustment.

	* :data:`~tasmania.namelist.adjustment_microphysics_kwargs`: Keyword arguments for the parameterization scheme \
		in charge of performing cloud microphysical adjustments. Please see \
		:mod:`~tasmania.parameterizations.adjustments` for many more details.

Simulation settings:
	* :data:`~tasmania.namelist.dt`: :class:`datetime.timedelta` object representing the timestep.
	* :data:`~tasmania.namelist.initial_time`: :class:`datetime.datetime` representing the initial simulation time.
	* :data:`~tasmania.namelist.simulation_time`: :class:`datetime.timedelta` object representing the simulation time.
	* :data:`~tasmania.namelist.initial_state_type`: Integer identifying the initial state. See the documentation for the method \
		:meth:`~tasmania.dycore.dycore_isentropic.DycoreIsentropic.get_initial_state()` of \
		:class:`~tasmania.dycore.dycore_isentropic.DycoreIsentropic`.
	* :data:`~tasmania.namelist.x_velocity_initial`: The initial, uniform :math:`x`-velocity ([:math:`m s^{-1}`]).
	* :data:`~tasmania.namelist.y_velocity_initial`: The initial, uniform :math:`y`-velocity ([:math:`m s^{-1}`]).
	* :data:`~tasmania.namelist.brunt_vaisala_initial`: The initial, uniform Brunt-Vaisala frequency.
	* :data:`~tasmania.namelist.temperature_initial`: The initial, uniform temperature ([:math:`K`]).
	* :data:`~tasmania.namelist.initial_state_kwargs`: Dictionary storing :data:`~tasmania.namelist.x_velocity_initial`, \
		:data:`~tasmania.namelist.y_velocity_initial`, :data:`~tasmania.namelist.brunt_vaisala_initial`, \
		and :data:`~tasmania.namelist.temperature_initial`.
	* :data:`~tasmania.namelist.backend`: GT4Py backend to use. Available options are:
		
			- :data:`gridtools.mode.NUMPY`: Numpy (i.e., vectorized) backend.

	* :data:`~tasmania.namelist.save_iterations`: List of the iterations at which the state should be saved.
	* :data:`~tasmania.namelist.save_dest`: Path to the location where results should be saved. 
	* :data:`~tasmania.namelist.tol`: Tolerance used to compare floats (see :mod:`utils`).
	* :data:`~tasmania.namelist.datatype`: Datatype for :class:`numpy.ndarray`. Either :data:`np.float32` or :data:`np.float64`.
"""
from datetime import datetime, timedelta
import numpy as np
import os

import gridtools as gt

#
# Physical constants
#
p_ref     = 1.e5
p_sl      = 1.e5
T_sl      = 288.15
beta      = 42.
Rd        = 287.05
Rv	      = 461.52
cp        = 1004.
g         = 9.81
L	      = 2.5e6
rho_water = 1000.
from sympl import DataArray
pippo = DataArray(10., attrs={'units': '1'})

#
# Grid settings
#
domain_x        = [0, 500.e3]
nx              = 51
domain_y        = [-250.e3, 250.e3]
ny              = 51
domain_z        = [300. + 100., 300.]
nz              = 50
z_interface     = None
topo_type       = 'gaussian'
topo_time       = timedelta(seconds = 1800.)
topo_kwargs     = {
				   'topo_max_height': 1000.,
				   'topo_width_x'   : 50.e3,
				   'topo_width_y'   : 50.e3,
				   'topo_str'       : '1. * 10000. * 10000. / (x * x + 10000. * 10000.)',
				   'topo_smooth'    : False,
				  }

#
# Model settings
#
model                    = 'isentropic_conservative' 
moist_on				 = False
horizontal_boundary_type = 'relaxed'

#
# Numerical settings
#
time_scheme             = 'forward_euler'
flux_scheme             = 'maccormack'
damp_on	                = False
damp_type               = 'rayleigh'
damp_depth              = 15
damp_max                = .0002
smooth_on               = True
smooth_type             = 'first_order'
smooth_damp_depth       = 0
smooth_coeff            = .03
smooth_coeff_max        = .03
smooth_moist_on		    = False
smooth_moist_type       = 'first_order'
smooth_moist_damp_depth = 30
smooth_moist_coeff      = .05
smooth_moist_coeff_max  = .25

#
# Microphysics settings
#
physics_dynamics_coupling_on   		= False
sedimentation_on	           		= True
sedimentation_flux_type		   		= 'second_order_upwind'
sedimentation_substeps		   		= 2
rain_evaporation_on			   		= True
slow_tendency_microphysics_on       = False
slow_tendency_microphysics_type     = 'kessler_wrf'
slow_tendency_microphysics_kwargs	= {
								  	   'a' : .0001,
								  	   'k1': .001,
								  	   'k2': 2.2,
								 	  }
fast_tendency_microphysics_on       = False
fast_tendency_microphysics_type     = ''
fast_tendency_microphysics_kwargs	= {}
adjustment_microphysics_on     		= False
adjustment_microphysics_type   		= 'kessler_wrf_saturation'
adjustment_microphysics_kwargs 		= {
								  	   'a' : .0001,
								  	   'k1': .001,
								  	   'k2': 2.2,
								 	  }

#
# Simulation settings
#
dt                    = timedelta(seconds = 24)
initial_time          = datetime(year = 1992, month = 2, day = 20)
simulation_time       = timedelta(hours = 12)
initial_state_type    = 0
initial_state_kwargs  = {
						 'x_velocity_initial'   : 15.,
						 'y_velocity_initial'   : 0.,
						 'brunt_vaisala_initial': .01,
						 'temperature'          : 250.,
						}
backend  		      = gt.mode.NUMPY
save_iterations		  = np.arange(30, 1801, 30)
save_dest		      = os.path.join(os.environ['TASMANIA_ROOT'], 'data/old_datasets/verification_1_maccormack.pickle')
tol      		      = 1.e-8		
datatype 		      = np.float32
