"""
This module provides classes applying damping in the vertical direction.
"""
import abc
import math
import numpy as np

import gridtools as gt
from tasmania.namelist import datatype
from tasmania.utils.utils import greater_or_equal_than as ge

class VerticalDamping:
	"""
	Abstract base class whose derived classes implement different vertical damping, i.e., wave absorbing, techniques.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	def __init__(self, dims, grid, damp_depth, damp_max, backend):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the (three-dimensional) arrays on which to apply vertical damping.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		damp_depth : int
			Number of vertical layers in the damping region.
		damp_max : float
			Maximum value for the damping coefficient.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils implementing the dynamical core.
		"""
		# Store arguments
		self._dims       = dims
		self._grid       = grid
		self._damp_depth = damp_depth
		self._damp_max   = damp_max
		self._backend    = backend

		# Compute lower-bound of damping region
		self._damp_lb = self._grid.z[self._damp_depth - 1]

		# Pointer to the GT4Py stencil implementing the absorber. This is properly redirected the first time
		# the damp method is invoked.
		self._stencil = None

	@abc.abstractmethod
	def apply(self, dt, phi_now, phi_new, phi_ref):
		"""
		Apply vertical damping to a generic field :math:`\phi`. 
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the time step.
		phi_now : array_like
			:class:`numpy.ndarray` representing the field :math:`\phi` at the current time level.
		phi_new : array_like
			:class:`numpy.ndarray` representing the field :math:`\phi` at the next time level, on
			which the absorber will be applied.
		phi_ref : array_like
			:class:`numpy.ndarray` representing a reference value for :math:`\phi`.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the damped field :math:`\phi`.
		"""
	
	@staticmethod
	def factory(damp_type, dims, grid, damp_depth, damp_max, backend):
		"""
		Static method which returns an instance of the derived class implementing the damping method 
		specified by :data:`damp_type`. 

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the (three-dimensional) arrays on which to apply vertical damping.
		damp_type : str
			String specifying the damper to implement. Either:

				* 'rayleigh', for a Rayleigh damper.

		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		damp_depth : int
			Number of vertical layers in the damping region. Default is 15.
		damp_max : float
			Maximum value for the damping coefficient. Default is 0.0002.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils implementing the dynamical core.

		Return
		------
		obj :
			An instance of the derived class implementing the damping method specified by :data:`damp_type`.
		"""
		if damp_type == 'rayleigh':
			return VerticalDampingRayleigh(dims, grid, damp_depth, damp_max, backend)
		else:
			raise ValueError('Unknown damping scheme. Available options: ''rayleigh''.')


class VerticalDampingRayleigh(VerticalDamping):
	"""
	This class inherits :class:`VerticalDamping` to implement a Rayleigh absorber.
	"""
	def __init__(self, dims, grid, damp_depth, damp_max, backend):
		"""
		Constructor.

		Parameters
		----------
		dims : tuple
			Tuple of the dimension of the (three-dimensional) arrays on which to apply vertical damping.
		grid : obj
			The underlying grid, as an instance of :class:`~grids.grid_xyz.GridXYZ` or one of its derived classes.
		damp_depth : int
			Number of vertical layers in the damping region.
		damp_max : float
			Maximum value for the damping coefficient.
		backend : obj
			:class:`gridtools.mode` specifying the backend for the GT4Py stencils implementing the dynamical core.
		"""
		super().__init__(dims, grid, damp_depth, damp_max, backend)

	def apply(self, dt, phi_now, phi_new, phi_ref):
		"""
		Apply vertical damping to a generic field :math:`\phi`. 

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the time step.
		phi_now : array_like
			:class:`numpy.ndarray` representing the field :math:`\phi` at the current time level.
		phi_new : array_like
			:class:`numpy.ndarray` representing the field :math:`\phi` at the next time level, on
			which the absorber will be applied.
		phi_ref : array_like
			:class:`numpy.ndarray` representing a reference value for :math:`\phi`.

		Return
		------
		array_like :
			:class:`numpy.ndarray` representing the damped field :math:`\phi`.
		"""
		# The first time this method is invoked, initialize the stencil
		if self._stencil is None:
			self._stencil_initialize()

		# Update the attributes which will carry the stencil's input field
		self._stencil_set_inputs(dt, phi_now, phi_new, phi_ref)

		# Run the stencil's compute function
		self._stencil.compute()

		# The layers below the damping region are not affected
		self._phi_out[:, :, self._damp_depth:] = phi_new[:, :, self._damp_depth:]

		return self._phi_out

	def _stencil_initialize(self):
		"""
		Initialize the GT4Py stencil applying Rayleigh vertical damping.

		Parameters
		----------
		phi_now : array_like
			:class:`numpy.ndarray` representing the field :math:`\phi` at the current time level.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		ni, nj, nk = self._dims
		za, zt = self._damp_lb, self._grid.z_half_levels[0]

		if nk == nz:
			# Compute the damping matrix which should be used in case of a z-unstaggered field
			z = self._grid.z.values
			r = ge(z, za) * self._damp_max * (1 - np.cos(math.pi * (z - za) / (zt - za)))
			self._rmat = np.tile(r[np.newaxis, np.newaxis, :], (ni, nj, 1))
		else:
			# Compute the damping matrix which should be used in case of a z-staggered field
			z = self._grid.z_half_levels.values
			r = ge(z, za) * self._damp_max * (1 - np.cos(math.pi * (z - za) / (zt - za)))
			self._rmat = np.tile(r[np.newaxis, np.newaxis, :], (ni, nj, 1))

		# Allocate the attributes which will carry the stencil's input fields
		self._dt = gt.Global()
		self._phi_now = np.zeros((ni, nj, nk), dtype = datatype)
		self._phi_new = np.zeros((ni, nj, nk), dtype = datatype)
		self._phi_ref = np.zeros((ni, nj, nk), dtype = datatype)

		# Allocate the Numpy array which will carry the stencil's output field
		self._phi_out = np.zeros((ni, nj, nk), dtype = datatype)

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._stencil_defs,
			inputs = {'phi_now': self._phi_now, 'phi_new': self._phi_new, 'phi_ref': self._phi_ref, 'R': self._rmat},
			global_inputs = {'dt': self._dt},
			outputs = {'phi_out': self._phi_out},
			domain = gt.domain.Rectangle((0, 0, 0), (ni - 1, nj - 1, self._damp_depth - 1)), 
			mode = self._backend)

	def _stencil_set_inputs(self, dt, phi_now, phi_new, phi_ref):
		"""
		Update the attributes which stores the stencil's input fields.

		Parameters
		----------
		dt : obj
			:class:`datetime.timedelta` representing the time step.
		phi_now : array_like
			:class:`numpy.ndarray` representing the field :math:`\phi` at the current time level.
		phi_new : array_like
			:class:`numpy.ndarray` representing the field :math:`\phi` at the next time level, on
			which the absorber will be applied.
		phi_ref : array_like
			:class:`numpy.ndarray` representing a reference value for :math:`\phi`.
		"""
		self._dt.value = 1.e-6 * dt.microseconds if dt.seconds == 0. else dt.seconds
		self._phi_now[:,:,:] = phi_now[:,:,:]
		self._phi_new[:,:,:] = phi_new[:,:,:]
		self._phi_ref[:,:,:] = phi_ref[:,:,:]

	def _stencil_defs(self, dt, phi_now, phi_new, phi_ref, R):
		"""
		The GT4Py stencil applying Rayleigh vertical damping.

		Parameters
		----------
		dt : obj
			:class:`gridtools.Global` representing the time step.
		phi_now : ibj
			:class:`gridtools.Equation` representing the field :math:`\phi` at the current time level.
		phi_new : obj
			:class:`gridtools.Equation` representing the field :math:`\phi` at the next time level, on
			which the absorber will be applied.
		phi_ref : obj
			:class:`gridtools.Equation` representing a reference value for :math:`\phi`.
		R : obj
			:class:`gridtools.Equation` representing the damping coefficient.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the damped field :math:`\phi`.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		phi_out = gt.Equation()

		# Computations
		phi_out[i, j, k] = phi_new[i, j, k] - dt * R[i, j, k] * (phi_now[i, j, k] - phi_ref[i, j, k])

		return phi_out
