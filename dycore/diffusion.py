import math
import numpy as np

import gridtools as gt
from namelist import datatype

class Diffusion:
	"""
	Class applying horizontal numerical diffusion to a generic (prognostic) variable by means of a GT4Py's stencil.
	"""
	def __init__(self, grid, idamp = True, damp_depth = 10, diff_coeff = .03, diff_max = .24, backend = gt.mode.NUMPY):
		"""
		Constructor.

		Parameters
		----------
			grid : obj
				The underlying grid, as an instance of :class:`~grids.xyz_grid.XYZGrid` or one of its derived classes.
			idamp : `bool`, optional
				:data:`True` if vertical damping is enabled, :data:`False` otherwise.
				In the former case, the diffusion coefficient is monotonically increased towards the
				top of the domain, thus to mimic the effect of a wave absorber.
			damp_depth : `int`, optional
				Depth of the damping region, i.e., number of vertical layers in the damping region. Default is 10.
			diff_coeff : `float`, optional
				Value for the diffusion coefficient far from the top boundary. Default is 0.03.
			diff_max : `float`, optional
				Maximum value for the diffusion coefficient. For the sake of numerical stability, it should not 
				exceed 0.25. Default is 0.24.
			backend : `obj`, optional
				:class:`gridtools.mode` specifying the backend for the GT4Py's stencil implementing numerical 
				diffusion. Default is :class:`gridtools.mode.NUMPY`.
		"""
		# Check that the diffusion coefficient is no greater than 0.25
		assert diff_max <= .25, \
			   'For the sake of numerical stability, the diffusion coefficient should not be greater than 0.25.'

		# Store useful input arguments
		self._grid       = grid
		self._idamp      = idamp
		self._damp_depth = damp_depth
		self._diff_coeff = diff_coeff
		self._diff_max   = diff_max
		self._backend    = backend

		# Initialize pointer to stencil; this will be correctly re-directed the first time 
		# the entry-point method apply is invoked
		self._stencil = None

	def apply(self, phi):
		"""
		Apply horizontal diffusion to a prognostic field.

		Parameters
		----------
			phi : array_like
				:class:`numpy.ndarray` representing the field to diffuse.

		Return
		------
			array_like :
				:class:`numpy.ndarray` representing the diffused field.
		"""
		# The first time this method is invoked, initialize the GT4Py's stencil
		if self._stencil is None:
			self._initialize_stencil(phi)

		# Update the Numpy array carrying the stencil's input field
		self._in_phi[:,:,:] = phi[:,:,:]

		# Run the stencil's compute function
		self._stencil.compute()

		# Set the outermost lateral layer of the output field, not affected by the stencil
		self._out_phi[ 0, :, :]    = self._in_phi[ 0, :, :]
		self._out_phi[-1, :, :]    = self._in_phi[-1, :, :]
		self._out_phi[1:-1,  0, :] = self._in_phi[1:-1,  0, :]
		self._out_phi[1:-1, -1, :] = self._in_phi[1:-1, -1, :]
		
		return self._out_phi

	def _initialize_stencil(self, phi):
		"""
		Initialize the GT4Py's stencil applying horizontal diffusion.

		Parameters
		----------
			phi : array_like
				:class:`numpy.ndarray` representing the field to diffuse.
		"""
		# Shortcuts
		nx, ny, nz = self._grid.nx, self._grid.ny, self._grid.nz
		ni, nj, nk = phi.shape[0], phi.shape[1], phi.shape[2]

		# Initialize the diffusion matrix
		self._tau = self._diff_coeff * np.ones((ni, nj, nk), dtype = datatype)

		# If vertical damping is disabled: the diffusivity is monotically increased towards
		# the top of the model, so to mimic the effect of a wave absorber
		if not self._idamp:
			n = self._damp_depth
			pert = np.sin(0.5 * math.pi * (n - np.arange(0, n, dtype = datatype)) / n) ** 2
			pert = np.tile(pert[np.newaxis, np.newaxis, :], (ni, nj, 1))
			self._tau[:, :, 0:n] += (self._diff_max - self._diff_coeff) * pert

		# Allocate the Numpy array which will carry the stencil's input field
		self._in_phi = np.zeros_like(phi)

		# Allocate memory for the stencil's output field
		self._out_phi = np.zeros_like(phi)

		# Set the computational domain
		_domain = gt.domain.Rectangle((1, 1, 0), (ni - 2, nj - 2, nk - 1))

		# Instantiate the stencil
		self._stencil = gt.NGStencil(
			definitions_func = self._defs_stencil,
			inputs = {'in_phi': self._in_phi, 'tau': self._tau},
			outputs = {'out_phi': self._out_phi},
			domain = _domain, 
			mode = self._backend)

	def _defs_stencil(self, in_phi, tau):
		"""
		The GT4Py's stencil applying horizontal diffusion. A standard 5-points formula is used.

		Parameters
		----------
			in_phi : obj 
				:class:`gridtools.Equation` representing the input field to diffuse.
			tau : obj 
				:class:`gridtools.Equation` representing the diffusion coefficient.

		Return
		------
			obj :
				:class:`gridtools.Equation` representing the diffused output field.
		"""
		# Indeces
		i = gt.Index()
		j = gt.Index()
		k = gt.Index()

		# Output field
		out_phi = gt.Equation()

		# Computations
		out_phi[i, j, k] = (1. - 4. * tau[i, j, k]) * in_phi[i, j, k] + tau[i, j, k] * \
						   (in_phi[i-1, j, k] + in_phi[i+1, j, k] + in_phi[i, j-1, k] + in_phi[i, j+1, k])

		return out_phi

