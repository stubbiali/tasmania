"""
This module contains:
    _Periodic(HorizontalBoundary)
    _Periodic{XZ, YZ}(HorizontalBoundary)
    _Relaxed(HorizontalBoundary)
    _Relaxed{XZ, YZ}(HorizontalBoundary)
    _RelaxedSymmetric{XZ, YZ}(_Relaxed)
"""
import copy
import numpy as np
from sympl import DataArray

from tasmania.dynamics.horizontal_boundary import HorizontalBoundary

try:
	from tasmania.namelist import datatype
except ImportError:
	datatype = np.float32


class _Periodic(HorizontalBoundary):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
	to implement horizontally periodic boundary conditions.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		nb : int
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	@property
	def mi(self):
		return self._grid.nx + 2*self.nb

	@property
	def mj(self):
		return self._grid.ny + 2*self.nb

	def from_physical_to_computational_domain(self, phi):
		"""
		Periodically extend the field :obj:`phi` with :attr:`nb` extra layers.
		"""
		# Shortcuts
		nx, ny, nz, nb = self._grid.nx, self._grid.ny, self._grid.nz, self.nb
		ni, nj, nk = phi.shape

		# Decorate the input field
		phi_ = np.zeros((ni + 2 * nb, nj + 2 * nb, nk), dtype=phi.dtype)
		phi_[nb:-nb, nb:-nb, :] = phi

		# Extend in the x-direction
		phi_[ :nb, nb:-nb, :] = phi_[(nx - 1):(nx - 1 + nb)    , nb:-nb, :]
		phi_[-nb:, nb:-nb, :] = phi_[(- nx + 1 - nb):(- nx + 1), nb:-nb, :]

		# Extend in the y-direction
		phi_[:,  :nb, :] = phi_[:, (ny - 1):(ny - 1 + nb)    , :]
		phi_[:, -nb:, :] = phi_[:, (- ny + 1 - nb):(- ny + 1), :]

		return phi_

	def from_computational_to_physical_domain(self, phi_, out_dims=None, change_sign=False):
		"""
		Shrink the field :obj:`phi_` by removing the :attr:`nb` outermost layers.
		"""
		# Shortcuts
		nx, ny, nz, nb = self._grid.nx, self._grid.ny, self._grid.nz, self.nb
		ni, nj, nk = phi_.shape

		# Check
		assert ((ni == nx + 2*nb) or (ni == nx + 1 + 2*nb)) and \
			   ((nj == ny + 2*nb) or (nj == ny + 1 + 2*nb)), \
			   'The input field does not have the dimensions one would expect. ' \
			   'Hint: has the field been previously extended?'

		# Shrink
		return phi_[nb:-nb, nb:-nb, :]

	def enforce(self, phi_new, phi_now=None):
		"""
		Apply horizontally periodic boundary conditions on :attr:`phi_new`.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation,
		yet it is retained as optional argument for compliancy with the
		class hierarchy interface.
		"""
		# Shortcuts
		nx, ny = self._grid.nx, self._grid.ny
		ni, nj, _ = phi_new.shape

		# Make the field periodic in the x-direction
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically
		# periodic once the computations are over, due to the periodic
		# extension performed before the computations started
		if ni == nx + 1:
			phi_new[ 0, :, :] = phi_new[-2, :, :]
			phi_new[-1, :, :] = phi_new[ 1, :, :]

		# Make the field periodic in the y-direction
		# Remark: This is applied only if the field is y-staggered
		# If the field is unstaggered, it should be automatically
		# periodic once the computations are over, due to the periodic
		# extension performed before the computations started
		if nj == ny + 1:
			phi_new[:,  0, :] = phi_new[:, -2, :]
			phi_new[:, -1, :] = phi_new[:,  1, :]

	def set_outermost_layers_x(self, phi_new, phi_now=None):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`x`-direction so to satisfy the periodic
		boundary conditions.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation,
		yet it is retained as optional argument for compliancy with the
		class hierarchy interface.
		"""
		# Shortcuts
		nx, ni = self._grid.nx, phi_new.shape[0]

		# Set the outermost layers
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically
		# periodic once the computations are over, due to the periodic
		# extension performed before the computations started
		if ni == nx + 1:
			phi_new[ 0, :, :] = phi_new[nx-1, :, :]
			phi_new[-1, :, :] = phi_new[ -nx, :, :]

	def set_outermost_layers_y(self, phi_new, phi_now = None):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`y`-direction so to satisfy the periodic
		boundary conditions.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation,
		yet it is retained as optional argument for compliancy with the
		class hierarchy interface.
		"""
		# Shortcuts
		ny, nj = self._grid.ny, phi_new.shape[1]

		# Set the outermost layers
		# Remark: This is applied only if the field is y-staggered
		# If the field is unstaggered, it should be automatically
		# periodic once the computations are over, due to the periodic
		# extension performed before the computations started
		if nj == ny + 1:
			phi_new[:,  0, :] = phi_new[:, ny-1, :]
			phi_new[:, -1, :] = phi_new[:,  -ny, :]

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self._grid.x, self._grid.nx, self._grid.dx.values.item()
		y, ny, dy = self._grid.y, self._grid.ny, self._grid.dy.values.item()
		z, nz, dz = self._grid.z, self._grid.nz, self._grid.dz.values.item()
		z_hl         = self._grid.z_on_interface_levels.values
		_z_interface = self._grid.z_interface
		_topo_type   = self._grid.topography.topo_type
		_topo_time   = self._grid.topography.topo_time
		_topo_kwargs = self._grid.topography.topo_kwargs

		# Determine the computational x-axis
		_domain_x = DataArray([x.values[0] - nb*dx, x.values[-1] + nb*dx],
							  dims=x.dims, attrs=x.attrs)
		_nx       = nx + 2*nb

		# Determine the computational y-axis
		_domain_y = DataArray([y.values[0] - nb*dy, y.values[-1] + nb*dy],
							  dims=y.dims, attrs=y.attrs)
		_ny       = ny + 2*nb

		# Determine the computational z-axis
		_domain_z = DataArray([z_hl[0], z_hl[-1]],
							  dims=self._grid.z.dims, attrs=z.attrs)
		_nz       = nz

		# Instantiate the computational grid
		return type(self._grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							    z_interface=_z_interface,
							    topo_type=_topo_type, topo_time=_topo_time,
								topo_kwargs=_topo_kwargs)


class _PeriodicXZ(HorizontalBoundary):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
	to implement horizontally periodic boundary conditions for fields
	defined on a computational domain consisting of only one grid
	point in the :math:`y`-direction.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		nb : int
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	@property
	def mi(self):
		return self._grid.nx + 2*self.nb

	@property
	def mj(self):
		return self._grid.ny + 2*self.nb

	def from_physical_to_computational_domain(self, phi):
		"""
		Periodically extend the field :obj:`phi` with :attr:`nb`
		extra layers in the :math:`x`-direction, then add :data:`nb`
		ghost layers in the :math:`y`-direction.
		"""
		# Shortcuts
		nx, ny, nz, nb = self._grid.nx, self._grid.ny, self._grid.nz, self.nb
		ni, nj, nk = phi.shape

		# Decorate the input field
		phi_ = np.zeros((ni + 2*nb, nj, nk), dtype=phi.dtype)
		phi_[nb:-nb, :, :] = phi

		# Extend in the x-direction
		phi_[ :nb, :, :] = phi_[(nx - 1):(nx - 1 + nb)    , :, :]
		phi_[-nb:, :, :] = phi_[(- nx + 1 - nb):(- nx + 1), :, :]

		# Repeat in the y-direction
		return np.concatenate((np.repeat(phi_[:, 0:1, :], nb, axis=1),
							   phi_,
							   np.repeat(phi_[:, -1:, :], nb, axis=1)), axis=1)

	def from_computational_to_physical_domain(self, phi_, out_dims=None, change_sign=False):
		"""
		Return the central :math:`xz`-slices of the input field :data:`phi_`,
		removing the :attr:`nb` outermost layers in :math:`y`-direction.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required
		by the implementation, yet they are retained as optional arguments for
		compliancy with the class hierarchy interface.
		"""
		nb = self.nb
		return phi_[nb:-nb, nb:-nb, :]

	def enforce(self, phi_new, phi_now=None):
		"""
		Apply horizontally periodic boundary conditions on :attr:`phi_new`.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation,
		yet it is retained as optional argument for compliancy with the
		class hierarchy interface.
		"""
		# Shortcuts
		nx, ni = self._grid.nx, phi_new.shape[0]

		# Make the field periodic in the x-direction
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically
		# periodic once the computations are over, due to the periodic
		# extension performed before the computations started
		if ni == nx + 1:
			phi_new[ 0, :, :] = phi_new[-2, :, :]
			phi_new[-1, :, :] = phi_new[ 1, :, :]

	def set_outermost_layers_x(self, phi_new, phi_now=None):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`x`-direction so to satisfy the periodic boundary conditions.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation,
		yet it is retained as optional argument for compliancy with the
		class hierarchy interface.
		"""
		# Shortcuts
		nx, ni = self._grid.nx, phi_new.shape[0]

		# Set the outermost layers
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically
		# periodic once the computations are over, due to the periodic
		# extension performed before the computations started
		if ni == nx + 1:
			phi_new[ 0, :, :] = phi_new[nx-1, :, :]
			phi_new[-1, :, :] = phi_new[ -nx, :, :]

	def set_outermost_layers_y(self, phi_new, phi_now=None):
		pass

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self._grid.x, self._grid.nx, self._grid.dx
		y, ny, dy = self._grid.y, self._grid.ny, self._grid.dy
		z, nz, dz = self._grid.z, self._grid.nz, self._grid.dz
		z_hl         = self._grid.z_on_interface_levels.values
		_z_interface = self._grid.z_interface
		_topo_type   = self._grid.topography.topo_type
		_topo_time   = self._grid.topography.topo_time
		_topo_kwargs = self._grid.topography.topo_kwargs

		# Determine the computational x-axis
		_domain_x = DataArray([x.values[0] - nb*dx, x.values[-1] + nb*dx],
							  dims=x.dims, attrs=x.attrs)
		_nx       = nx + 2*nb

		# Determine the computational y-axis
		_domain_y = DataArray([y.values[0] - nb*dy, y.values[-1] + nb*dy],
							  dims=y.dims, attrs=y.attrs)
		_ny       = ny + 2*nb

		# Determine the computational z-axis
		_domain_z = DataArray([z_hl[0], z_hl[-1]],
							  dims=z.dims, attrs=z.attrs)
		_nz       = nz

		# Instantiate the computational grid
		return type(self._grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							    z_interface=_z_interface,
							    topo_type=_topo_type, topo_time=_topo_time,
								topo_kwargs=_topo_kwargs)


class _PeriodicYZ(HorizontalBoundary):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
	to implement horizontally periodic boundary conditions for fields
	defined on a computational domain consisting of only one grid
	point in the :math:`x`-direction.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		nb : int
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	@property
	def mi(self):
		return self._grid.nx + 2*self.nb

	@property
	def mj(self):
		return self._grid.ny + 2*self.nb

	def from_physical_to_computational_domain(self, phi):
		"""
		Periodically extend the field :obj:`phi` with :attr:`nb`
		extra layers in the :math:`y`-direction, then add :data:`nb`
		ghost layers in the :math:`x`-direction.
		"""
		# Shortcuts
		nx, ny, nb = self._grid.nx, self._grid.ny, self.nb
		ni, nj, nk = phi.shape

		# Decorate the input field
		phi_ = np.zeros((ni, nj + 2*nb, nk), dtype=phi.dtype)
		phi_[:, nb:-nb, :] = phi

		# Extend in the y-direction
		phi_[:,  :nb, :] = phi_[:, (ny - 1):(ny - 1 + nb)    , :]
		phi_[:, -nb:, :] = phi_[:, (- ny + 1 - nb):(- ny + 1), :]

		# Repeat in the x-direction
		return np.concatenate((np.repeat(phi_[0:1, :, :], nb, axis=0),
							   phi_,
							   np.repeat(phi_[-1:, :, :], nb, axis=0)), axis=0)

	def from_computational_to_physical_domain(self, phi_, out_dims=None, change_sign=False):
		"""
		Return the central :math:`yz`-slices of the input field :data:`phi_`,
		removing the :attr:`nb` outermost layers in :math:`x`-direction.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required
		by the implementation, yet they are retained as optional arguments for
		compliancy with the class hierarchy interface.
		"""
		nb = self.nb
		return phi_[nb:-nb, nb:-nb, :]

	def enforce(self, phi_new, phi_now=None):
		"""
		Apply horizontally periodic boundary conditions on :attr:`phi_new`.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation,
		yet it is retained as optional argument for compliancy with the
		class hierarchy interface.
		"""
		# Shortcuts
		ny, nj = self._grid.ny, phi_new.shape[1]

		# Make the field periodic in the x-direction
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically
		# periodic once the computations are over, due to the periodic
		# extension performed before the computations started
		if nj == ny + 1:
			phi_new[:,  0, :] = phi_new[:, -2, :]
			phi_new[:, -1, :] = phi_new[:,  1, :]

	def set_outermost_layers_x(self, phi_new, phi_now=None):
		pass

	def set_outermost_layers_y(self, phi_new, phi_now=None):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`y`-direction so to satisfy the periodic
		boundary conditions.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation,
		yet it is retained as optional argument for compliancy with the
		class hierarchy interface.
		"""
		# Shortcuts
		ny, nj = self._grid.ny, phi_new.shape[1]

		# Set the outermost layers
		# Remark: This is applied only if the field is y-staggered
		# If the field is unstaggered, it should be automatically
		# periodic once the computations are over, due to the periodic
		# extension performed before the computations started
		if nj == ny + 1:
			phi_new[:,  0, :] = phi_new[:, ny-1, :]
			phi_new[:, -1, :] = phi_new[:,  -ny, :]

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self._grid.x, self._grid.nx, self._grid.dx.values.item()
		y, ny, dy = self._grid.y, self._grid.ny, self._grid.dy.values.item()
		z, nz, dz = self._grid.z, self._grid.nz, self._grid.dz.values.item()
		z_hl         = self._grid.z_on_interface_levels.values
		_z_interface = self._grid.z_interface
		_topo_type   = self._grid.topography.topo_type
		_topo_time   = self._grid.topography.topo_time
		_topo_kwargs = self._grid.topography.topo_kwargs

		# Determine the computational x-axis
		_domain_x = DataArray([x.values[0] - nb*dx, x.values[-1] + nb*dx],
							  dims=x.dims, attrs=x.attrs)
		_nx       = nx + 2*nb

		# Determine the computational y-axis
		_domain_y = DataArray([y.values[0] - nb*dy, y.values[-1] + nb*dy],
							  dims=y.dims, attrs=y.attrs)
		_ny       = ny + 2*nb

		# Determine the computational z-axis
		_domain_z = DataArray([z_hl[0], z_hl[-1]],
							  dims=self._grid.z.dims, attrs=z.attrs)
		_nz       = nz

		# Instantiate the computational grid
		return type(self._grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
								z_interface=_z_interface,
								topo_type=_topo_type, topo_time=_topo_time,
								topo_kwargs=_topo_kwargs)


class _Relaxed(HorizontalBoundary):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
	to implement horizontally relaxed boundary conditions.

	Attributes
	----------
	nr : int
		Number of layers which will be affected by relaxation.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		nb : int
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

		dtype = grid.x.values.dtype

		#self._rel  = np.array([1., .99, .95, .8, .5, .2, .05, .01], dtype=dtype)
		self._rel = np.array([1., .54, .24, .09, .036, .013, .005, .002], dtype=dtype)

		self._rel[:nb] = 1.
		self._rrel = self._rel[::-1]
		self.nr = self._rel.size

		# The relaxation matrices for a x-staggered field
		self._stgx_s = np.repeat(self._rel[np.newaxis, :], grid.nx - 2*self.nr + 1, axis=0)
		self._stgx_n = np.repeat(self._rrel[np.newaxis, :], grid.nx - 2*self.nr + 1, axis=0)

		# The relaxation matrices for a y-staggered field
		self._stgy_w = np.repeat(self._rel[:, np.newaxis], grid.ny - 2*self.nr + 1, axis=1)
		self._stgy_e = np.repeat(self._rrel[:, np.newaxis], grid.ny - 2*self.nr + 1, axis=1)

		# The corner relaxation matrices
		cnw = np.zeros((self.nr, self.nr), dtype=dtype)
		for i in range(self.nr):
			cnw[i, i:] = self._rel[i]
			cnw[i:, i] = self._rel[i]
		cne = cnw[:, ::-1]
		cse = np.transpose(cnw)
		csw = np.transpose(cne)

		# Append the corner relaxation matrices to their neighbours
		self._stgx_s = np.concatenate((cnw, self._stgx_s, cne), axis=0)
		self._stgx_n = np.concatenate((csw, self._stgx_n, cse), axis=0)
		self._stgy_w = np.concatenate((cnw, self._stgy_w, cne), axis=1)
		self._stgy_e = np.concatenate((csw, self._stgy_e, cse), axis=1)

		# Repeat all relaxation matrices along the z-axis
		self._stgx_s = np.repeat(self._stgx_s[:, :, np.newaxis], grid.nz+1, axis=2)
		self._stgx_n = np.repeat(self._stgx_n[:, :, np.newaxis], grid.nz+1, axis=2)
		self._stgy_w = np.repeat(self._stgy_w[:, :, np.newaxis], grid.nz+1, axis=2)
		self._stgy_e = np.repeat(self._stgy_e[:, :, np.newaxis], grid.nz+1, axis=2)

	@property
	def mi(self):
		return self._grid.nx

	@property
	def mj(self):
		return self._grid.ny

	def from_physical_to_computational_domain(self, phi):
		"""
		As no extension is required to apply relaxed boundary conditions,
		return a shallow copy of the input field :obj:`phi`.
		"""
		return phi

	def from_computational_to_physical_domain(self, phi_, out_dims=None, change_sign=False):
		"""
		As no extension is required to apply relaxed boundary conditions,
		return a shallow copy of the input field :obj:`phi_`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required
		by the implementation, yet they are retained as optional arguments for
		compliancy with the class hierarchy interface.
		"""
		return phi_

	def enforce(self, phi_new, phi_now):
		"""
		Enforce relaxed lateral boundary conditions.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be
		time-independent, so that they can be inferred from the solution
		at current time.
		"""
		# Shortcuts
		nx, ny = self._grid.nx, self._grid.ny
		nb, nr = self.nb, self.nr
		ni, nj, nk = phi_now.shape

		# The boundary values
		west  = np.repeat(phi_now[0:1,   :, :], nr, axis=0)
		east  = np.repeat(phi_now[-1:,   :, :], nr, axis=0)
		south = np.repeat(phi_now[  :, 0:1, :], nr, axis=1)
		north = np.repeat(phi_now[  :, -1:, :], nr, axis=1)

		# Set the outermost layers
		phi_new[ :nb, nb:-nb, :] = phi_now[ :nb, nb:-nb, :]
		phi_new[-nb:, nb:-nb, :] = phi_now[-nb:, nb:-nb, :]
		phi_new[:,  :nb, :] = phi_now[:,  :nb, :]
		phi_new[:, -nb:, :] = phi_now[:, -nb:, :]

		# Apply the relaxed boundary conditions in the x-direction
		if nj == ny:  # unstaggered
			#phi_new[ :nr, :, :] = self._stgy_w[:, :-1, :nk] * west + \
			#					  (1. - self._stgy_w[:, :-1, :nk]) * phi_new[ :nr, :, :]
			#phi_new[-nr:, :, :] = self._stgy_e[:, :-1, :nk] * east + \
			#					  (1. - self._stgy_e[:, :-1, :nk]) * phi_new[-nr:, :, :]
			phi_new[ :nr, :, :] -= self._stgy_w[:, :-1, :nk] * (phi_new[ :nr, :, :] - west)
			phi_new[-nr:, :, :] -= self._stgy_e[:, :-1, :nk] * (phi_new[-nr:, :, :] - east)
		else:		  # y-staggered
			#phi_new[ :nr, :, :] = self._stgy_w[:, :, :nk] * west + \
			#					  (1. - self._stgy_w[:, :, :nk]) * phi_new[ :nr, :, :]
			#phi_new[-nr:, :, :] = self._stgy_e[:, :, :nk] * east + \
			#					  (1. - self._stgy_e[:, :, :nk]) * phi_new[-nr:, :, :]
			phi_new[ :nr, :, :] -= self._stgy_w[:, :, :nk] * (phi_new[ :nr, :, :] - west)
			phi_new[-nr:, :, :] -= self._stgy_e[:, :, :nk] * (phi_new[-nr:, :, :] - east)

		# Apply the relaxed boundary conditions in the y-direction
		if ni == nx:  # unstaggered
			#phi_new[:,  :nr, :] = self._stgx_s[:-1, :, :nk] * south + \
			#					  (1. - self._stgx_s[:-1, :, :nk]) * phi_new[:,  :nr, :]
			#phi_new[:, -nr:, :] = self._stgx_n[:-1, :, :nk]*north + \
			#					  (1. - self._stgx_n[:-1, :, :nk]) * phi_new[:, -nr:, :]
			phi_new[:,  :nr, :] -= self._stgx_s[:-1, :, :nk] * (phi_new[:,  :nr, :] - south)
			phi_new[:, -nr:, :] -= self._stgx_n[:-1, :, :nk] * (phi_new[:, -nr:, :] - north)
		else:		  # x-staggered
			#phi_new[:,  :nr, :] = self._stgx_s[:, :, :nk] * south + \
			#					  (1. - self._stgx_s[:, :, :nk]) * phi_new[:,  :nr, :]
			#phi_new[:, -nr:, :] = self._stgx_n[:, :, :nk] * north + \
			#					  (1. - self._stgx_n[:, :, :nk]) * phi_new[:, -nr:, :]
			phi_new[:,  :nr, :] -= self._stgx_s[:, :, :nk] * (phi_new[:,  :nr, :] - south)
			phi_new[:, -nr:, :] -= self._stgx_n[:, :, :nk] * (phi_new[:, -nr:, :] - north)

	def set_outermost_layers_x(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`x`-direction equal to the corresponding layers
		of :obj:`phi_now`. In other words, enforce Dirichlet
		conditions in :math:`x`-direction.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be
		time-independent, so that they can be inferred from the solution
		at current time.
		"""
		phi_new[ 0, :, :] = phi_now[ 0, :, :]
		phi_new[-1, :, :] = phi_now[-1, :, :]

	def set_outermost_layers_y(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`y`-direction equal to the corresponding layers
		of :obj:`phi_now`. In other words, apply Dirichlet
		conditions in :math:`y`-direction.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be
		time-independent, so that they can be inferred from the solution
		at current time.
		"""
		phi_new[:,  0, :] = phi_now[:,  0, :]
		phi_new[:, -1, :] = phi_now[:, -1, :]

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain,
		i.e., a deep-copy of the grid stored in this object.
		"""
		return copy.deepcopy(self._grid)


class _RelaxedXZ(HorizontalBoundary):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
	to implement horizontally relaxed boundary conditions for fields
	defined on a computational domain consisting of only one grid point
	in the :math:`y`-direction.

	Attributes
	----------
	nr : int
		Number of layers which will be affected by relaxation.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		nb : int
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

		# The relaxation coefficients
		dtype = grid.x.values.dtype

		#self._rel  = np.array([1., .99, .95, .8, .5, .2, .05, .01], dtype=dtype)
		self._rel = np.array([1., .54, .24, .09, .036, .013, .005, .002], dtype=dtype)

		self._rel[:nb] = 1.
		self._rrel = self._rel[::-1]
		self.nr = self._rel.size

		# The relaxation matrices
		self._stg_w = np.repeat(self._rel[:, np.newaxis, np.newaxis], grid.nz+1, axis=2)
		self._stg_e = np.repeat(self._rrel[:, np.newaxis, np.newaxis], grid.nz+1, axis=2)

	@property
	def mi(self):
		return self._grid.nx

	@property
	def mj(self):
		return self._grid.ny + 2*self.nb

	def from_physical_to_computational_domain(self, phi):
		"""
		While no extension is required to apply relaxed boundary
		conditions along the :math:`x`-direction, :data:`nb` ghost
		layers are appended in the :math:`y`-direction.
		"""
		nb = self.nb
		return np.concatenate((np.repeat(phi[:, 0:1, :], nb, axis=1),
							   phi,
							   np.repeat(phi[:, -1:, :], nb, axis=1)), axis=1)

	def from_computational_to_physical_domain(self, phi_, out_dims=None, change_sign=False):
		"""
		Remove the :data:`nb` outermost :math:`xz`-slices from
		the input field :obj:`phi_`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not
		required by the implementation, yet they are retained as optional
		arguments for compliancy with the class hierarchy interface.
		"""
		nb = self.nb
		return phi_[:, nb:-nb, :]

	def enforce(self, phi_new, phi_now):
		"""
		Enforce relaxed lateral boundary conditions along the :math:`x`-axis.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be
		time-independent, so that they can be inferred from the solution
		at current time.
		"""
		# Shortcuts
		nr = self.nr
		nk = phi_now.shape[2]

		# The boundary values
		west  = np.repeat(phi_now[0:1, :, :], nr, axis=0)
		east  = np.repeat(phi_now[-1:, :, :], nr, axis=0)

		# Apply the relaxed boundary conditions in the x-direction
		#phi_new[ :nr, :, :] = self._stg_w[:, :, :nk] * west + \
		#					  (1. - self._stg_w[:, :, :nk]) * phi_new[ :nr, :, :]
		#phi_new[-nr:, :, :] = self._stg_e[:, :, :nk] * east + \
		#					  (1. - self._stg_e[:, :, :nk]) * phi_new[-nr:, :, :]
		phi_new[ :nr, :, :] -= self._stg_w[:, :, :nk] * (phi_new[ :nr, :, :] - west)
		phi_new[-nr:, :, :] -= self._stg_e[:, :, :nk] * (phi_new[-nr:, :, :] - east)

	def set_outermost_layers_x(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`x`-direction equal to the corresponding layers
		of :obj:`phi_now`. In other words, enforce Dirichlet
		conditions in :math:`x`-direction.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be
		time-independent, so that they can be inferred from the solution
		at current time.
		"""
		phi_new[ 0, :, :] = phi_now[ 0, :, :]
		phi_new[-1, :, :] = phi_now[-1, :, :]

	def set_outermost_layers_y(self, phi_new, phi_now=None):
		pass

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self._grid.x, self._grid.nx, self._grid.dx.values.item()
		y, ny, dy = self._grid.y, self._grid.ny, self._grid.dy.values.item()
		z, nz, dz = self._grid.z, self._grid.nz, self._grid.dz.values.item()
		z_hl         = self._grid.z_on_interface_levels.values
		_z_interface = self._grid.z_interface
		_topo_type   = self._grid.topography.topo_type
		_topo_time   = self._grid.topography.topo_time
		_topo_kwargs = self._grid.topography.topo_kwargs

		# Determine the computational x-axis
		_domain_x = DataArray([x.values[0], x.values[-1]],
							  dims=x.dims, attrs=x.attrs)
		_nx       = nx

		# Determine the computational y-axis
		_domain_y = DataArray([y.values[0] - nb*dy, y.values[0] + nb*dy],
							  dims=y.dims, attrs=y.attrs)
		_ny       = ny + 2 * nb

		# Determine the computational z-axis
		_domain_z = DataArray([z_hl[0], z_hl[-1]],
							  dims=z.dims, attrs=z.attrs)
		_nz       = nz

		# Instantiate the computational grid
		return type(self._grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							    z_interface=_z_interface,
							    topo_type=_topo_type, topo_time=_topo_time,
								topo_kwargs=_topo_kwargs)


class _RelaxedYZ(HorizontalBoundary):
	"""
	This class inherits
	:class:`~tasmania.dynamics.horizontal_boundary.HorizontalBoundary`
	to implement horizontally relaxed boundary conditions for fields
	defined on a computational domain consisting of only one grid point
	in the :math:`x`-direction.

	Attributes
	----------
	nr : int
		Number of layers which will be affected by relaxation.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		nb : int
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

		dtype = grid.x.values.dtype

		#self._rel = np.array([1., .99, .95, .8, .5, .2, .05, .01], dtype=dtype)
		self._rel = np.array([1., .54, .24, .09, .036, .013, .005, .002], dtype=dtype)

		self._rel[:nb] = 1.
		self._rrel = self._rel[::-1]
		self.nr = self._rel.size

		# The relaxation matrices
		self._stg_s = np.repeat(self._rel[np.newaxis, :, np.newaxis], grid.nz+1, axis=2)
		self._stg_n = np.repeat(self._rrel[np.newaxis, :, np.newaxis], grid.nz+1, axis=2)

	@property
	def mi(self):
		return self._grid.nx + 2*self.nb

	@property
	def mj(self):
		return self._grid.ny

	def from_physical_to_computational_domain(self, phi):
		"""
		While no extension is required to apply relaxed boundary
		conditions along the :math:`y`-direction, :data:`nb` ghost
		layers are appended in the :math:`x`-direction.
		"""
		nb = self.nb
		return np.concatenate((np.repeat(phi[0:1, :, :], nb, axis=0),
							   phi,
							   np.repeat(phi[-1:, :, :], nb, axis=0)), axis=0)

	def from_computational_to_physical_domain(self, phi_, out_dims=None, change_sign=False):
		"""
		Return a deep copy of the central :math:`yz`-slices of
		the input field :obj:`phi_`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not
		required by the implementation, yet they are retained as optional
		arguments for compliancy with the class hierarchy interface.
		"""
		nb = self.nb
		return phi_[nb:-nb, :, :]

	def enforce(self, phi_new, phi_now):
		"""
		Enforce relaxed lateral boundary conditions along the :math:`x`-axis.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be
		time-independent, so that they can be inferred from the solution
		at current time.
		"""
		# Shortcuts
		nr = self.nr
		nk = phi_now.shape[2]

		# The boundary values
		south = np.repeat(phi_now[:, 0:1, :], nr, axis=1)
		north = np.repeat(phi_now[:, -1:, :], nr, axis=1)

		# Apply the relaxed boundary conditions in the x-direction
		#phi_new[:,  :nr, :] = self._stg_s[:, :, :nk] * south + \
		#					  (1. - self._stg_s[:, :, :nk]) * phi_new[:,  :nr, :]
		#phi_new[:, -nr:, :] = self._stg_n[:, :, :nk] * north + \
		#					  (1. - self._stg_n[:, :, :nk]) * phi_new[:, -nr:, :]
		phi_new[:,  :nr, :] -= self._stg_s[:, :, :nk] * (phi_new[:,  :nr, :] - south)
		phi_new[:, -nr:, :] -= self._stg_n[:, :, :nk] * (phi_new[:, -nr:, :] - north)

	def set_outermost_layers_x(self, phi_new, phi_now=None):
		pass

	def set_outermost_layers_y(self, phi_new, phi_now):
		"""
		Set the outermost layers of :obj:`phi_new` in the
		:math:`y`-direction equal to the corresponding layers
		of :obj:`phi_now`. In other words, enforce Dirichlet
		conditions in :math:`y`-direction.

		Note
		----
		The Dirichlet conditions at the boundaries are assumed to be
		time-independent, so that they can be inferred from the solution
		at current time.
		"""
		phi_new[:,  0, :] = phi_now[:,  0, :]
		phi_new[:, -1, :] = phi_now[:, -1, :]

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self._grid.x, self._grid.nx, self._grid.dx.values.item()
		y, ny, dy = self._grid.y, self._grid.ny, self._grid.dy.values.item()
		z, nz, dz = self._grid.z, self._grid.nz, self._grid.dz.values.item()
		z_hl         = self._grid.z_on_interface_levels.values
		_z_interface = self._grid.z_interface
		_topo_type   = self._grid.topography.topo_type
		_topo_time   = self._grid.topography.topo_time
		_topo_kwargs = self._grid.topography.topo_kwargs

		# Determine the computational x-axis
		_domain_x = DataArray([x.values[0] - nb*dx, x.values[0] + nb*dx],
							  dims=x.dims, attrs=x.attrs)
		_nx       = nx + 2*nb

		# Determine the computational y-axis
		_domain_y = DataArray([y.values[0], y.values[-1]],
							  dims=y.dims, attrs=y.attrs)
		_ny       = ny

		# Determine the computational z-axis
		_domain_z = DataArray([z_hl[0], z_hl[-1]],
							  dims=z.dims, attrs=z.attrs)
		_nz       = nz

		# Instantiate the computational grid
		return type(self._grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							    z_interface=_z_interface,
							    topo_type=_topo_type, topo_time=_topo_time,
								topo_kwargs=_topo_kwargs)


class _RelaxedSymmetricXZ(_Relaxed):
	"""
	This class inherits
	:class:`~tasmania.dynamics._horizontal_boundary._Relaxed`
	to implement horizontally relaxed boundary conditions for fields
	symmetric with respect to the :math:`xz`-plane
	:math:`y = y_c = 0.5 (a_y + b_y)`, where :math:`a_y` and :math:`b_y`
	denote the end-points of the interval included by the domain in the
	:math:`y`-direction.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		nb : int
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Return the :math:`y`-lowermost half of the domain.
		To accommodate symmetric conditions, we retain (at least)
		:attr:`nb` additional layers in the positive direction of
		the :math:`y`-axis.
		"""
		ny, nj, nb = self._grid.ny, phi.shape[1], self.nb
		half = int((nj + 1) / 2)
		return phi[:, :half + nb, :]

	def from_computational_to_physical_domain(self, phi_, out_dims, change_sign=False):
		"""
		Mirror the computational domain with respect to the
		:math:`xz`-plane :math:`y = y_c`.
		"""
		nb = self.nb
		half = phi_.shape[1] - nb

		phi = np.zeros(out_dims, dtype=phi_.dtype)
		phi[:, :half, :] = phi_[:, :half, :]

		if out_dims[1] % 2 == 0:
			phi[:, half:, :] = - np.flip(phi[:, :half, :], axis=1) if change_sign else \
							   np.flip(phi[:, :half, :], axis=1)
		else:
			phi[:, half:, :] = - np.flip(phi[:, :half-1, :], axis=1) if change_sign else \
							   np.flip(phi[:, :half-1, :], axis=1)

		return phi

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self._grid.x, self._grid.nx, self._grid.dx.values.item()
		y, ny, dy = self._grid.y, self._grid.ny, self._grid.dy.values.item()
		z, nz, dz = self._grid.z, self._grid.nz, self._grid.dz.values.item()
		z_hl         = self._grid.z_on_interface_levels.values
		_z_interface = self._grid.z_interface
		_topo_type   = self._grid.topography.topo_type
		_topo_time   = self._grid.topography.topo_time
		_topo_kwargs = self._grid.topography.topo_kwargs

		# Determine the computational x-axis
		_domain_x = DataArray([x.values[0], x.values[-1]],
							  dims=x.dims, attrs=x.attrs)
		_nx       = nx

		# Determine the computational y-axis
		half = int((ny - 1) / 2)
		_domain_y = DataArray([y.values[0], y.values[half + nb]],
							  dims=y.dims, attrs=y.attrs)
		_ny       = half + nb + 1

		# Determine the computational z-axis
		_domain_z = DataArray([z_hl[0], z_hl[-1]],
							  dims=z.dims, attrs=z.attrs)
		_nz       = nz

		# Instantiate the computational grid
		return type(self._grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							    z_interface=_z_interface,
							    topo_type=_topo_type, topo_time=_topo_time,
								topo_kwargs=_topo_kwargs)


class _RelaxedSymmetricYZ(_Relaxed):
	"""
	This class inherits
	:class:`~tasmania.dynamics._horizontal_boundary._Relaxed`
	to implement horizontally relaxed boundary conditions for
	fields symmetric with respect to the :math:`yz`-plane
	:math:`x = x_c = 0.5 (a_x + b_x)`, where :math:`a_x` and
	:math:`b_x` denote the end-points of the interval included
	 by the domain in the :math:`x`-direction.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of
			:class:`~tasmania.grids.grid_xyz.GridXYZ`
			or one of its derived classes.
		nb : int
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Return the :math:`x`-lowermost half of the domain.
		To accommodate symmetric conditions, we retain (at least)
		:attr:`nb` additional layers in the positive direction of
		the :math:`x`-axis.
		"""
		nx, ni, nb = self._grid.nx, phi.shape[0], self.nb
		half = int((ni + 1) / 2)
		return np.copy(phi[:half + nb, :, :])

	def from_computational_to_physical_domain(self, phi_, out_dims, change_sign=False):
		"""
		Mirror the computational domain with respect to the
		:math:`yz`-axis :math:`x = x_c`.
		"""
		nb = self.nb
		half = phi_.shape[0] - nb

		phi = np.zeros(out_dims, dtype=phi_.dtype)
		phi[:half, :, :] = phi_[:half, :, :]

		if out_dims[0] % 2 == 0:
			phi[half:, :, :] = - np.flip(phi[:half, :, :], axis=0) if change_sign \
							   else np.flip(phi[:half, :, :], axis=0)
		else:
			phi[half:, :, :] = - np.flip(phi[:half-1, :, :], axis=0) if change_sign \
							   else np.flip(phi[:half-1, :, :], axis=0)

		return phi

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self._grid.x, self._grid.nx, self._grid.dx.values.item()
		y, ny, dy = self._grid.y, self._grid.ny, self._grid.dy.values.item()
		z, nz, dz = self._grid.z, self._grid.nz, self._grid.dz.values.item()
		z_hl         = self._grid.z_on_interface_levels.values
		_z_interface = self._grid.z_interface
		_topo_type   = self._grid.topography.topo_type
		_topo_time   = self._grid.topography.topo_time
		_topo_kwargs = self._grid.topography.topo_kwargs

		# Determine the computational x-axis
		half = int((nx - 1) / 2)
		_domain_x = DataArray([x.values[0], x.values[half + nb]],
							  dims=x.dims, attrs=x.attrs)
		_nx       = half + nb + 1

		# Determine the computational y-axis
		_domain_y = DataArray([y.values[0], y.values[-1]],
							  dims=y.dims, attrs=y.attrs)
		_ny       = ny

		# Determine the computational z-axis
		_domain_z = DataArray([z_hl[0], z_hl[-1]],
							  dims=z.dims, attrs=z.attrs)
		_nz       = nz

		# Instantiate the computational grid
		return type(self._grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							    z_interface=_z_interface,
							    topo_type=_topo_type, topo_time=_topo_time,
								topo_kwargs=_topo_kwargs)
