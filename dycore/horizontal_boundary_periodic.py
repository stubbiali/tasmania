"""
Classes implementing periodic horizontal boundary conditions.
"""
import abc
import numpy as np

from tasmania.dycore.horizontal_boundary import HorizontalBoundary
from tasmania.namelist import datatype

class Periodic(HorizontalBoundary):
	"""
	This class inherits :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` to implement horizontally 
	periodic boundary conditions.

	Attributes
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes representing the underlying grid.
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Periodically extend the field :obj:`phi` with :attr:`nb` extra layers.

		Parameters
		----------
		phi : array_like
			The :class:`numpy.ndarray` to extend.

		Return
		------
		array_like :
			The extended :class:`numpy.ndarray`.
		"""
		# Shortcuts
		nx, ny, nz, nb = self.grid.nx, self.grid.ny, self.grid.nz, self.nb
		ni, nj, nk = phi.shape

		# Decorate the input field
		phi_ = np.zeros((ni + 2 * nb, nj + 2 * nb, nk), dtype = datatype)
		phi_[nb:-nb, nb:-nb, :] = phi

		# Extend in the x-direction
		phi_[ :nb, nb:-nb, :] = phi_[(nx - 1):(nx - 1 + nb)    , nb:-nb, :]
		phi_[-nb:, nb:-nb, :] = phi_[(- nx + 1 - nb):(- nx + 1), nb:-nb, :]

		# Extend in the y-direction
		phi_[:,  :nb, :] = phi_[:, (ny - 1):(ny - 1 + nb)    , :]
		phi_[:, -nb:, :] = phi_[:, (- ny + 1 - nb):(- ny + 1), :]

		return phi_

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Shrink the field :obj:`phi_` by removing the :attr:`nb` outermost layers.

		Parameters
		----------
		phi_ : array_like 
			The :class:`numpy.ndarray` to shrink.
		out_dims : tuple 
			Tuple of the output array dimensions.
		change_sign : bool 
			:obj:`True` if the field should change sign through the symmetry plane (if any), :obj:`False` otherwise.

		Return
		------
		array_like :
			The shrunk :class:`numpy.ndarray`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		nx, ny, nz, nb = self.grid.nx, self.grid.ny, self.grid.nz, self.nb
		ni, nj, nk = phi_.shape

		# Check
		assert ((ni == nx + 2*nb) or (ni == nx + 1 + 2*nb)) and ((nj == ny + 2*nb) or (nj == ny + 1 + 2*nb)), \
			   'The input field does not have the dimensions one would expect.' \
			   'Hint: has the field been previously extended?'

		# Shrink
		return phi_[nb:-nb, nb:-nb, :]

	def apply(self, phi_new, phi_now = None):
		"""
		Apply horizontally periodic boundary conditions on :attr:`phi_new`.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		nx, ny, nz, nb = self.grid.nx, self.grid.ny, self.grid.nz, self.nb
		ni, nj, _ = phi_now.shape

		# Make the field periodic in the x-direction
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically periodic once the computations are over, 
		# due to the periodic extension performed before the computations started
		if ni == nx + 1:
			phi_new[ 0, :, :] = phi_new[-2, :, :]
			phi_new[-1, :, :] = phi_new[ 1, :, :]

		# Make the field periodic in the y-direction
		# Remark: This is applied only if the field is y-staggered
		# If the field is unstaggered, it should be automatically periodic once the computations are over, 
		# due to the periodic extension performed before the computations started
		if nj == ny + 1:
			phi_new[:,  0, :] = phi_new[:, -2, :]
			phi_new[:, -1, :] = phi_new[:,  1, :]

	def set_outermost_layers_x(self, phi_new, phi_now = None):
		"""
		Set the outermost layers of :obj:`phi_new` in the :math:`x`-direction so to satisfy the periodic 
		boundary conditions. 

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		nx, ni = self.grid.nx, phi_new.shape[0]

		# Set the outermost layers
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically periodic once the computations are over, 
		# due to the periodic extension performed before the computations started
		if ni == nx + 1:
			phi_new[0, :, :], phi_new[-1, :, :] = phi_new[nx-1, :, :], phi_new[-nx, :, :]

	def set_outermost_layers_y(self, phi_new, phi_now = None):
		"""
		Set the outermost layers of :obj:`phi_new` in the :math:`y`-direction so to satisfy the periodic 
		boundary conditions. 

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		ny, nj = self.grid.ny, phi_new.shape[1]

		# Set the outermost layers
		# Remark: This is applied only if the field is y-staggered
		# If the field is unstaggered, it should be automatically periodic once the computations are over, 
		# due to the periodic extension performed before the computations started
		if nj == ny + 1:
			phi_new[:, 0, :], phi_new[:, -1, :] = phi_new[:, ny-1, :], phi_new[:, -ny, :]

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.

		Return
		------
		obj :
			Instance of the same class of :obj:`tasmania.dycore.horizontal_boundary.HorizontalBoundary.grid` 
			representing the underlying computational grid.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self.grid.x, self.grid.nx, self.grid.dx
		y, ny, dy = self.grid.y, self.grid.ny, self.grid.dy
		z, nz, dz = self.grid.z_on_interface_levels, self.grid.nz, self.grid.dz
		_z_interface = self.grid.z_interface
		_topo_type   = self.grid.topo_type
		_topo_time   = self.grid.topo_time
		_topo_kwargs = self.grid.topo_kwargs

		# Determine the computational x-axis
		_domain_x = [x[0] - nb * dx, x[-1] + nb * dx]
		_nx       = nx + 2 * nb
		_units_x  = None if x.attrs is None else x.attrs.get('dims', None)
		_dims_x   = x.dims

		# Determine the computational y-axis
		_domain_y = [y[0] - nb * dy, y[-1] + nb * dy]
		_ny       = ny + 2 * nb
		_units_y  = None if y.attrs is None else y.attrs.get('dims', None)
		_dims_y   = y.dims

		# Determine the computational z-axis
		_domain_z = [z[0], z[-1]]
		_nz       = nz
		_units_z  = None if z.attrs is None else z.attrs.get('dims', None)
		_dims_z   = z.dims

		# Instantiate the computational grid
		return type(self.grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							   units_x = _units_x, dims_x = _dims_x,
							   units_y = _units_y, dims_y = _dims_y,
							   units_z = _units_z, dims_z = _dims_z,
							   z_interface = _z_interface,
							   topo_type = _topo_type, topo_time = _topo_time, **_topo_kwargs)
		

class PeriodicXZ(HorizontalBoundary):
	"""
	This class inherits :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` to implement horizontally 
	periodic boundary conditions for fields defined on a computational domain consisting of only one grid 
	point in the :math:`y`-direction.

	Attributes
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes representing the underlying grid.
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Periodically extend the field :obj:`phi` with :attr:`nb` extra layers in the :math:`x`-direction,
		then add :data:`nb` ghost layers in the :math:`y`-direction.

		Parameters
		----------
		phi : array_like
			The :class:`numpy.ndarray` to extend.

		Return
		------
		array_like :
			The extended :class:`numpy.ndarray`.
		"""
		# Shortcuts
		nx, ny, nz, nb = self.grid.nx, self.grid.ny, self.grid.nz, self.nb
		ni, nj, nk = phi.shape

		# Decorate the input field
		phi_ = np.zeros((ni + 2*nb, nj, nk), dtype = datatype)
		phi_[nb:-nb, :, :] = phi

		# Extend in the x-direction
		phi_[ :nb, :, :] = phi_[(nx - 1):(nx - 1 + nb)    , :, :]
		phi_[-nb:, :, :] = phi_[(- nx + 1 - nb):(- nx + 1), :, :]

		# Repeat in the y-direction
		return np.concatenate((np.repeat(phi_[:, 0:1, :], nb, axis = 1),
							   phi_,
							   np.repeat(phi_[:, -1:, :], nb, axis = 1)), axis = 1)

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Return the central :math:`xz`-slices of the input field :data:`phi_`, removing the :attr:`nb` outermost 
		layers in :math:`y`-direction.

		Parameters
		----------
		phi_ : array_like 
			The :class:`numpy.ndarray` to shrink.
		out_dims : tuple 
			Tuple of the output array dimensions.
		change_sign : bool 
			:obj:`True` if the field should change sign through the symmetry plane (if any), :obj:`False` otherwise.

		Return
		------
		array_like :
			The shrunk :class:`numpy.ndarray`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		nb = self.nb
		return phi_[nb:-nb, nb:-nb, :]

	def apply(self, phi_new, phi_now = None):
		"""
		Apply horizontally periodic boundary conditions on :attr:`phi_new`.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		nx, ni = self.grid.nx, phi_now.shape[0]

		# Make the field periodic in the x-direction
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically periodic once the computations are over, 
		# due to the periodic extension performed before the computations started
		if ni == nx + 1:
			phi_new[ 0, :, :] = phi_new[-2, :, :]
			phi_new[-1, :, :] = phi_new[ 1, :, :]

	def set_outermost_layers_x(self, phi_new, phi_now = None):
		"""
		Set the outermost layers of :obj:`phi_new` in the :math:`x`-direction so to satisfy the periodic 
		boundary conditions. 

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		nx, ni = self.grid.nx, phi_new.shape[0]

		# Set the outermost layers
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically periodic once the computations are over, 
		# due to the periodic extension performed before the computations started
		if ni == nx + 1:
			phi_new[0, :, :], phi_new[-1, :, :] = phi_new[nx-1, :, :], phi_new[-nx, :, :]

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.

		Return
		------
		obj :
			Instance of the same class of :obj:`tasmania.dycore.horizontal_boundary.HorizontalBoundary.grid` 
			representing the underlying computational grid.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self.grid.x, self.grid.nx, self.grid.dx
		y, ny, dy = self.grid.y, self.grid.ny, self.grid.dy
		z, nz, dz = self.grid.z_on_interface_levels, self.grid.nz, self.grid.dz
		_z_interface = self.grid.z_interface
		_topo_type   = self.grid.topo_type
		_topo_time   = self.grid.topo_time
		_topo_kwargs = self.grid.topo_kwargs

		# Determine the computational x-axis
		_domain_x = [x[0] - nb * dx, x[-1] + nb * dx]
		_nx       = nx + 2 * nb
		_units_x  = None if x.attrs is None else x.attrs.get('dims', None)
		_dims_x   = x.dims

		# Determine the computational y-axis
		_domain_y = [y[0] - nb * dy, y[-1] + nb * dy]
		_ny       = ny + 2 * nb
		_units_y  = None if y.attrs is None else y.attrs.get('dims', None)
		_dims_y   = y.dims

		# Determine the computational z-axis
		_domain_z = [z[0], z[-1]]
		_nz       = nz
		_units_z  = None if z.attrs is None else z.attrs.get('dims', None)
		_dims_z   = z.dims

		# Instantiate the computational grid
		return type(self.grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							   units_x = _units_x, dims_x = _dims_x,
							   units_y = _units_y, dims_y = _dims_y,
							   units_z = _units_z, dims_z = _dims_z,
							   z_interface = _z_interface,
							   topo_type = _topo_type, topo_time = _topo_time, **_topo_kwargs)


class PeriodicYZ(HorizontalBoundary):
	"""
	This class inherits :class:`~tasmania.dycore.horizontal_boundary.HorizontalBoundary` to implement horizontally 
	periodic boundary conditions for fields defined on a computational domain consisting of only one grid 
	point in the :math:`x`-direction.

	Attributes
	----------
	grid : obj
		:class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes representing the underlying grid.
	nb : int
		Number of boundary layers.
	"""
	def __init__(self, grid, nb):
		"""
		Constructor.

		Parameters
		----------
		grid : obj
			The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ` or one of its derived classes.
		nb : int 
			Number of boundary layers.
		"""
		super().__init__(grid, nb)

	def from_physical_to_computational_domain(self, phi):
		"""
		Periodically extend the field :obj:`phi` with :attr:`nb` extra layers in the :math:`y`-direction,
		then add :data:`nb` ghost layers in the :math:`x`-direction.

		Parameters
		----------
		phi : array_like
			The :class:`numpy.ndarray` to extend.

		Return
		------
		array_like :
			The extended :class:`numpy.ndarray`.
		"""
		# Shortcuts
		nx, ny, nz, nb = self.grid.nx, self.grid.ny, self.grid.nz, self.nb
		ni, nj, nk = phi.shape

		# Decorate the input field
		phi_ = np.zeros((nx, nj + 2*nb, nk), dtype = datatype)
		phi_[:, nb:-nb, :] = phi

		# Extend in the x-direction
		phi_[ :nb, :, :] = phi_[(nx - 1):(nx - 1 + nb)    , :, :]
		phi_[-nb:, :, :] = phi_[(- nx + 1 - nb):(- nx + 1), :, :]

		# Repeat in the x-direction
		return np.concatenate((np.repeat(phi_[0:1, :, :], nb, axis = 0),
							   phi_,
							   np.repeat(phi_[-1:, :, :], nb, axis = 0)), axis = 0)

	def from_computational_to_physical_domain(self, phi_, out_dims = None, change_sign = True):
		"""
		Return the central :math:`yz`-slices of the input field :data:`phi_`, removing the :attr:`nb` outermost 
		layers in :math:`x`-direction.

		Parameters
		----------
		phi_ : array_like 
			The :class:`numpy.ndarray` to shrink.
		out_dims : tuple 
			Tuple of the output array dimensions.
		change_sign : bool 
			:obj:`True` if the field should change sign through the symmetry plane (if any), :obj:`False` otherwise.

		Return
		------
		array_like :
			The shrunk :class:`numpy.ndarray`.

		Note
		----
		The arguments :data:`out_dims` and :data:`change_sign` are not required by the implementation, 
		yet they are retained as optional arguments for compliancy with the class hierarchy interface.
		"""
		nb = self.nb
		return phi_[nb:-nb, nb:-nb, :]

	def apply(self, phi_new, phi_now = None):
		"""
		Apply horizontally periodic boundary conditions on :attr:`phi_new`.

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		ny, nj = self.grid.ny, phi_now.shape[1]

		# Make the field periodic in the x-direction
		# Remark: This is applied only if the field is x-staggered
		# If the field is unstaggered, it should be automatically periodic once the computations are over, 
		# due to the periodic extension performed before the computations started
		if nj == ny + 1:
			phi_new[:,  0, :] = phi_new[:, -2, :]
			phi_new[:, -1, :] = phi_new[:,  1, :]

	def set_outermost_layers_y(self, phi_new, phi_now = None):
		"""
		Set the outermost layers of :obj:`phi_new` in the :math:`y`-direction so to satisfy the periodic 
		boundary conditions. 

		Parameters
		----------
		phi_new : array_like
			:class:`numpy.ndarray` representing the field on which applying the boundary conditions.
		phi_now : `array_like`, optional 
			:class:`numpy.ndarray` representing the field at the current time.

		Note
		----
		The argument :data:`phi_now` is not required by the implementation, yet it is retained as optional
		argument for compliancy with the class hierarchy interface.
		"""
		# Shortcuts
		ny, nj = self.grid.ny, phi_new.shape[1]

		# Set the outermost layers
		# Remark: This is applied only if the field is y-staggered
		# If the field is unstaggered, it should be automatically periodic once the computations are over, 
		# due to the periodic extension performed before the computations started
		if nj == ny + 1:
			phi_new[:, 0, :], phi_new[:, -1, :] = phi_new[:, ny-1, :], phi_new[:, -ny, :]

	def get_computational_grid(self):
		"""
		Get the *computational* grid underlying the computational domain.

		Return
		------
		obj :
			Instance of the same class of :obj:`tasmania.dycore.horizontal_boundary.HorizontalBoundary.grid` 
			representing the underlying computational grid.
		"""
		# Shortcuts
		nb = self.nb
		x, nx, dx = self.grid.x, self.grid.nx, self.grid.dx
		y, ny, dy = self.grid.y, self.grid.ny, self.grid.dy
		z, nz, dz = self.grid.z_on_interface_levels, self.grid.nz, self.grid.dz
		_z_interface = self.grid.z_interface
		_topo_type   = self.grid.topo_type
		_topo_time   = self.grid.topo_time
		_topo_kwargs = self.grid.topo_kwargs

		# Determine the computational x-axis
		_domain_x = [x[0] - nb * dx, x[-1] + nb * dx]
		_nx       = nx + 2 * nb
		_units_x  = None if x.attrs is None else x.attrs.get('dims', None)
		_dims_x   = x.dims

		# Determine the computational y-axis
		_domain_y = [y[0] - nb * dy, y[-1] + nb * dy]
		_ny       = ny + 2 * nb
		_units_y  = None if y.attrs is None else y.attrs.get('dims', None)
		_dims_y   = y.dims

		# Determine the computational z-axis
		_domain_z = [z[0], z[-1]]
		_nz       = nz
		_units_z  = None if z.attrs is None else z.attrs.get('dims', None)
		_dims_z   = z.dims

		# Instantiate the computational grid
		return type(self.grid)(_domain_x, _nx, _domain_y, _ny, _domain_z, _nz,
							   units_x = _units_x, dims_x = _dims_x,
							   units_y = _units_y, dims_y = _dims_y,
							   units_z = _units_z, dims_z = _dims_z,
							   z_interface = _z_interface,
							   topo_type = _topo_type, topo_time = _topo_time, **_topo_kwargs)
