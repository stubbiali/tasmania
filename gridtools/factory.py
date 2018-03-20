from gridtools.ngcompiler import NGCompiler
from gridtools.frontend.crappy.frontend import Frontend
from gridtools.user_interface.mode import Mode
from gridtools.user_interface.vertical_direction import VerticalDirection
from gridtools.backend.compute_extents import AccessExtentsComputation
from gridtools.backend.assignment_statements_detection import AssignmentStatementsDetection
from gridtools.backend.python_debug_generation import PythonDebugGeneration
from gridtools.backend.old_stencil_generation import OldStencilGeneration
from gridtools.backend.numpy_generation import NumpyGeneration


class Factory:
	"""
    This class is responsible for creating the gridtools4py internal objects according to the configuration parameters
    specified by the user. All the logic that takes into account the configuration parameters and decides accordingly
    which objects have to be created is encapsulated in this class. Thus the rest of the system simply asks the factory
    to create polymorphic objects and then uses such objects. The rest of the system should not be polluted by
    conditional statements depending on the configuration parameters.
	"""
	def __init__(self, stencil_configs):
		self._stencil_configs = stencil_configs

	def create_compiler(self):
		return NGCompiler(self.create_passes())

	def create_passes(self):
		return self.create_frontend_passes() + self.create_middleend_passes() + self.create_backend_passes()

	def create_frontend_passes(self):
		return [Frontend()]

	def create_middleend_passes(self):
		return [AccessExtentsComputation(), AssignmentStatementsDetection()]

	def create_backend_passes(self):
		if self._stencil_configs.mode == Mode.ALPHA:
			return [OldStencilGeneration()]
		elif self._stencil_configs.mode == Mode.DEBUG:
			return [PythonDebugGeneration()]
		elif self._stencil_configs.mode == Mode.NUMPY:
			return [NumpyGeneration()]
