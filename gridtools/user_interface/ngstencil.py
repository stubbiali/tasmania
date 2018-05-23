from gridtools.factory import Factory
from gridtools.user_interface.mode import Mode
from gridtools.user_interface.vertical_direction import VerticalDirection


class NGStencil:
	"""
	This class allows to define a Stencil. An instance of this class shall be used to:
	- specify the definitions of the finite difference equations to be solved.
	- bind instances of numpy arrays to symbols defined in the finite difference equations as inputs/outputs.
	- specify the data domain on which the computations should be performed.
	- specify the desired backend to perform the computation.
	- specify the direction which should be pursued when iterating over the vertical axis.
	"""
	def __init__(self, **kwargs):
		# Set default values
		definitions_func = inputs = outputs = domain = None
		constant_inputs = global_inputs = {}
		mode = Mode.DEBUG
		vertical_direction = VerticalDirection.PARALLEL

		# Read keyword arguments
		for key in kwargs:
			if key == 'definitions_func':
				definitions_func = kwargs[key]
			elif key == 'inputs':
				inputs = kwargs[key]
			elif key == 'constant_inputs':
				constant_inputs = kwargs[key]
			elif key == 'global_inputs':
				global_inputs = kwargs[key]
			elif key == 'outputs':
				outputs = kwargs[key]
			elif key == 'domain':
				domain = kwargs[key]
			elif key == 'mode':
				mode = kwargs[key]
			elif key == 'vertical_direction':
				vertical_direction = kwargs[key]
			else:
				raise ValueError("\n  NGStencil accepts the following keyword arguments: \n" 
								 "  - definitions_func, \n"
								 "  - inputs, \n"
								 "  - constant_inputs [default: {}], \n"
								 "  - global_inputs [default: {}], \n"
								 "  - outputs, \n"
								 "  - domain, \n"
								 "  - mode [default: DEBUG], \n"
								 "  - vertical_direction [default: PARALLEL]. \n"
								 "  The order does not matter.")

		# Consistency check
		assert definitions_func is not None, "Please specify a definitions function."
		assert inputs is not None, "Please specify stencil's inputs." 
		assert outputs is not None, "Please specify stencil's outputs." 
		assert domain is not None, "Please specify the computational domain." 

		self._configs = NGStencilConfigs(definitions_func = definitions_func,
										 inputs = inputs,
										 constant_inputs = constant_inputs,
										 global_inputs = global_inputs,
										 outputs = outputs,
										 domain = domain,
										 mode = mode,
										 vertical_direction = vertical_direction)
		self._factory = Factory(self._configs)
		self._driver = None

	@property
	def inputs(self):
		return self._configs.inputs

	@property
	def constant_inputs(self):
		return self._configs.constant_inputs

	@property
	def global_inputs(self):
		return self._configs.global_inputs

	@property
	def outputs(self):
		return self._configs.outputs

	@property
	def domain(self):
		return self._configs.domain

	@property
	def mode(self):
		return self._configs.mode

	def compute(self):
		if self._driver is None:
			self._driver = self._factory.create_compiler()
			self._compute_func = self._driver.compile(self._configs)
		return self._compute_func()

	def get_extent(self):
		if self._driver is None:
			self._driver = self._factory.create_compiler()
			self._compute_func = self._driver.compile(self._configs)
		return self._driver.get_extent()


class NGStencilConfigs:
	"""
	This class is used internally in gridtools4py to encapsulate and pass around
	the configurations specified by the user for a given stencil.
	"""
	def __init__(self, *, definitions_func, inputs, constant_inputs, 				 
				 global_inputs, outputs, domain, mode, vertical_direction):
		self.definitions_func = definitions_func
		self.inputs = inputs
		self.constant_inputs = constant_inputs
		self.global_inputs = global_inputs
		self.outputs = outputs
		self.domain = domain
		self.mode = mode
		self.vertical_direction = vertical_direction
