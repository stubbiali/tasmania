import os
import sys
from copy import copy
import jinja2
import tempfile

from gridtools.user_interface.domain import Rectangle
from gridtools.user_interface.vertical_direction import VerticalDirection
from gridtools.intermediate_representation import utils as irutils
from gridtools.intermediate_representation import graph as irgraph
from gridtools.backend import translate_to_numpy as t2np


class NumpyGeneration:
	def process(self, ir):
		"""
		Set the computation function performing the stencil.

		:param ir	Stencil's intermediate representation

		:return Stencil's intermediate representation update with the associated computation function
		"""
		# Generate source code
		python_code = self._render_python_code(ir)

		# Import Numpy function implementing the stencil
		numpy_func = self._import_numpy_function(ir, python_code)

		# Ensure all output arrays have the same dimensions
		self._check_domain_type_and_outputs_shapes(ir)

		# Computation function as a lambda
		ir.computation_func = self._generate_numpy_stub_function(ir, numpy_func)
		
		return ir


	def _render_python_code(self, ir):
		"""
		Generate the source code implementing the stencil.

		:param ir	Stencil's intermediate representation

		:return Temporary source file
		"""
		# Extract stencil's name, arguments and temporaries
		stencil_name = self._generate_stencil_name(ir)
		stencil_arguments = self._generate_stencil_arguments(ir)
		stencil_temporaries, temporaries_shape = self._generate_stencil_temporaries(ir)

		# Get further stages' properties
		stages_properties = self._generate_stages_properties(ir)

		# Generate source file
		return self._generate_python_code(stages_properties,
										  stencil_name,
										  stencil_arguments,
										  stencil_temporaries,
										  temporaries_shape)


	def _import_numpy_function(self, ir, python_code):
		"""
		Import the Numpy function implementing the stencil.

		:param ir			Stencil's intermediate representation
		:param python_code	Python source code

		:return Pointer to Numpy function implementing the stencil
		"""
		function_name = self._generate_stencil_name(ir) + "_numpy"
		module_name = self._dump_python_code_into_temporary_file(python_code)
		sys.path.append("/tmp")  # add location of the temporary python module to the PYTHONPATH
		exec("from {} import {}".format(module_name, function_name))
		#exec("from {} import {}".format("tmpyfhc8zm2", function_name))
		return locals()[function_name]


	def _generate_stencil_name(self, ir):
		"""
		Deduce stencil's name from definitions function.

		:param ir	Stencil's intermediate representation

		:return String with stencil's name
		"""
		return ir.stencil_configs.definitions_func.__name__


	def _generate_stages_properties(self, ir):
		"""
		Extract useful stages' properties, e.g., source code.

		:param ir	Stencil's intermediate representation

		:return Stages' properties as a list of dictionaries
		"""
		stages_properties = []
		domain = ir.stencil_configs.domain
		vertical_direction = ir.stencil_configs.vertical_direction

		for g in ir.assignment_statements_graphs:
			stage_outputs = irutils.find_roots(g)
			domain = ir.stencil_configs.domain

			# Determine access extents for the output
			out_access_extents = [domain.up_left[0] - stage_outputs[0].access_extent[0],
								  domain.down_right[0] + stage_outputs[0].access_extent[1] + 1,
								  domain.up_left[1] - stage_outputs[0].access_extent[2],
								  domain.down_right[1] + stage_outputs[0].access_extent[3] + 1]
			if stage_outputs[0].rank > 2:
				if vertical_direction is VerticalDirection.BACKWARD:
					out_access_extents += [domain.up_left[2] - stage_outputs[0].access_extent[4] - 1,
								 	   	   domain.down_right[2] + stage_outputs[0].access_extent[5]]
				else:	
					out_access_extents += [domain.up_left[2] - stage_outputs[0].access_extent[4],
								 	   	   domain.down_right[2] + stage_outputs[0].access_extent[5] + 1]
			
			# Translate the stage
			stage_src = t2np.translate_stage_graph(g, out_access_extents, vertical_direction)

			# Store needed properties, distinguishing based on vertical direction
			if (stage_outputs[0].rank < 3) or (vertical_direction is VerticalDirection.PARALLEL):
				stages_properties.append({"vertical_mode": "vectorized",
										  "expression": stage_src})
			else:
				if vertical_direction is VerticalDirection.FORWARD:
					k_start, k_stop = out_access_extents[4], out_access_extents[5]
				else:
					k_start, k_stop = out_access_extents[5], out_access_extents[4]
				stages_properties.append({"vertical_mode": "serialized",
										  "k_start": k_start,
										  "k_stop": k_stop,
										  "expression": stage_src})
		return stages_properties


	def _generate_stencil_arguments(self, ir):
		"""
		Deduce stencil's arguments, i.e., inputs and outputs.

		:param ir	Stencil's intermediate representation

		:return List of stencil arguments
		"""
		args = [arg for arg in ir.stencil_configs.global_inputs] + \
			   [arg for arg in ir.stencil_configs.constant_inputs] + \
			   [arg for arg in ir.stencil_configs.inputs] + \
			   [arg for arg in ir.stencil_configs.outputs]
		return args


	def _generate_stencil_temporaries(self, ir):
		"""
		Extract graph's nodes corresponding to temporary expressions and determine shape of temporary arrays.

		:param ir	Stencil's intermediate representation

		:return Graph's nodes corresponding to temporary expressions
		:return Shape of temporary arrays
		"""
		# In order to be safe with the shape of temporary arrays:
		# - For X and Y axes, we increment the user defined domain by the
		#	stencil's minimum halo
		# - For Z axis, we take the biggest value from the stencil's outputs
		domain = ir.stencil_configs.domain
		temps_up_left_x = domain.up_left[0] - ir.minimum_halo[0]
		temps_up_left_y = domain.up_left[1] - ir.minimum_halo[2]
		temps_down_right_x = domain.down_right[0] + ir.minimum_halo[1]
		temps_down_right_y = domain.down_right[1] + ir.minimum_halo[3]
		shape = ( (temps_down_right_x-temps_up_left_x+1),
				  (temps_down_right_y-temps_up_left_y+1) )

		if len(domain.up_left) > 2:
			outputs_z = [out.shape[2] for _, out in ir.stencil_configs.outputs.items()]
			shape = shape + ( max(outputs_z), )
		
		# Extract temporary nodes
		# Note: the user may want to get back a temporary array. Think for instance at the 
		# fluid height in a conservative-form solver for the shallow water equations. 
		# To accomodate for this, we consider all the temporary nodes which do not appear
		# in the user-specified dictionary ir.stencil_configs.outputs
		outputs = ir.stencil_configs.outputs
		temporaries = [str(n) for n in ir.graph \
					   if (irutils.is_temporary_named_expression(ir.graph, n) and str(n) not in outputs)]

		return temporaries, shape


	def _generate_python_code(self, stages_properties, stencil_name, stencil_arguments, stencil_temporaries, temporaries_shape):
		"""
		Generate source file implementing the stencil.

		:param stages_properties	Stages's properties, e.g., Python expression
		:param stencil_name			Stencil's name
		:param stencil_arguments	Stencil's arguments, i.e., inouts and outputs
		:param stencil_temporaries	Graph's nodes corresponding to temporary expressions
		:param temporaries_shape	Shape of temporary arrays

		:return Source file
		"""
		# Set up Jinja environment
		jinja_environment = jinja2.Environment(loader=jinja2.FileSystemLoader('/'))
		jinja_template_path = os.path.dirname(__file__) + "/numpy_template.py"
		stencil_template = jinja_environment.get_template(jinja_template_path)

		# Render Python code
		python_code = stencil_template.render(stencil_name = stencil_name,
											  stencil_args = stencil_arguments,
											  temps = stencil_temporaries,
											  temps_shape = temporaries_shape,
											  stages = stages_properties)
		return python_code


	def _dump_python_code_into_temporary_file(self, python_code):
		"""
		Write Python code in a temporary file.

		:param python_code	Python code to write

		:return Name of the resulting Python module
		"""
		# Generate a temporary Python file, then dump Python code into it 
		_, name = tempfile.mkstemp(suffix=".py")
		with open(name, "w") as fd:
			fd.write(python_code)

		# Extract module's name
		module_name = os.path.splitext(os.path.basename(name))[0]
		return module_name


	def _check_domain_type_and_outputs_shapes(self, ir):
		"""
		Ensure output arrays have all the same dimensions.

		:param ir	Stencil's intermediate representation
		"""
		outputs_shapes = [o.shape for o in ir.stencil_configs.outputs.values()]
		number_of_outputs = len(outputs_shapes)
		if number_of_outputs != 1:
			assert number_of_outputs != 0
			outputs_shapes_are_all_equal = all(s == outputs_shapes[0] for s in outputs_shapes)
			if not outputs_shapes_are_all_equal:
				raise ValueError("Output arrays do not have all the same shape")
		
		# Only support for rectangular domains
		if type(ir.stencil_configs.domain) is not Rectangle:
			raise NotImplementedError("Handling for domains other then Rectangle is not implemented yet")


	def _generate_numpy_stub_function(self, ir, numpy_func):
		"""
		Convert Numpy function to a lambda function.

		:param ir			Stencil's intermediate representation
		:param numpy_func	Numpy function implementing the stencil

		:return Arguments-free lambda implementing the stencil
		"""
		numpy_func_kwargs = copy(ir.stencil_configs.inputs)
		numpy_func_kwargs.update(ir.stencil_configs.constant_inputs)
		numpy_func_kwargs.update(ir.stencil_configs.global_inputs)
		numpy_func_kwargs.update(ir.stencil_configs.outputs)
		return lambda: numpy_func(**numpy_func_kwargs)
