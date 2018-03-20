import os
import sys
from copy import copy
import jinja2
import tempfile

from gridtools.user_interface.domain import Rectangle
from gridtools.intermediate_representation import utils as irutils
from gridtools.intermediate_representation import graph as irgraph
from gridtools.backend import translate_to_python_debug as t2pdb


class PythonDebugGeneration:
    def process(self, ir):
        python_code = self._render_python_code(ir)
        debug_func = self._import_debug_function(ir, python_code)
        self._check_domain_type_and_outputs_shapes(ir)
        ir.computation_func = self._generate_debug_stub_function(ir, debug_func)
        return ir

    def _render_python_code(self, ir):
        stencil_name = self._generate_stencil_name(ir)
        stages_info = self._generate_stages_properties(ir)
        stencil_arguments = self._generate_stencil_arguments(ir)
        stencil_temporaries, temporaries_shape = self._generate_stencil_temporaries(ir)
        return self._generate_python_code(stages_info,
                                          stencil_name,
                                          stencil_arguments,
                                          stencil_temporaries,
                                          temporaries_shape)

    def _import_debug_function(self, ir, python_code):
        function_name = self._generate_stencil_name(ir) + "_debug"
        module_name = self._dump_python_code_into_temporary_file(python_code)
        sys.path.append("/tmp")  # add location of the temporary python module to the PYTHONPATH
        exec("from {} import {}".format(module_name, function_name))
        return locals()[function_name]

    def _generate_stencil_name(self, ir):
        return ir.stencil_configs.definitions_func.__name__

    def _generate_stages_properties(self, ir):
        stages_properties = []
        domain = ir.stencil_configs.domain
        for g in ir.assignment_statements_graphs:
            stage_src = t2pdb.translate_graph(g)
            stage_outputs = irutils.find_roots(g)
            stages_properties.append({"ndim": stage_outputs[0].rank,
                                      "i_start": domain.up_left[0] - stage_outputs[0].access_extent[0],
                                      "i_end": domain.down_right[0] + stage_outputs[0].access_extent[1],
                                      "j_start": domain.up_left[1] - stage_outputs[0].access_extent[2],
                                      "j_end": domain.down_right[1] + stage_outputs[0].access_extent[3],
                                      "expressions": stage_src})
        return stages_properties

    def _generate_stencil_arguments(self, ir):
        roots = irutils.find_roots(ir.graph)
        leaves = [n for n in irutils.find_leaves(ir.graph) if type(n) is not irgraph.NodeConstant]
        stencil_arguments = roots + leaves
        return stencil_arguments

    def _generate_stencil_temporaries(self, ir):
        # In order to be safe with the shape of temporary arrays:
        # - For X and Y axes, we increment the user defined domain by the
        #   stencil's minimum halo
        # - For Z axis, we take the biggest value from the stencil's outputs
        domain = ir.stencil_configs.domain
        temps_up_left_x = domain.up_left[0] - ir.minimum_halo[0]
        temps_up_left_y = domain.up_left[1] - ir.minimum_halo[2]
        temps_down_right_x = domain.down_right[0] + ir.minimum_halo[1]
        temps_down_right_y = domain.down_right[1] + ir.minimum_halo[3]
        outputs_z = [out.shape[2] for _, out in ir.stencil_configs.outputs.items()]
        shape = ( (temps_down_right_x-temps_up_left_x+1),
                  (temps_down_right_y-temps_up_left_y+1),
                  max(outputs_z) )

        temporaries = [n for n in ir.graph if irutils.is_temporary_named_expression(ir.graph, n)]

        return temporaries, shape

    def _generate_python_code(self, stages_info, stencil_name, stencil_arguments, stencil_temporaries, temporaries_shape):
        stencil_arguments_str = [str(arg) for arg in stencil_arguments]
        stencil_temporaries_str = [str(temp) for temp in stencil_temporaries]
        jinja_environment = jinja2.Environment(loader=jinja2.FileSystemLoader('/'))
        jinja_template_path = os.path.dirname(__file__) + "/python_debug_template.py"
        stencil_template = jinja_environment.get_template(jinja_template_path)
        python_code = stencil_template.render(stencil_name=stencil_name,
                                              temps=stencil_temporaries_str,
                                              temps_shape=temporaries_shape,
                                              stages=stages_info,
                                              stencil_args=stencil_arguments_str)
        return python_code

    def _dump_python_code_into_temporary_file(self, python_code):
        _, name = tempfile.mkstemp(suffix=".py")
        with open(name, "w") as fd:
            fd.write(python_code)
        module_name = os.path.splitext(os.path.basename(name))[0]
        return module_name

    def _check_domain_type_and_outputs_shapes(self, ir):
        outputs_shapes = [o.shape for o in ir.stencil_configs.outputs.values()]
        number_of_outputs = len(outputs_shapes)
        if number_of_outputs != 1:
            assert number_of_outputs != 0
            outputs_shapes_are_all_equal = all(s == outputs_shapes[0] for s in outputs_shapes)
            if not outputs_shapes_are_all_equal:
                raise ValueError("Output arrays do not have all the same shape")

        if type(ir.stencil_configs.domain) is not Rectangle:
            raise NotImplementedError("Handling for domains other then Rectangle is not implemented yet")

    def _generate_debug_stub_function(self, ir, debug_func):
        debug_func_kwargs = self._generate_debug_func_keyword_arguments(ir)
        return lambda: debug_func(**debug_func_kwargs)

    def _generate_debug_func_keyword_arguments(self, ir):
        debug_kwargs = copy(ir.stencil_configs.inputs)
        debug_kwargs.update(ir.stencil_configs.outputs)
        return debug_kwargs
