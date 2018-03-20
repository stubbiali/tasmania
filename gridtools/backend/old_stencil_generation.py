import os
import sys
from copy import copy
import jinja2
import tempfile

from gridtools.user_interface.domain import Rectangle
from gridtools.intermediate_representation import utils as irutils
from gridtools.intermediate_representation import graph as irgraph
from gridtools.backend import graph_to_old_gt4py


class OldStencilGeneration:
    def process(self, ir):
        python_code = self._render_python_code(ir)
        stencil = self._instantiate_old_stencil(ir, python_code)
        stencil = self._initialize_old_stencil(ir, stencil)
        ir.computation_func = self._generate_old_stencil_stub_function(ir, stencil)
        return ir

    def _render_python_code(self, ir):
        stencil_name = self._generate_stencil_name(ir)
        stages_info = self._generate_stages_properties(ir)
        stencil_arguments = self._generate_stencil_arguments(ir)
        stencil_temporaries = self._generate_stencil_temporaries(ir)
        return self._generate_python_code(stages_info, stencil_name, stencil_arguments, stencil_temporaries)

    def _instantiate_old_stencil(self, ir, python_code):
        stencil_name = self._generate_stencil_name(ir) + "AutoStencil"
        module_name = self._dump_python_code_into_temporary_file(python_code)
        constructor_argument = self._generate_stencil_constructor_argument(ir)
        sys.path.append("/tmp")  # add location of the temporary python module to the PYTHONPATH
        exec("from {} import {}".format(module_name, stencil_name))
        exec("stencil = {}({})".format(stencil_name, str(constructor_argument)))
        return locals()["stencil"]

    def _initialize_old_stencil(self, ir, stencil):
        halo = self._generate_halo(ir)
        stencil.set_halo(halo)
        stencil.set_k_direction("forward")
        stencil.set_backend("python")
        return stencil

    def _generate_stencil_name(self, ir):
        return ir.stencil_configs.definitions_func.__name__

    def _generate_stages_properties(self, ir):
        stages_properties = []
        for g in ir.assignment_statements_graphs:
            stage_src, stage_args, stage_outputs = graph_to_old_gt4py.translate_graph(g)
            stages_properties.append({"name": stage_outputs[0],
                                      "args": stage_args,
                                      "outputs": stage_outputs,
                                      "expressions": stage_src})
        return stages_properties

    def _generate_stencil_arguments(self, ir):
        roots = irutils.find_roots(ir.graph)
        leaves = [n for n in irutils.find_leaves(ir.graph) if type(n) is not irgraph.NodeConstant]
        stencil_arguments = roots + leaves
        return stencil_arguments

    def _generate_stencil_temporaries(self, ir):
        return [n for n in ir.graph if irutils.is_temporary_named_expression(ir.graph, n)]

    def _generate_python_code(self, stages_info, stencil_name, stencil_arguments, stencil_temporaries):
        stencil_arguments_str = [str(arg) for arg in stencil_arguments]
        stencil_temporaries_str = [str(temp) for temp in stencil_temporaries]
        jinja_environment = jinja2.Environment(loader=jinja2.FileSystemLoader('/'))
        jinja_template_path = os.path.dirname(__file__) + "/gt4py_stencil_template.py"
        stencil_template = jinja_environment.get_template(jinja_template_path)
        python_code = stencil_template.render(stencil_name=stencil_name,
                                              temps=stencil_temporaries_str,
                                              stages=stages_info,
                                              stencil_args=stencil_arguments_str)
        return python_code

    def _dump_python_code_into_temporary_file(self, python_code):
        _, name = tempfile.mkstemp(suffix=".py")
        with open(name, "w") as fd:
            fd.write(python_code)
        module_name = os.path.splitext(os.path.basename(name))[0]
        return module_name

    def _generate_stencil_constructor_argument(self, ir):
        inputs_shapes = [i.shape for i in ir.stencil_configs.inputs.values()]
        assert len(inputs_shapes) > 0
        inputs_shapes_are_all_equal = all(s == inputs_shapes[0] for s in inputs_shapes)
        if not inputs_shapes_are_all_equal:
            raise NotImplementedError("Handling for different shapes of the stencil's inputs is not implemented yet."
                                      " At the moment we stupidly take the shape of the stencil inputs and we"
                                      " use it as the shape of the temporaries, i.e. the shape that the stencil's"
                                      " constructor takes as parameters")
        constructor_argument = inputs_shapes[0]
        return constructor_argument

    def _generate_halo(self, ir):
        outputs_shapes = [o.shape for o in ir.stencil_configs.outputs.values()]
        number_of_outputs = len(outputs_shapes)
        if number_of_outputs != 1:
            assert number_of_outputs != 0
            outputs_shapes_are_all_equal = all(s == outputs_shapes[0] for s in outputs_shapes)
            if not outputs_shapes_are_all_equal:
                raise ValueError("Output arrays do not have all the same shape")
        output = list(ir.stencil_configs.outputs.values())[0]

        domain = ir.stencil_configs.domain
        if type(domain) is not Rectangle:
            raise NotImplementedError("Handling for domains other then rectangle is not implemented yet")

        halo_i_negative = domain.up_left[0]
        halo_i_positive = output.shape[0] - 1 - domain.down_right[0]
        halo_j_negative = domain.up_left[1]
        halo_j_positive = output.shape[1] - 1 - domain.down_right[1]
        halo = (halo_i_negative, halo_i_positive, halo_j_negative, halo_j_positive)
        return halo

    def _generate_old_stencil_stub_function(self, ir, stencil):
        run_func_kwargs = self._generate_old_stencil_run_func_keyword_arguments(ir)
        return lambda: stencil.run(**run_func_kwargs)

    def _generate_old_stencil_run_func_keyword_arguments(self, ir):
        run_kwargs = copy(ir.stencil_configs.inputs)
        run_kwargs.update(ir.stencil_configs.outputs)
        return run_kwargs
