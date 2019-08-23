
import time

import numpy as np
from numpy import dtype


from gridtools import Boundary, FieldInfo, ParameterInfo, StencilObject


class _stencil_defs_numpy(StencilObject):

    _gt_backend_ = "numpy"

    _gt_source_ = {}

    _gt_field_info_ = {'in_phi': FieldInfo(boundary=Boundary(((2, 2), (2, 2), (0, 0))), dtype=dtype('float64')), 'in_gamma': FieldInfo(boundary=Boundary(
        ((0, 0), (0, 0), (0, 0))), dtype=dtype('float64')), 'out_phi': FieldInfo(boundary=Boundary(((0, 0), (0, 0), (0, 0))), dtype=dtype('float64'))}

    _gt_parameter_info_ = {'dx': ParameterInfo(dtype=dtype('float64')), 'dy': ParameterInfo(dtype=dtype('float64'))}

    _gt_constants_ = {}

    _gt_default_domain_ = None

    _gt_default_origin_ = None

    _gt_options_ = {'name': '_stencil_defs_numpy', 'module': 'tasmania.python.dwarfs.horizontal_diffusion',
                    'min_signature': False, 'rebuild': True, 'default_domain': None, 'default_origin': None, 'backend_opts': {}}

    @property
    def backend(self):
        return type(self)._gt_backend_

    @property
    def source(self):
        return type(self)._gt_source_

    @property
    def field_info(self) -> dict:
        return type(self)._gt_field_info_

    @property
    def parameter_info(self) -> dict:
        return type(self)._gt_parameter_info_

    @property
    def constants(self) -> dict:
        return type(self)._gt_constants_

    @property
    def default_domain(self) -> dict:
        return type(self)._gt_default_domain_

    @property
    def default_origin(self) -> dict:
        return type(self)._gt_default_origin_

    @property
    def options(self) -> dict:
        return type(self)._gt_options_

    def __call__(self, in_phi, in_gamma, out_phi, dx, dy, domain=None, origin=None, exec_info=None):
        if exec_info is not None:
            exec_info["call_start_time"] = time.perf_counter()

        self.call_run(
            field_args=dict(in_phi=in_phi, in_gamma=in_gamma, out_phi=out_phi),
            parameter_args=dict(dx=dx, dy=dy),
            domain=domain,
            origin=origin,
            exec_info=exec_info
        )

    def run(self, in_phi, in_gamma, out_phi, dx, dy, _domain_, _origin_, exec_info):
        if exec_info is not None:
            exec_info["domain"] = _domain_
            exec_info["origin"] = _origin_
            exec_info["run_start_time"] = time.perf_counter()
        # Sliced views of the stencil fields (domain + borders)
        in_phi__O = _origin_['in_phi']
        in_gamma__O = _origin_['in_gamma']
        out_phi__O = _origin_['out_phi']

        # K splitters
        _splitters_ = [0, _domain_[2]]
        # Computations
        # stage__6:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        out_phi[out_phi__O[0]: out_phi__O[0] + _domain_[0], out_phi__O[1]: out_phi__O[1] + _domain_[1], out_phi__O[2] + interval_k_start:out_phi__O[2] + interval_k_end] = in_gamma[in_gamma__O[0]: in_gamma__O[0] + _domain_[0], in_gamma__O[1]: in_gamma__O[1] + _domain_[1], in_gamma__O[2] + interval_k_start:in_gamma__O[2] + interval_k_end] * (((((((-in_phi[in_phi__O[0] - 2: in_phi__O[0] + _domain_[0] - 2, in_phi__O[1]: in_phi__O[1] + _domain_[1], in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end]) + (16.0 * in_phi[in_phi__O[0] - 1: in_phi__O[0] + _domain_[0] - 1, in_phi__O[1]: in_phi__O[1] + _domain_[1], in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end])) - (30.0 * in_phi[in_phi__O[0]: in_phi__O[0] + _domain_[0], in_phi__O[1]: in_phi__O[1] + _domain_[1], in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end])) + (16.0 * in_phi[in_phi__O[0] + 1: in_phi__O[0] + _domain_[0] + 1, in_phi__O[1]: in_phi__O[1] + _domain_[1], in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end])) - in_phi[in_phi__O[0] + 2: in_phi__O[0] + _domain_[
                                                                                                                                                                                                                                                                                                                                                      0] + 2, in_phi__O[1]: in_phi__O[1] + _domain_[1], in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end]) / ((12.0 * dx) * dx)) + ((((((-in_phi[in_phi__O[0]: in_phi__O[0] + _domain_[0], in_phi__O[1] - 2: in_phi__O[1] + _domain_[1] - 2, in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end]) + (16.0 * in_phi[in_phi__O[0]: in_phi__O[0] + _domain_[0], in_phi__O[1] - 1: in_phi__O[1] + _domain_[1] - 1, in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end])) - (30.0 * in_phi[in_phi__O[0]: in_phi__O[0] + _domain_[0], in_phi__O[1]: in_phi__O[1] + _domain_[1], in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end])) + (16.0 * in_phi[in_phi__O[0]: in_phi__O[0] + _domain_[0], in_phi__O[1] + 1: in_phi__O[1] + _domain_[1] + 1, in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end])) - in_phi[in_phi__O[0]: in_phi__O[0] + _domain_[0], in_phi__O[1] + 2: in_phi__O[1] + _domain_[1] + 2, in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end]) / ((12.0 * dy) * dy)))
