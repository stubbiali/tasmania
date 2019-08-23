
import time

import numpy as np
from numpy import dtype


from gridtools import Boundary, FieldInfo, ParameterInfo, StencilObject


class _stencil_defs_numpy(StencilObject):

    _gt_backend_ = "numpy"

    _gt_source_ = {}

    _gt_field_info_ = {'in_phi': FieldInfo(boundary=Boundary(((3, 3), (3, 3), (0, 0))), dtype=dtype('float64')), 'in_gamma': FieldInfo(boundary=Boundary(
        ((0, 0), (0, 0), (0, 0))), dtype=dtype('float64')), 'out_phi': FieldInfo(boundary=Boundary(((0, 0), (0, 0), (0, 0))), dtype=dtype('float64'))}

    _gt_parameter_info_ = {'dx': ParameterInfo(dtype=dtype('float64')), 'dy': ParameterInfo(dtype=dtype('float64'))}

    _gt_constants_ = {}

    _gt_default_domain_ = None

    _gt_default_origin_ = None

    _gt_options_ = {'name': '_stencil_defs_numpy', 'module': 'tasmania.python.dwarfs.horizontal_hyperdiffusion',
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

        # Allocation of temporary fields
        __gt_stage_laplacian_10_8_lap_y = np.empty((_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8_lap_y__O = (1, 1, 0)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi = np.empty(
            (_domain_[0] + 6, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O = (3, 2, 0)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx = np.empty(
            (_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O = (1, 1, 0)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi = np.empty(
            (_domain_[0], _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O = (0, 1, 0)
        lap2 = np.empty((_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        lap2__O = (0, 0, 0)
        __gt_stage_laplacian_9_8_phi = np.empty((_domain_[0] + 6, _domain_[1] + 6, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8_phi__O = (3, 3, 0)
        __gt_stage_laplacian_10_8_dx = np.empty((_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8_dx__O = (1, 1, 0)
        lap1 = np.empty((_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        lap1__O = (1, 1, 0)
        __gt_stage_laplacian_11_8_lap_x = np.empty((_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8_lap_x__O = (0, 0, 0)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx = np.empty(
            (_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O = (2, 2, 0)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy = np.empty(
            (_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O = (0, 0, 0)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi = np.empty(
            (_domain_[0] + 4, _domain_[1] + 6, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O = (2, 3, 0)
        __gt_stage_laplacian_9_8_lap_x = np.empty((_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8_lap_x__O = (2, 2, 0)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap = np.empty(
            (_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O = (1, 1, 0)
        __gt_stage_laplacian_9_8_dy = np.empty((_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8_dy__O = (2, 2, 0)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx = np.empty(
            (_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O = (0, 0, 0)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi = np.empty(
            (_domain_[0] + 2, _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O = (1, 0, 0)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap = np.empty(
            (_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O = (0, 0, 0)
        __gt_stage_laplacian_11_8_dx = np.empty((_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8_dx__O = (0, 0, 0)
        __gt_stage_laplacian_11_8_lap_y = np.empty((_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8_lap_y__O = (0, 0, 0)
        __gt_stage_laplacian_9_8_lap = np.empty((_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8_lap__O = (2, 2, 0)
        __gt_stage_laplacian_10_8_lap = np.empty((_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8_lap__O = (1, 1, 0)
        __gt_stage_laplacian_10_8_lap_x = np.empty((_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8_lap_x__O = (1, 1, 0)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap = np.empty(
            (_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O = (1, 1, 0)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy = np.empty(
            (_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O = (2, 2, 0)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy = np.empty(
            (_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O = (1, 1, 0)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap = np.empty(
            (_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O = (0, 0, 0)
        __gt_stage_laplacian_11_8_phi = np.empty((_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8_phi__O = (1, 1, 0)
        __gt_stage_laplacian_11_8_dy = np.empty((_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8_dy__O = (0, 0, 0)
        __gt_stage_laplacian_10_8_phi = np.empty((_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8_phi__O = (2, 2, 0)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi = np.empty(
            (_domain_[0] + 4, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O = (2, 1, 0)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap = np.empty(
            (_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O = (2, 2, 0)
        __gt_stage_laplacian_11_8_lap = np.empty((_domain_[0], _domain_[1], _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_11_8_lap__O = (0, 0, 0)
        __gt_stage_laplacian_9_8_dx = np.empty((_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8_dx__O = (2, 2, 0)
        __gt_stage_laplacian_9_8_lap_y = np.empty((_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8_lap_y__O = (2, 2, 0)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi = np.empty(
            (_domain_[0] + 2, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O = (1, 2, 0)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap = np.empty(
            (_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O = (2, 2, 0)
        __gt_stage_laplacian_10_8_dy = np.empty((_domain_[0] + 2, _domain_[1] + 2, _domain_[2]), dtype=np.float64)
        __gt_stage_laplacian_10_8_dy__O = (1, 1, 0)
        lap0 = np.empty((_domain_[0] + 4, _domain_[1] + 4, _domain_[2]), dtype=np.float64)
        lap0__O = (2, 2, 0)

        # Computations
        # stage__84:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8_dx[__gt_stage_laplacian_9_8_dx__O[0] - 2: __gt_stage_laplacian_9_8_dx__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_dx__O[1] -
                                    2: __gt_stage_laplacian_9_8_dx__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_dx__O[2] + interval_k_start:__gt_stage_laplacian_9_8_dx__O[2] + interval_k_end] = dx

        # stage__87:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8_dy[__gt_stage_laplacian_9_8_dy__O[0] - 2: __gt_stage_laplacian_9_8_dy__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_dy__O[1] -
                                    2: __gt_stage_laplacian_9_8_dy__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_dy__O[2] + interval_k_start:__gt_stage_laplacian_9_8_dy__O[2] + interval_k_end] = dy

        # stage__90:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8_phi[__gt_stage_laplacian_9_8_phi__O[0] - 3: __gt_stage_laplacian_9_8_phi__O[0] + _domain_[0] + 3, __gt_stage_laplacian_9_8_phi__O[1] - 3: __gt_stage_laplacian_9_8_phi__O[1] + _domain_[1] + 3, __gt_stage_laplacian_9_8_phi__O[2] +
                                     interval_k_start:__gt_stage_laplacian_9_8_phi__O[2] + interval_k_end] = in_phi[in_phi__O[0] - 3: in_phi__O[0] + _domain_[0] + 3, in_phi__O[1] - 3: in_phi__O[1] + _domain_[1] + 3, in_phi__O[2] + interval_k_start:in_phi__O[2] + interval_k_end]

        # stage__93:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx[__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[2] +
                                                               interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_end] = __gt_stage_laplacian_9_8_dx[__gt_stage_laplacian_9_8_dx__O[0] - 2: __gt_stage_laplacian_9_8_dx__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_dx__O[1] - 2: __gt_stage_laplacian_9_8_dx__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_dx__O[2] + interval_k_start:__gt_stage_laplacian_9_8_dx__O[2] + interval_k_end]

        # stage__96:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[0] - 3: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] + 3, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[2] +
                                                                interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_end] = __gt_stage_laplacian_9_8_phi[__gt_stage_laplacian_9_8_phi__O[0] - 3: __gt_stage_laplacian_9_8_phi__O[0] + _domain_[0] + 3, __gt_stage_laplacian_9_8_phi__O[1] - 2: __gt_stage_laplacian_9_8_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_phi__O[2] + interval_k_start:__gt_stage_laplacian_9_8_phi__O[2] + interval_k_end]

        # stage__99:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap[__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_end] = ((__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[0] - 3: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_end] - (2.0 * __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            2] + interval_k_end])) + __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[0] - 1: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] + 3, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_end]) / (__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx[__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_end] * __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx[__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_end])

        # stage__102:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8_lap_x[__gt_stage_laplacian_9_8_lap_x__O[0] - 2: __gt_stage_laplacian_9_8_lap_x__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_lap_x__O[1] - 2: __gt_stage_laplacian_9_8_lap_x__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_lap_x__O[2] + interval_k_start:__gt_stage_laplacian_9_8_lap_x__O[2] + interval_k_end] = __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap[__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[0] -
                                                                                                                                                                                                                                                                                                                                                                                                                    2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_end]

        # stage__105:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy[__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[2] +
                                                               interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_end] = __gt_stage_laplacian_9_8_dy[__gt_stage_laplacian_9_8_dy__O[0] - 2: __gt_stage_laplacian_9_8_dy__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_dy__O[1] - 2: __gt_stage_laplacian_9_8_dy__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_dy__O[2] + interval_k_start:__gt_stage_laplacian_9_8_dy__O[2] + interval_k_end]

        # stage__108:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[1] - 3: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] + 3, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[2] +
                                                                interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_end] = __gt_stage_laplacian_9_8_phi[__gt_stage_laplacian_9_8_phi__O[0] - 2: __gt_stage_laplacian_9_8_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_phi__O[1] - 3: __gt_stage_laplacian_9_8_phi__O[1] + _domain_[1] + 3, __gt_stage_laplacian_9_8_phi__O[2] + interval_k_start:__gt_stage_laplacian_9_8_phi__O[2] + interval_k_end]

        # stage__111:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap[__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_end] = ((__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[1] - 3: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_end] - (2.0 * __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            2] + interval_k_end])) + __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[1] - 1: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] + 3, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_end]) / (__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy[__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_end] * __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy[__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[0] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_end])

        # stage__114:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8_lap_y[__gt_stage_laplacian_9_8_lap_y__O[0] - 2: __gt_stage_laplacian_9_8_lap_y__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_lap_y__O[1] - 2: __gt_stage_laplacian_9_8_lap_y__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_lap_y__O[2] + interval_k_start:__gt_stage_laplacian_9_8_lap_y__O[2] + interval_k_end] = __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap[__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[0] -
                                                                                                                                                                                                                                                                                                                                                                                                                    2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[1] - 2: __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_9_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_end]

        # stage__117:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_9_8_lap[__gt_stage_laplacian_9_8_lap__O[0] - 2: __gt_stage_laplacian_9_8_lap__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_lap__O[1] - 2: __gt_stage_laplacian_9_8_lap__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_lap__O[2] + interval_k_start:__gt_stage_laplacian_9_8_lap__O[2] + interval_k_end] = __gt_stage_laplacian_9_8_lap_x[__gt_stage_laplacian_9_8_lap_x__O[0] - 2: __gt_stage_laplacian_9_8_lap_x__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_lap_x__O[1] -
                                                                                                                                                                                                                                                                                                                                                                             2: __gt_stage_laplacian_9_8_lap_x__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_lap_x__O[2] + interval_k_start:__gt_stage_laplacian_9_8_lap_x__O[2] + interval_k_end] + __gt_stage_laplacian_9_8_lap_y[__gt_stage_laplacian_9_8_lap_y__O[0] - 2: __gt_stage_laplacian_9_8_lap_y__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_lap_y__O[1] - 2: __gt_stage_laplacian_9_8_lap_y__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_lap_y__O[2] + interval_k_start:__gt_stage_laplacian_9_8_lap_y__O[2] + interval_k_end]

        # stage__120:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        lap0[lap0__O[0] - 2: lap0__O[0] + _domain_[0] + 2, lap0__O[1] - 2: lap0__O[1] + _domain_[1] + 2, lap0__O[2] + interval_k_start:lap0__O[2] + interval_k_end] = __gt_stage_laplacian_9_8_lap[__gt_stage_laplacian_9_8_lap__O[0] -
                                                                                                                                                                                                   2: __gt_stage_laplacian_9_8_lap__O[0] + _domain_[0] + 2, __gt_stage_laplacian_9_8_lap__O[1] - 2: __gt_stage_laplacian_9_8_lap__O[1] + _domain_[1] + 2, __gt_stage_laplacian_9_8_lap__O[2] + interval_k_start:__gt_stage_laplacian_9_8_lap__O[2] + interval_k_end]

        # stage__123:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8_dx[__gt_stage_laplacian_10_8_dx__O[0] - 1: __gt_stage_laplacian_10_8_dx__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_dx__O[1] -
                                     1: __gt_stage_laplacian_10_8_dx__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_dx__O[2] + interval_k_start:__gt_stage_laplacian_10_8_dx__O[2] + interval_k_end] = dx

        # stage__126:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8_dy[__gt_stage_laplacian_10_8_dy__O[0] - 1: __gt_stage_laplacian_10_8_dy__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_dy__O[1] -
                                     1: __gt_stage_laplacian_10_8_dy__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_dy__O[2] + interval_k_start:__gt_stage_laplacian_10_8_dy__O[2] + interval_k_end] = dy

        # stage__129:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8_phi[__gt_stage_laplacian_10_8_phi__O[0] - 2: __gt_stage_laplacian_10_8_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_10_8_phi__O[1] - 2: __gt_stage_laplacian_10_8_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_10_8_phi__O[2] +
                                      interval_k_start:__gt_stage_laplacian_10_8_phi__O[2] + interval_k_end] = lap0[lap0__O[0] - 2: lap0__O[0] + _domain_[0] + 2, lap0__O[1] - 2: lap0__O[1] + _domain_[1] + 2, lap0__O[2] + interval_k_start:lap0__O[2] + interval_k_end]

        # stage__132:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx[__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[2] +
                                                                interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_end] = __gt_stage_laplacian_10_8_dx[__gt_stage_laplacian_10_8_dx__O[0] - 1: __gt_stage_laplacian_10_8_dx__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_dx__O[1] - 1: __gt_stage_laplacian_10_8_dx__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_dx__O[2] + interval_k_start:__gt_stage_laplacian_10_8_dx__O[2] + interval_k_end]

        # stage__135:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[0] - 2: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[2] +
                                                                 interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_end] = __gt_stage_laplacian_10_8_phi[__gt_stage_laplacian_10_8_phi__O[0] - 2: __gt_stage_laplacian_10_8_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_10_8_phi__O[1] - 1: __gt_stage_laplacian_10_8_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_phi__O[2] + interval_k_start:__gt_stage_laplacian_10_8_phi__O[2] + interval_k_end]

        # stage__138:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap[__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_end] = ((__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[0] - 2: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0], __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_end] - (2.0 * __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   2] + interval_k_end])) + __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[0]: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] + 2, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_end]) / (__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx[__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_end] * __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx[__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_end])

        # stage__141:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8_lap_x[__gt_stage_laplacian_10_8_lap_x__O[0] - 1: __gt_stage_laplacian_10_8_lap_x__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_lap_x__O[1] - 1: __gt_stage_laplacian_10_8_lap_x__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_lap_x__O[2] + interval_k_start:__gt_stage_laplacian_10_8_lap_x__O[2] + interval_k_end] = __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap[__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[0] -
                                                                                                                                                                                                                                                                                                                                                                                                                            1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_end]

        # stage__144:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy[__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[2] +
                                                                interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_end] = __gt_stage_laplacian_10_8_dy[__gt_stage_laplacian_10_8_dy__O[0] - 1: __gt_stage_laplacian_10_8_dy__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_dy__O[1] - 1: __gt_stage_laplacian_10_8_dy__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_dy__O[2] + interval_k_start:__gt_stage_laplacian_10_8_dy__O[2] + interval_k_end]

        # stage__147:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[1] - 2: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[2] +
                                                                 interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_end] = __gt_stage_laplacian_10_8_phi[__gt_stage_laplacian_10_8_phi__O[0] - 1: __gt_stage_laplacian_10_8_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_phi__O[1] - 2: __gt_stage_laplacian_10_8_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_10_8_phi__O[2] + interval_k_start:__gt_stage_laplacian_10_8_phi__O[2] + interval_k_end]

        # stage__150:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap[__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_end] = ((__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[1] - 2: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1], __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_end] - (2.0 * __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   2] + interval_k_end])) + __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[1]: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] + 2, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_end]) / (__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy[__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_end] * __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy[__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[0] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_end])

        # stage__153:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8_lap_y[__gt_stage_laplacian_10_8_lap_y__O[0] - 1: __gt_stage_laplacian_10_8_lap_y__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_lap_y__O[1] - 1: __gt_stage_laplacian_10_8_lap_y__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_lap_y__O[2] + interval_k_start:__gt_stage_laplacian_10_8_lap_y__O[2] + interval_k_end] = __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap[__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[0] -
                                                                                                                                                                                                                                                                                                                                                                                                                            1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[1] - 1: __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_10_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_end]

        # stage__156:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_10_8_lap[__gt_stage_laplacian_10_8_lap__O[0] - 1: __gt_stage_laplacian_10_8_lap__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_lap__O[1] - 1: __gt_stage_laplacian_10_8_lap__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_lap__O[2] + interval_k_start:__gt_stage_laplacian_10_8_lap__O[2] + interval_k_end] = __gt_stage_laplacian_10_8_lap_x[__gt_stage_laplacian_10_8_lap_x__O[0] - 1: __gt_stage_laplacian_10_8_lap_x__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_lap_x__O[1] -
                                                                                                                                                                                                                                                                                                                                                                                     1: __gt_stage_laplacian_10_8_lap_x__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_lap_x__O[2] + interval_k_start:__gt_stage_laplacian_10_8_lap_x__O[2] + interval_k_end] + __gt_stage_laplacian_10_8_lap_y[__gt_stage_laplacian_10_8_lap_y__O[0] - 1: __gt_stage_laplacian_10_8_lap_y__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_lap_y__O[1] - 1: __gt_stage_laplacian_10_8_lap_y__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_lap_y__O[2] + interval_k_start:__gt_stage_laplacian_10_8_lap_y__O[2] + interval_k_end]

        # stage__159:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        lap1[lap1__O[0] - 1: lap1__O[0] + _domain_[0] + 1, lap1__O[1] - 1: lap1__O[1] + _domain_[1] + 1, lap1__O[2] + interval_k_start:lap1__O[2] + interval_k_end] = __gt_stage_laplacian_10_8_lap[__gt_stage_laplacian_10_8_lap__O[0] -
                                                                                                                                                                                                    1: __gt_stage_laplacian_10_8_lap__O[0] + _domain_[0] + 1, __gt_stage_laplacian_10_8_lap__O[1] - 1: __gt_stage_laplacian_10_8_lap__O[1] + _domain_[1] + 1, __gt_stage_laplacian_10_8_lap__O[2] + interval_k_start:__gt_stage_laplacian_10_8_lap__O[2] + interval_k_end]

        # stage__162:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8_dx[__gt_stage_laplacian_11_8_dx__O[0]: __gt_stage_laplacian_11_8_dx__O[0] + _domain_[0], __gt_stage_laplacian_11_8_dx__O[1]                                     : __gt_stage_laplacian_11_8_dx__O[1] + _domain_[1], __gt_stage_laplacian_11_8_dx__O[2] + interval_k_start:__gt_stage_laplacian_11_8_dx__O[2] + interval_k_end] = dx

        # stage__165:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8_dy[__gt_stage_laplacian_11_8_dy__O[0]: __gt_stage_laplacian_11_8_dy__O[0] + _domain_[0], __gt_stage_laplacian_11_8_dy__O[1]                                     : __gt_stage_laplacian_11_8_dy__O[1] + _domain_[1], __gt_stage_laplacian_11_8_dy__O[2] + interval_k_start:__gt_stage_laplacian_11_8_dy__O[2] + interval_k_end] = dy

        # stage__168:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8_phi[__gt_stage_laplacian_11_8_phi__O[0] - 1: __gt_stage_laplacian_11_8_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_11_8_phi__O[1] - 1: __gt_stage_laplacian_11_8_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_11_8_phi__O[2] +
                                      interval_k_start:__gt_stage_laplacian_11_8_phi__O[2] + interval_k_end] = lap1[lap1__O[0] - 1: lap1__O[0] + _domain_[0] + 1, lap1__O[1] - 1: lap1__O[1] + _domain_[1] + 1, lap1__O[2] + interval_k_start:lap1__O[2] + interval_k_end]

        # stage__171:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx[__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[2] +
                                                                interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_end] = __gt_stage_laplacian_11_8_dx[__gt_stage_laplacian_11_8_dx__O[0]: __gt_stage_laplacian_11_8_dx__O[0] + _domain_[0], __gt_stage_laplacian_11_8_dx__O[1]: __gt_stage_laplacian_11_8_dx__O[1] + _domain_[1], __gt_stage_laplacian_11_8_dx__O[2] + interval_k_start:__gt_stage_laplacian_11_8_dx__O[2] + interval_k_end]

        # stage__174:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[0] - 1: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[2] +
                                                                 interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_end] = __gt_stage_laplacian_11_8_phi[__gt_stage_laplacian_11_8_phi__O[0] - 1: __gt_stage_laplacian_11_8_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_11_8_phi__O[1]: __gt_stage_laplacian_11_8_phi__O[1] + _domain_[1], __gt_stage_laplacian_11_8_phi__O[2] + interval_k_start:__gt_stage_laplacian_11_8_phi__O[2] + interval_k_end]

        # stage__177:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap[__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_end] = ((__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[0] - 1: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] - 1, __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_end] - (2.0 * __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   2] + interval_k_end])) + __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi[__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[0] + 1: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[0] + _domain_[0] + 1, __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_phi__O[2] + interval_k_end]) / (__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx[__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_end] * __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx[__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_dx__O[2] + interval_k_end])

        # stage__180:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8_lap_x[__gt_stage_laplacian_11_8_lap_x__O[0]: __gt_stage_laplacian_11_8_lap_x__O[0] + _domain_[0], __gt_stage_laplacian_11_8_lap_x__O[1]: __gt_stage_laplacian_11_8_lap_x__O[1] + _domain_[1], __gt_stage_laplacian_11_8_lap_x__O[2] + interval_k_start:__gt_stage_laplacian_11_8_lap_x__O[2] + interval_k_end] = __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap[__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[0]                                                                                                                                                                                                                                                                                                                                                                                                            : __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_x_2_9_lap__O[2] + interval_k_end]

        # stage__183:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy[__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[2] +
                                                                interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_end] = __gt_stage_laplacian_11_8_dy[__gt_stage_laplacian_11_8_dy__O[0]: __gt_stage_laplacian_11_8_dy__O[0] + _domain_[0], __gt_stage_laplacian_11_8_dy__O[1]: __gt_stage_laplacian_11_8_dy__O[1] + _domain_[1], __gt_stage_laplacian_11_8_dy__O[2] + interval_k_start:__gt_stage_laplacian_11_8_dy__O[2] + interval_k_end]

        # stage__186:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[1] - 1: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[2] +
                                                                 interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_end] = __gt_stage_laplacian_11_8_phi[__gt_stage_laplacian_11_8_phi__O[0]: __gt_stage_laplacian_11_8_phi__O[0] + _domain_[0], __gt_stage_laplacian_11_8_phi__O[1] - 1: __gt_stage_laplacian_11_8_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_11_8_phi__O[2] + interval_k_start:__gt_stage_laplacian_11_8_phi__O[2] + interval_k_end]

        # stage__189:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap[__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_end] = ((__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[1] - 1: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] - 1, __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_end] - (2.0 * __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                   2] + interval_k_end])) + __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi[__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[1] + 1: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[1] + _domain_[1] + 1, __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_phi__O[2] + interval_k_end]) / (__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy[__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_end] * __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy[__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[0]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_dy__O[2] + interval_k_end])

        # stage__192:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8_lap_y[__gt_stage_laplacian_11_8_lap_y__O[0]: __gt_stage_laplacian_11_8_lap_y__O[0] + _domain_[0], __gt_stage_laplacian_11_8_lap_y__O[1]: __gt_stage_laplacian_11_8_lap_y__O[1] + _domain_[1], __gt_stage_laplacian_11_8_lap_y__O[2] + interval_k_start:__gt_stage_laplacian_11_8_lap_y__O[2] + interval_k_end] = __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap[__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[0]                                                                                                                                                                                                                                                                                                                                                                                                            : __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[0] + _domain_[0], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[1]: __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[1] + _domain_[1], __gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_start:__gt_stage_laplacian_11_8___gt_stage_laplacian_y_3_9_lap__O[2] + interval_k_end]

        # stage__195:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        __gt_stage_laplacian_11_8_lap[__gt_stage_laplacian_11_8_lap__O[0]: __gt_stage_laplacian_11_8_lap__O[0] + _domain_[0], __gt_stage_laplacian_11_8_lap__O[1]: __gt_stage_laplacian_11_8_lap__O[1] + _domain_[1], __gt_stage_laplacian_11_8_lap__O[2] + interval_k_start:__gt_stage_laplacian_11_8_lap__O[2] + interval_k_end] = __gt_stage_laplacian_11_8_lap_x[__gt_stage_laplacian_11_8_lap_x__O[0]: __gt_stage_laplacian_11_8_lap_x__O[0] + _domain_[0], __gt_stage_laplacian_11_8_lap_x__O[1]                                                                                                                                                                                                                                                                                                                                                                     : __gt_stage_laplacian_11_8_lap_x__O[1] + _domain_[1], __gt_stage_laplacian_11_8_lap_x__O[2] + interval_k_start:__gt_stage_laplacian_11_8_lap_x__O[2] + interval_k_end] + __gt_stage_laplacian_11_8_lap_y[__gt_stage_laplacian_11_8_lap_y__O[0]: __gt_stage_laplacian_11_8_lap_y__O[0] + _domain_[0], __gt_stage_laplacian_11_8_lap_y__O[1]: __gt_stage_laplacian_11_8_lap_y__O[1] + _domain_[1], __gt_stage_laplacian_11_8_lap_y__O[2] + interval_k_start:__gt_stage_laplacian_11_8_lap_y__O[2] + interval_k_end]

        # stage__198:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        lap2[lap2__O[0]: lap2__O[0] + _domain_[0], lap2__O[1]: lap2__O[1] + _domain_[1], lap2__O[2] + interval_k_start:lap2__O[2] + interval_k_end] = __gt_stage_laplacian_11_8_lap[__gt_stage_laplacian_11_8_lap__O[0]: __gt_stage_laplacian_11_8_lap__O[0] +
                                                                                                                                                                                    _domain_[0], __gt_stage_laplacian_11_8_lap__O[1]: __gt_stage_laplacian_11_8_lap__O[1] + _domain_[1], __gt_stage_laplacian_11_8_lap__O[2] + interval_k_start:__gt_stage_laplacian_11_8_lap__O[2] + interval_k_end]

        # stage__201:
        interval_k_start = 0
        interval_k_end = _domain_[2]
        out_phi[out_phi__O[0]: out_phi__O[0] + _domain_[0], out_phi__O[1]: out_phi__O[1] + _domain_[1], out_phi__O[2] + interval_k_start:out_phi__O[2] + interval_k_end] = in_gamma[in_gamma__O[0]: in_gamma__O[0] + _domain_[0], in_gamma__O[1]                                                                                                                                                                                    : in_gamma__O[1] + _domain_[1], in_gamma__O[2] + interval_k_start:in_gamma__O[2] + interval_k_end] * lap2[lap2__O[0]: lap2__O[0] + _domain_[0], lap2__O[1]: lap2__O[1] + _domain_[1], lap2__O[2] + interval_k_start:lap2__O[2] + interval_k_end]
