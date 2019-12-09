# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2019, ETH Zurich
# All rights reserved.
#
# This file is part of the Tasmania project. Tasmania is free software:
# you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation,
# either version 3 of the License, or any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.
#
# SPDX-License-Identifier: GPL-3.0-or-later
#
import numpy as np
from sympl import (
    DiagnosticComponent,
    DiagnosticComponentComposite as SymplDiagnosticComponentComposite,
    TendencyComponent,
    TendencyComponentComposite,
    ImplicitTendencyComponent,
    ImplicitTendencyComponentComposite,
)

from tasmania.python.framework.composite import (
    DiagnosticComponentComposite as TasmaniaDiagnosticComponentComposite,
)
from tasmania.python.framework.concurrent_coupling import ConcurrentCoupling
from tasmania.python.framework.tendency_steppers import TendencyStepper
from tasmania.python.utils.dict_utils import DataArrayDictOperator
from tasmania.python.utils.framework_utils import (
    check_properties_compatibility,
    get_input_properties,
    get_output_properties,
)


class ParallelSplitting:
    """
    Callable class which integrates a bundle of physical processes pursuing
    the parallel splitting strategy.

    Attributes
    ----------
    input_properties : dict[str, dict]
        Dictionary whose keys are strings denoting variables which
        should be present in the input model dictionary representing
        the current state, and whose values are dictionaries specifying
        fundamental properties (dims, units) of those variables.
    provisional_input_properties : dict[str, dict]
        Dictionary whose keys are strings denoting variables which
        should be present in the input model dictionary representing
        the provisional state, and whose values are dictionaries specifying
        fundamental properties (dims, units) of those variables.
    output_properties : dict[str, dict]
        Dictionary whose keys are strings denoting variables which
        will be present in the input model dictionary representing
        the current state when the call operator returns, and
        whose values are dictionaries specifying fundamental properties
        (dims, units) of those variables.
    provisional_output_properties : dict[str, dict]
        Dictionary whose keys are strings denoting variables which
        will be present in the input model dictionary representing
        the provisional state when the call operator returns, and
        whose values are dictionaries specifying fundamental properties
        (dims, units) of those variables.

    References
    ----------
    Donahue, A. S., and P. M. Caldwell. (2018). \
        Impact of physics parameterization ordering in a global atmosphere model. \
        *Journal of Advances in Modeling earth Systems*, *10*:481-499.
    """

    allowed_diagnostic_type = (
        DiagnosticComponent,
        SymplDiagnosticComponentComposite,
        TasmaniaDiagnosticComponentComposite,
    )
    allowed_tendency_type = (
        TendencyComponent,
        TendencyComponentComposite,
        ImplicitTendencyComponent,
        ImplicitTendencyComponentComposite,
        ConcurrentCoupling,
    )
    allowed_component_type = allowed_diagnostic_type + allowed_tendency_type

    def __init__(
        self,
        *args,
        execution_policy="serial",
        retrieve_diagnostics_from_provisional_state=False,
        gt_powered=False,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=np.float64,
        rebuild=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        *args : dict
            Dictionaries containing the components to wrap and specifying
            fundamental properties (time_integrator, substeps) of those processes.
            Particularly:

                * 'component' is an instance of

                        - :class:`sympl.DiagnosticComponent`
                        - :class:`sympl.DiagnosticComponentComposite`
                        - :class:`sympl.TendencyComponent`,
                        - :class:`sympl.TendencyComponentComposite`,
                        - :class:`sympl.ImplicitTendencyComponent`,
                        - :class:`sympl.ImplicitTendencyComponentComposite`, or
                        - :class:`tasmania.ConcurrentCoupling`

                    representing the process;

                * if 'component' is an instance of

                        - :class:`sympl.TendencyComponent`,
                        - :class:`sympl.TendencyComponentComposite`,
                        - :class:`sympl.ImplicitTendencyComponent`,
                        - :class:`sympl.ImplicitTendencyComponentComposite`, or
                        - :class:`tasmania.ConcurrentCoupling`,

                    'time_integrator' is a string specifying the scheme to
                    integrate the process forward in time. Available options:

                        - 'forward_euler', for the forward Euler scheme;
                        - 'rk2', for the two-stage second-order Runge-Kutta (RK) scheme;
                        - 'rk3ws', for the three-stage RK scheme as used in the
                            `COSMO model <http://www.cosmo-model.org>`_; this method is
                            nominally second-order, and third-order for linear problems.

                * if 'component' is a

                        - :class:`sympl.TendencyComponent`,
                        - :class:`sympl.TendencyComponentComposite`,
                        - :class:`sympl.ImplicitTendencyComponent`,
                        - :class:`sympl.ImplicitTendencyComponentComposite`, or
                        - :class:`tasmania.ConcurrentCoupling`,

                    'gt_powered' specifies if all the time-intensive math
                    operations performed inside 'time_integrator' should harness
                    GT4Py. Defaults to `gt_powered` (see later).

                * if 'component' is a

                        - :class:`sympl.TendencyComponent`,
                        - :class:`sympl.TendencyComponentComposite`,
                        - :class:`sympl.ImplicitTendencyComponent`,
                        - :class:`sympl.ImplicitTendencyComponentComposite`, or
                        - :class:`tasmania.ConcurrentCoupling`,

                    'time_integrator_kwargs' is a dictionary of configuration
                    options for 'time_integrator'. The dictionary may include
                    the following keys:

                        - backend (str): The GT4Py backend;
                        - backend_opts (dict): Dictionary of backend-specific options;
                        - build_info (dict): Dictionary of building options;
                        - dtype (data-type): Data type of the storages;
                        - exec_info (dict): Dictionary which will store statistics
                            and diagnostics gathered at run time;
                        - default_origin (tuple): Storage default origin;
                        - rebuild (bool): `True` to trigger the stencils compilation
                            at any class instantiation, `False` to rely on the caching
                            mechanism implemented by GT4Py.

                * if 'component' is either an instance of or wraps objects of class

                        - :class:`tasmania.TendencyComponent`,
                        - :class:`tasmania.ImplicitTendencyComponent`, or
                        - :class:`tasmania.ConcurrentCoupling`,

                    'enforce_horizontal_boundary' is either `True` if the
                    boundary conditions should be enforced after each stage of
                    the time integrator, or `False` not to apply the boundary
                    constraints at all. Defaults to `False`;

                * if 'component' is a

                        - :class:`sympl.TendencyComponent`,
                        - :class:`sympl.TendencyComponentComposite`,
                        - :class:`sympl.ImplicitTendencyComponent`,
                        - :class:`sympl.ImplicitTendencyComponentComposite`, or
                        - :class:`tasmania.ConcurrentCoupling`,

                    'substeps' represents the number of substeps to carry out
                    to integrate the process. Defaults to 1.

                * if 'component' is a

                        - :class:`sympl.TendencyComponent`,
                        - :class:`sympl.TendencyComponentComposite`,
                        - :class:`sympl.ImplicitTendencyComponent`,
                        - :class:`sympl.ImplicitTendencyComponentComposite`, or
                        - :class:`tasmania.ConcurrentCoupling`,

                    'add_diagnostics_to_provisional_input' says whether the computed
                    diagnostics should be added to the input or provisional state.

        execution_policy : `str`, optional
            String specifying the runtime mode in which parameterizations
            should be invoked. Either:

                * 'serial' (default), to run the physical packages sequentially,
                    and add diagnostics to the current state incrementally;
                * 'as_parallel', to run the physical packages *as* we
                    were in a parallel region where each package may be
                    assigned to a different thread/process; in other terms,
                    we do not rely on the order in which parameterizations
                    are passed to this object, and diagnostics are added
                    to the current state in a single step just before returning.

        retrieve_diagnostics_from_provisional_state : `bool`, optional
            `True` (respectively, `False`) to feed the
            :class:`sympl.DiagnosticComponent` objects with the provisional
            (resp., current) state, and add the so-retrieved diagnostics
            to the provisional (resp., current) state dictionary.
            Defaults to `False`.
        gt_powered : `bool`, optional
            `True` to perform additions and subtractions using GT4Py (leveraging
            field versioning), `False` to perform the operations in plain Python.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages passed to the stencil.
        exec_info : `dict`, optional
            Dictionary which will store statistics and diagnostics gathered at run time.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        **kwargs:
            Catch-all for unused keyword arguments.
        """
        self._component_list = []
        self._substeps = []

        for process in args:
            try:
                bare_component = process["component"]
            except KeyError:
                msg = "Missing mandatory key ''component'' in one item of ''processes''."
                raise KeyError(msg)

            assert isinstance(
                bare_component, self.__class__.allowed_component_type
            ), "''component'' value should be either a {}.".format(
                ", ".join(str(ctype) for ctype in self.__class__.allowed_component_type)
            )

            if isinstance(bare_component, self.__class__.allowed_diagnostic_type):
                self._component_list.append(bare_component)
                self._substeps.append(1)
            else:
                integrator = process.get("time_integrator", "forward_euler")
                enforce_hb = process.get("enforce_horizontal_boundary", False)
                integrator_gt_powered = process.get("gt_powered", gt_powered)
                integrator_kwargs = process.get(
                    "time_integrator_kwargs",
                    {"backend": "numpy", "dtype": np.float32, "rebuild": False},
                )

                self._component_list.append(
                    TendencyStepper.factory(
                        integrator,
                        bare_component,
                        enforce_horizontal_boundary=enforce_hb,
                        gt_powered=integrator_gt_powered,
                        **integrator_kwargs
                    )
                )

                substeps_ = process.get("substeps", 1)
                substeps = substeps_ if substeps_ > 0 else 1
                self._substeps.append(substeps)

        self._policy = execution_policy
        self._call = (
            self._call_serial if execution_policy == "serial" else self._call_asparallel
        )

        if (
            execution_policy == "as_parallel"
            and retrieve_diagnostics_from_provisional_state
        ):
            import warnings

            warnings.warn(
                "Argument retrieve_diagnostics_from_provisional_state "
                "only effective when execution policy set on 'serial'."
            )
            self._diagnostics_from_provisional = False
        else:
            self._diagnostics_from_provisional = (
                retrieve_diagnostics_from_provisional_state
            )

        # Set properties
        self.input_properties = self._init_input_properties()
        self.provisional_input_properties = self._init_provisional_input_properties()
        self.output_properties = self._init_output_properties()
        self.provisional_output_properties = self._init_provisional_output_properties()

        # Ensure that dimensions and units of the variables present
        # in both input_properties and output_properties are compatible
        # across the two dictionaries
        check_properties_compatibility(
            self.input_properties,
            self.output_properties,
            properties1_name="input_properties",
            properties2_name="output_properties",
        )

        # Ensure that dimensions and units of the variables present
        # in both provisional_input_properties and provisional_output_properties
        # are compatible across the two dictionaries
        check_properties_compatibility(
            self.provisional_input_properties,
            self.provisional_output_properties,
            properties1_name="provisional_input_properties",
            properties2_name="provisional_output_properties",
        )

        self._dict_op = DataArrayDictOperator(
            gt_powered,
            backend=backend,
            backend_opts=backend_opts,
            build_info=build_info,
            dtype=dtype,
            rebuild=rebuild,
        )

    def _init_input_properties(self):
        if not self._diagnostics_from_provisional:
            return get_input_properties(
                tuple(
                    {
                        "component": component,
                        "attribute_name": "input_properties",
                        "consider_diagnostics": self._policy == "serial",
                    }
                    for component in self.component_list
                )
            )
        else:
            return get_input_properties(
                tuple(
                    {
                        "component": component,
                        "attribute_name": "input_properties",
                        "consider_diagnostics": True,
                    }
                    for component in self.component_list
                    if not isinstance(component, self.__class__.allowed_diagnostic_type)
                )
            )

    def _init_provisional_input_properties(self):
        # We require that all prognostic variables affected by the
        # parameterizations are included in the provisional state
        return_dict = get_input_properties(
            tuple(
                {
                    "component": component,
                    "attribute_name": "output_properties",
                    "consider_diagnostics": False,
                }
                for component in self.component_list
                if not isinstance(component, self.__class__.allowed_diagnostic_type)
            )
        )

        if self._diagnostics_from_provisional:
            return_dict.update(
                get_input_properties(
                    tuple(
                        {
                            "component": component,
                            "attribute_name": "input_properties",
                            "consider_diagnostics": True,
                        }
                        for component in self.component_list
                        if isinstance(component, self.__class__.allowed_diagnostic_type)
                    )
                )
            )

        return return_dict

    def _init_output_properties(self):
        if not self._diagnostics_from_provisional:
            return get_output_properties(
                tuple(
                    {
                        "component": component,
                        "attribute_name": "input_properties",
                        "consider_diagnostics": True,
                    }
                    for component in self.component_list
                )
            )
        else:
            return get_output_properties(
                tuple(
                    {
                        "component": component,
                        "attribute_name": "input_properties",
                        "consider_diagnostics": True,
                    }
                    for component in self.component_list
                    if not isinstance(component, self.__class__.allowed_diagnostic_type)
                )
            )

    def _init_provisional_output_properties(self):
        return_dict = self.provisional_input_properties

        if self._diagnostics_from_provisional:
            return_dict.update(
                get_output_properties(
                    tuple(
                        {
                            "component": component,
                            "attribute_name": None,
                            "consider_diagnostics": True,
                        }
                        for component in self.component_list
                        if isinstance(component, self.__class__.allowed_diagnostic_type)
                    )
                )
            )

        return return_dict

    @property
    def component_list(self):
        """
        Return
        ------
        tuple :
            The wrapped components.
        """
        return tuple(self._component_list)

    def __call__(self, state, state_prv, timestep):
        """
        Advance the model state one timestep forward in time by pursuing
        the parallel splitting method.

        Parameters
        ----------
        state : dict[str, sympl.DataArray]
            The model state at the current time level, i.e., at the beginning
            of the current timestep.
        state_prv : dict[str, sympl.DataArray]
            A provisional model state. Ideally, this should be the state output
            by the dynamical core, i.e., the outcome of a one-timestep time
            integration which takes only the dynamical processes into consideration.
        timestep : datetime.timedelta
            The time step.

        Note
        ----
        `state` may be modified in-place with the diagnostics retrieved
        from `state` itself.
        `state_prv` is modified in-place with the temporary states provided
        by each process. In other words, when this method returns, `state_prv`
        will represent the state at the next time level.
        """
        # step the solution
        self._call(state, state_prv, timestep)

        # Ensure the provisional state is now defined at the next time level
        state_prv["time"] = state["time"] + timestep

    def _call_serial(self, state, state_prv, timestep):
        """ Process the components in 'serial' runtime mode. """
        for component, substeps in zip(
            self._component_list, self._substeps
        ):
            if not isinstance(component, self.__class__.allowed_diagnostic_type):
                diagnostics, state_tmp = component(state, timestep / substeps)

                if substeps > 1:
                    state_tmp.update(
                        {
                            key: value
                            for key, value in state.items()
                            if key not in state_tmp
                        }
                    )

                    for _ in range(1, substeps):
                        _, state_aux = component(state_tmp, timestep / substeps)
                        state_tmp.update(state_aux)

                self._dict_op.iaddsub(
                    state_prv,
                    state_tmp,
                    state,
                    field_properties=self.provisional_output_properties,
                )

                # name = 'mass_fraction_of_cloud_liquid_water_in_air'
                # if name in diagnostics:
                #     if diagnostics[name].values.max() > 0:
                #         import ipdb
                #         ipdb.set_trace()

                state.update(diagnostics)
            else:
                arg = state_prv if self._diagnostics_from_provisional else state

                try:
                    diagnostics = component(arg)
                except TypeError:
                    diagnostics = component(arg, timestep)

                arg.update(diagnostics)

    def _call_asparallel(self, state, state_prv, timestep):
        """ Process the components in 'as_parallel' runtime mode. """
        agg_diagnostics = {}

        for component, substeps in zip(self.component_list, self._substeps):
            if not isinstance(component, self.__class__.allowed_diagnostic_type):
                diagnostics, state_tmp = component(state, timestep / substeps)

                if substeps > 1:
                    state_tmp.update(
                        {
                            key: value
                            for key, value in state.items()
                            if key not in state_tmp
                        }
                    )

                    for _ in range(1, substeps):
                        _, state_aux = component(state_tmp, timestep / substeps)
                        state_tmp.update(state_aux)

                self._dict_op.iaddsub(
                    state_prv,
                    state_tmp,
                    state,
                    field_properties=self.provisional_output_properties,
                )
                agg_diagnostics.update(diagnostics)
            else:
                try:
                    diagnostics = component(state)
                except TypeError:
                    diagnostics = component(state, timestep)

                agg_diagnostics.update(diagnostics)

        state.update(agg_diagnostics)
