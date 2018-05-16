API Documentation
=================

Axis
----

.. autoclass:: tasmania.grids.axis.Axis
   :members:  


Dynamics
--------

Diagnostics
^^^^^^^^^^^

.. autoclass:: tasmania.dycore.diagnostic_isentropic.DiagnosticIsentropic
   :members:

Dynamical cores
^^^^^^^^^^^^^^^

.. autoclass:: tasmania.dycore.dycore.DynamicalCore
   :members:

.. autoclass:: tasmania.dycore.dycore_isentropic.DynamicalCoreIsentropic
   :members:

.. autoclass:: tasmania.dycore.dycore_isentropic_nonconservative.DynamicalCoreIsentropicNonconservative
   :members:

Lateral boundary conditions
^^^^^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tasmania.dycore.horizontal_boundary.HorizontalBoundary
   :members:

.. autoclass:: tasmania.dycore.horizontal_boundary_periodic.Periodic
   :members:

.. autoclass:: tasmania.dycore.horizontal_boundary_periodic.PeriodicXZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_boundary_periodic.PeriodicYZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_boundary_relaxed.Relaxed
   :members:

.. autoclass:: tasmania.dycore.horizontal_boundary_relaxed.RelaxedXZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_boundary_relaxed.RelaxedYZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_boundary_relaxed.RelaxedSymmetricXZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_boundary_relaxed.RelaxedSymmetricYZ
   :members:

Horizontal smoothing
^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tasmania.dycore.horizontal_smoothing.HorizontalSmoothing
   :members:

.. autoclass:: tasmania.dycore.horizontal_smoothing.HorizontalSmoothingFirstOrderXYZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_smoothing.HorizontalSmoothingFirstOrderXZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_smoothing.HorizontalSmoothingFirstOrderYZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_smoothing.HorizontalSmoothingSecondOrderXYZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_smoothing.HorizontalSmoothingSecondOrderXZ
   :members:

.. autoclass:: tasmania.dycore.horizontal_smoothing.HorizontalSmoothingSecondOrderYZ
   :members:

Numerical fluxes
^^^^^^^^^^^^^^^^

.. autoclass:: tasmania.dycore.flux_isentropic.FluxIsentropic
   :members:

.. autoclass:: tasmania.dycore.flux_isentropic_upwind.FluxIsentropicUpwind
   :members:

.. autoclass:: tasmania.dycore.flux_isentropic_centered.FluxIsentropicCentered
   :members:

.. autoclass:: tasmania.dycore.flux_isentropic_maccormack.FluxIsentropicMacCormack
   :members:

.. autoclass:: tasmania.dycore.flux_isentropic_nonconservative.FluxIsentropicNonconservative
   :members:

.. autoclass:: tasmania.dycore.flux_isentropic_nonconservative_centered.FluxIsentropicNonconservativeCentered
   :members:

Prognostics
^^^^^^^^^^^

.. autoclass:: tasmania.dycore.prognostic_isentropic.PrognosticIsentropic
   :members:

.. autoclass:: tasmania.dycore.prognostic_isentropic_centered.PrognosticIsentropicCentered
   :members:

.. autoclass:: tasmania.dycore.prognostic_isentropic_forward_euler.PrognosticIsentropicForwardEuler
   :members:

.. autoclass:: tasmania.dycore.prognostic_isentropic_nonconservative.PrognosticIsentropicNonconservative
   :members:

.. autoclass:: tasmania.dycore.prognostic_isentropic_nonconservative_centered.PrognosticIsentropicNonconservativeCentered
   :members:

Sedimentation flux
^^^^^^^^^^^^^^^^^^

.. autoclass:: tasmania.dycore.flux_sedimentation.FluxSedimentation
   :members:

.. autoclass:: tasmania.dycore.flux_sedimentation.FluxSedimentationUpwindFirstOrder
   :members:

.. autoclass:: tasmania.dycore.flux_sedimentation.FluxSedimentationUpwindSecondOrder
   :members:

Wave absorber
^^^^^^^^^^^^^

.. autoclass:: tasmania.dycore.vertical_damping.VerticalDamping
   :members:

.. autoclass:: tasmania.dycore.vertical_damping.VerticalDampingRayleigh
   :members:


Grids
-----

Two-dimensional grids
^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tasmania.grids.grid_xy.GridXY
   :members:

.. autoclass:: tasmania.grids.grid_xz.GridXZ
   :members:

.. autoclass:: tasmania.grids.sigma.Sigma2d
   :members:

.. autoclass:: tasmania.grids.gal_chen.GalChen2d
   :members:

.. autoclass:: tasmania.grids.sleve.SLEVE2d
   :members:

Three-dimensional grids
^^^^^^^^^^^^^^^^^^^^^^^

.. autoclass:: tasmania.grids.grid_xyz.GridXYZ
   :members:

.. autoclass:: tasmania.grids.sigma.Sigma3d
   :members:

.. autoclass:: tasmania.grids.gal_chen.GalChen3d
   :members:

.. autoclass:: tasmania.grids.sleve.SLEVE3d
   :members:


Model
-----

.. autoclass:: tasmania.model.Model
   :members:


Namelist
--------

.. automodule:: tasmania.namelist
   :members:


Parameterizations
-----------------

.. autoclass:: tasmania.parameterizations.adjustments.Adjustment
   :members:

.. autoclass:: tasmania.parameterizations.slow_tendencies.SlowTendency
   :members:

.. autoclass:: tasmania.parameterizations.fast_tendencies.FastTendency
   :members:

Microphysics
^^^^^^^^^^^^

.. autoclass:: tasmania.parameterizations.adjustments.AdjustmentMicrophysics
   :members:

.. autoclass:: tasmania.parameterizations.adjustment_microphysics_kessler_wrf.AdjustmentMicrophysicsKesslerWRF
   :members:

.. autoclass:: tasmania.parameterizations.adjustment_microphysics_kessler_wrf_saturation.AdjustmentMicrophysicsKesslerWRFSaturation
   :members:

.. autoclass:: tasmania.parameterizations.slow_tendencies.SlowTendencyMicrophysics
   :members:

.. autoclass:: tasmania.parameterizations.slow_tendency_microphysics_kessler_wrf.SlowTendencyMicrophysicsKesslerWRF
   :members:

.. autoclass:: tasmania.parameterizations.slow_tendency_microphysics_kessler_wrf_saturation.SlowTendencyMicrophysicsKesslerWRFSaturation
   :members:

Storages
--------

.. autoclass:: tasmania.storages.grid_data.GridData
   :members:

.. autoclass:: tasmania.storages.state_isentropic.StateIsentropic
   :members:


Topography
----------

.. automodule:: tasmania.grids.topography
   :members:

Parsers
^^^^^^^

.. autoclass:: tasmania.grids.parser.parser_1d.Parser1d
   :members: __cinit__, evaluate

.. autoclass:: tasmania.grids.parser.parser_2d.Parser2d
   :members: __cinit__, evaluate


Utilities
---------

General-purpose utilities
^^^^^^^^^^^^^^^^^^^^^^^^^

.. automodule:: tasmania.utils.utils
   :members:

.. automodule:: tasmania.set_namelist
   :members:

Meteo utilities
^^^^^^^^^^^^^^^

.. automodule:: tasmania.utils.utils_meteo
   :members:

Plotting utilities
^^^^^^^^^^^^^^^^^^

.. automodule:: tasmania.utils.utils_plot
   :members:
