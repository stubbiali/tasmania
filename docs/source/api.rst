=================
Computational domain
=================

Computational domain
====================




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
