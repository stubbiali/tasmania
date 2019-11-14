# What's New

## v0.6.0 

### Branch `new_calling_api`, `benchmarking`

  - Migrate to the new stencil calling API of GT4Py.
  - Support to both device and managed memory for GPU storages.

## v0.5.0 

### Branch `new_storage_api`

  - Migrate to the new storage API of GT4Py.
  - GPU storages are allocated as managed memory. Because of some bugs in the 
    implementation of the ufuncs, the isentropic model does not run with the 
    `gtcuda` backend as the state cannot be initialized.

## v0.4.0 

### Branch `gt4py_v04_2`

  - Pass dedicated storages to the stencils. The storages *do* subclass `numpy.ndarray`.
  - Use explicitly-synced device memory.

### Branch `gt4py_v04_2`

  - Use Unified Memory Address (UMA) device memory.

## v0.3.0 

### Branch `gt4py_v04_1`

  - Provide stencils with `numpy.ndarray`s.
  - Avoid copying input fields into pre-allocated buffers.

## v0.2.0 

### Branch `gt4py_v04_0`

  - Upgrade to GT4Py-v0.4.0.
  - Pass dedicated storages to the stencils. The storages *do not* subclass `numpy.ndarray`.

## v0.1.0 

### Branch `gt4py_v03`

  - Leverage GT4Py-v0.3.0.

