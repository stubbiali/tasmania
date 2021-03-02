This directory contains the drivers used to benchmark Tasmania.

The drivers are divided into different folders based on the model which they steer:

- `burgers/` for the Burgers' model;
- `isentropic_dry/` for the isentropic model in dry configuration;
- `isentropic_moist/` for the isentropic model including the water species.

The models are listed in increasing level of complexity.

Each driver is named `driver_namelist_{PDC}.py` where `PDC` denotes the
algorithm used to couple the physics with the dynamics:

- `fc` for the full-coupling scheme;
- `lfc` for the *lazy* full-coupling scheme;
- `ps` for the parallel splitting method;
- `sts` for the sequential-tendency splitting method;
- `sus` for the sequential-update splitting method;
- `ssus` for the symmetrized sequential-update splitting method.

The configuration parameters for `driver_namelist_{PDC}.py` are grabbed from the
module `namelist_{PDC}.py` sitting within the same folder.

To run a driver, jump into the corresponding folder and execute
```bash
python driver_namelist_{PDC}.py [-b BACKEND] [--no-log]
```
The command line option `-b` allows to override the string identifying the
backend to be used. Available options are:

- `numpy` for NumPy;
- `cupy` for CuPy;
- `gt4py:{GT_BACKEND}` for the `GT_BACKEND` backend of GT4Py.

Unless the flag `--no-log` is specified, each run produces two files:

- `{FOLDER}_log_{PDC}_{BACKEND}.txt`: Text file reporting the time spent in each
section of the code;
- `{FOLDER}_exec_{PDC}_{BACKEND}.csv`: Table collecting some performance
counters for each stencil.

In addition, the total run time is written into `{FOLDER}_run_{PDC}.csv`. The
location of the three output files is dictated by the namelist variable
`prefix`.
