This folder contains the following Python scripts:

  - `assess_convergence.py`: visualize the root mean square difference between a 
    field computed at a given grid resolution, and the same field calculated at a 
    higher (reference) resolution;
  - `compare_solutions.py`: assess if two data sets (e.g. calculated using different
    backends) are equal;
  - `compress.py`: remove some model variables from a data set;
  - `make_animation.py`: generate a movie;
  - `make_barplot.py`: generate a bar plot;
  - `make_plot.py`: draw a figure consisting of a single panel;
  - `make_plot_composite.py`: draw a figure consisting of multiple panels;
  - `plot_burgers_error.py`: visualize the convergence analysis for the Burgers' equations;
  - `plot_burgers_time.py`: visualize the run times for the Burgers' equations at
    different grid resolutions;
  - `plot_burgers_error_time.py`: visualize both the convergence analysis and
    the completion times for the Burgers' equations;
  - `plot_grid.py`: plot a vertical cross-section of a terrain-following grid;
  - `plot_stability_function_rk2rk2.py`: visualize the linear stability region
    for all the coupling methods; both the dynamics and the physics are integrated
    using RK2;
  - `plot_stability_function_rk3rk2.py`: visualize the linear stability region
    for all the coupling methods; the dynamics is integreated using RK3, while the 
    physics is integrated using RK2.
    