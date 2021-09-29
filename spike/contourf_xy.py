# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2021, ETH Zurich
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
def make_animation_contourf_xy(
    grid, states_list, field_to_plot, z_level, save_dest, **kwargs
):
    """
    Given a list of model states, generate an animation showing the time-evolution of the contourf plot
    of a specified field at a cross-section parallel to the :math:`xy`-plane.

    Parameters
    ----------
    grid : obj
            The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
            or one of its derived classes.
    states_list : list
            List of model state dictionaries. Each state must necessarily contain the key `time`.
    field_to_plot : str
            String specifying the field to plot. This might be:

            * the name of a variable stored in the input model state;
            * 'horizontal_velocity', for the horizontal velocity; the current object must contain either:

                    - `x_velocity` and `y_velocity`;
                    - `x_velocity_at_u_locations` and `y_velocity_at_v_locations`;
                    - `air_density`, `x_momentum`, and `y_momentum`;
                    - `air_isentropic_density`, `x_momentum_isentropic`, and `y_momentum_isentropic`.

    z_level : int
            :obj:`k`-index identifying the cross-section.
    save_dest : str
            Path to the location where the movie will be saved. The path should include the format extension.
    **kwargs :
            Keyword arguments to specify different plotting settings.
            See :func:`~tasmania.plot.contourf_xy.plot_animation_contourf_xy` for the complete list.
    """
    # Extract, compute, or interpolate the field to plot
    for t, state in enumerate(states_list):
        time_ = state["time"]

        if field_to_plot in state.keys():
            var_ = state[field_to_plot].values[:, :, z_level]
        elif field_to_plot == "horizontal_velocity":
            try:
                u, v = get_numpy_arrays(
                    state,
                    (slice(0, None), slice(0, None), z_level),
                    "x_velocity",
                    "y_velocity",
                )
            except KeyError:
                pass

            try:
                u, v = get_numpy_arrays(
                    state,
                    (slice(0, None), slice(0, None), z_level),
                    "x_velocity_at_u_locations",
                    "y_velocity_at_v_locations",
                )
                u = 0.5 * (u[:-1, :] + u[1:, :])
                v = 0.5 * (v[:, :-1] + v[:, 1:])
            except KeyError:
                pass

            try:
                rho, U, V = get_numpy_arrays(
                    state,
                    (slice(0, None), slice(0, None), z_level),
                    "air_density",
                    "x_momentum",
                    "y_momentum",
                )
                u, v = U / rho, V / rho
            except KeyError:
                pass

            try:
                s, U, V = get_numpy_arrays(
                    state,
                    (slice(0, None), slice(0, None), z_level),
                    "air_isentropic_density",
                    "x_momentum_isentropic",
                    "y_momentum_isentropic",
                )
                u, v = U / s, V / s
            except KeyError:
                pass

            var_ = np.sqrt(u ** 2 + v ** 2)
        else:
            raise ValueError("Unknown field to plot {}.".format(field_to_plot))

        if t == 0:
            time = [
                time_,
            ]
            var = np.copy(var_[:, :, np.newaxis])
        else:
            time.append(time_)
            var = np.concatenate((var, var_[:, :, np.newaxis]), axis=2)

    # Shortcuts
    nx, ny = grid.nx, grid.ny
    ni, nj, nt = var.shape

    for t, state in enumerate(states_list):
        # The underlying x-grid
        x = grid.x[:] if ni == nx else grid.x_at_u_locations[:]
        xv_ = np.repeat(x[:, np.newaxis], nj, axis=1)

        # The underlying y-grid
        y = grid.y[:] if nj == ny else grid.y_at_v_locations[:]
        yv_ = np.repeat(y[np.newaxis, :], ni, axis=0)

        # The topography height
        _topo_ = np.copy(grid.topography_height)
        topo_ = np.zeros((ni, nj), dtype=datatype)
        if ni == nx and nj == ny:
            topo_[:, :] = _topo_[:, :]
        elif ni == nx + 1 and nj == ny:
            topo_[1:-1, :] = 0.5 * (_topo_[:-1, :] + _topo_[1:, :])
            topo_[0, :], topo_[-1, :] = topo_[1, :], topo_[-2, :]
        elif ni == nx and nj == ny + 1:
            topo_[:, 1:-1] = 0.5 * (_topo_[:, :-1] + _topo_[:, 1:])
            topo_[:, 0], topo_[:, -1] = topo_[:, 1], topo_[:, -2]
        else:
            topo_[1:-1, 1:-1] = 0.25 * (
                _topo_[:-1, :-1]
                + _topo_[1:, :-1]
                + _topo_[:-1, :1]
                + _topo_[1:, 1:]
            )
            topo_[0, 1:-1], topo_[-1, 1:-1] = topo_[1, 1:-1], topo_[-2, 1:-1]
            topo_[:, 0], topo_[:, -1] = topo_[:, 1], topo_[:, -2]

        if t == 0:
            xv = np.copy(xv_[:, :, np.newaxis])
            yv = np.copy(yv_[:, :, np.newaxis])
            topo = np.copy(topo_[:, :, np.newaxis])
        else:
            xv = np.concatenate((xv, xv_[:, :, np.newaxis]), axis=2)
            yv = np.concatenate((yv, yv_[:, :, np.newaxis]), axis=2)
            topo = np.concatenate((topo, topo_[:, :, np.newaxis]), axis=2)

    # Plot
    plot_animation_contourf_xy(time, xv, yv, var, topo, save_dest, **kwargs)


def plot_animation_contourf_xy(
    time, x, y, field, topography, save_dest, **kwargs
):
    """
    Generate an animation showing the time evolution of the contourf plot of a gridded field
    at a cross-section parallel to the :math:`xy`-plane.

    Parameters
    ----------
    time : list
            List of :class:`datetime.datetime`\s representing the time instant of each snapshot.
    x : array_like
            Three-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
            The first axis represents the :math:`x`-direction, the second axis represents the
            :math:`y`-direction, and the third axis represents the time dimension.
    y : array_like
            Three-dimensional :class:`numpy.ndarray` representing the underlying :math:`y`-grid.
            The first axis represents the :math:`x`-direction, the second axis represents the
            :math:`y`-direction, and the third axis represents the time dimension.
    field : array_like
            Three-dimensional :class:`numpy.ndarray` representing the field to plot.
            The first axis represents the :math:`x`-direction, the second axis represents the
            :math:`y`-direction, and the third axis represents the time dimension.
    topography : array_like
            Three-dimensional :class:`numpy.ndarray` representing the underlying topography.
            The first axis represents the :math:`x`-direction, the second axis represents the
            :math:`y`-direction, and the third axis represents the time dimension.
    save_dest : str
            Path to the location where the movie will be saved. The path should include the format extension.

    Keyword arguments
    -----------------
    fontsize : int
            The fontsize to be used. Default is 12.
    figsize : sequence
            Sequence representing the figure size. Default is [8,8].
    title : str
            The figure title. Default is an empty string.
    x_label : str
            Label for the :math:`x`-axis. Default is 'x'.
    x_factor : float
            Scaling factor for the :math:`x`-axis. Default is 1.
    x_lim : sequence
            Specify the limits for the :math:`x`-axis, so that the portion of the :math:`x`-axis which is
            visualized is :obj:`x_factor * (x_lim[0], x_lim[1])`. By default, the entire domain is shown.
    y_label : str
            Label for the :math:`y`-axis. Default is 'y'.
    y_factor : float
            Scaling factor for the :math:`y`-axis. Default is 1.
    y_lim : sequence
            Specify the limits for the :math:`y`-axis, so that the portion of the :math:`y`-axis which is
            visualized is :obj:`y_factor * (y_lim[0], y_lim[1])`. By default, the entire domain is shown.
    field_bias : float
            Bias for the field, so that the contourf plot for :obj:`field - field_bias` is generated. Default is 0.
    field_factor : float
            Scaling factor for the field, so that the contourf plot for :obj:`field_factor * field` is generated.
            If a bias is specified, then the contourf plot for :obj:`field_factor * (field - field_bias)` is generated.
            Default is 1.
    cmap_name : str
            Name of the Matplotlib's color map to be used. All the color maps provided by Matplotlib,
            as well as the corresponding inverted versions, are available.
    cbar_levels : int
            Number of levels for the color bar. Default is 14.
    cbar_ticks_step : int
            Distance between two consecutive labelled ticks of the color bar. Default is 1, i.e.,
            all ticks are displayed with the corresponding label.
    cbar_center : float
            Center of the range covered by the color bar. By default, the color bar covers the spectrum
            ranging from the minimum to the maximum assumed by the field.
    cbar_half_width : float
            Half-width of the range covered by the color bar. By default, the color bar covers the spectrum
            ranging from the minimum to the maximum assumed by the field.
    cbar_x_label : str
            Label for the horizontal axis of the color bar. Default is an empty string.
    cbar_y_label : str
            Label for the vertical axis of the color bar. Default is an empty string.
    cbar_orientation : str
            Orientation of the color bar. Either 'vertical' (default) or 'horizontal'.
    fps : int
            Frames per second. Default is 15.
    text : str
            Text to be added to the figure as anchored text. By default, no extra text is shown.
    text_loc : str
            String specifying the location where the text box should be placed. Default is 'upper right';
            please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
    """
    # Shortcuts
    ni, nj, nt = field.shape

    # Get keyword arguments
    fontsize = kwargs.get("fontsize", 12)
    figsize = kwargs.get("figsize", [8, 8])
    title = kwargs.get("title", "")
    x_label = kwargs.get("x_label", "x")
    x_factor = kwargs.get("x_factor", 1.0)
    x_lim = kwargs.get("x_lim", None)
    y_label = kwargs.get("y_label", "y")
    y_factor = kwargs.get("y_factor", 1.0)
    y_lim = kwargs.get("y_lim", None)
    field_bias = kwargs.get("field_bias", 0.0)
    field_factor = kwargs.get("field_factor", 1.0)
    cmap_name = kwargs.get("cmap_name", "RdYlBu")
    cbar_levels = kwargs.get("cbar_levels", 14)
    cbar_ticks_step = kwargs.get("cbar_ticks_step", 1)
    cbar_center = kwargs.get("cbar_center", None)
    cbar_half_width = kwargs.get("cbar_half_width", None)
    cbar_x_label = kwargs.get("cbar_x_label", "")
    cbar_y_label = kwargs.get("cbar_y_label", "")
    cbar_title = kwargs.get("cbar_title", "")
    cbar_orientation = kwargs.get("cbar_orientation", "vertical")
    fps = kwargs.get("fps", 15)
    text = kwargs.get("text", None)
    text_loc = kwargs.get("text_loc", "upper right")

    # Global settings
    mpl.rcParams["font.size"] = fontsize

    # Rescale the axes and the field for visualization purposes
    x *= x_factor
    x_lim = None if x_lim is None else [x_factor * lim for lim in x_lim]
    y *= y_factor
    y_lim = None if y_lim is None else [y_factor * lim for lim in y_lim]
    field = (field - field_bias) * field_factor

    # Instantiate figure and axes objects
    fig, ax = plt.subplots(figsize=figsize)

    # Instantiate writer class
    ffmpeg_writer = manimation.writers["ffmpeg"]
    metadata = {"title": ""}
    writer = ffmpeg_writer(fps=fps, metadata=metadata)

    # Create color bar for colormap
    field_min, field_max = np.amin(field), np.amax(field)
    if cbar_center is None or not (
        lt(field_min, cbar_center) and lt(cbar_center, field_max)
    ):
        cbar_lb, cbar_ub = field_min, field_max
    else:
        half_width = (
            max(cbar_center - field_min, field_max - cbar_center)
            if cbar_half_width is None
            else cbar_half_width
        )
        cbar_lb, cbar_ub = cbar_center - half_width, cbar_center + half_width
    color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint=True)

    # Create colormap
    if cmap_name == "BuRd":
        cm = plot_utils.reverse_colormap(plt.get_cmap("RdBu"), "BuRd")
    else:
        cm = plt.get_cmap(cmap_name)

    with writer.saving(fig, save_dest, nt):
        for n in range(nt):
            # Clean the canvas
            ax.cla()

            # Draw topography isolevels
            plt.contour(
                x[:, :, n], y[:, :, n], topography[:, :, n], colors="gray"
            )

            # Plot the field
            plt.contourf(
                x[:, :, n], y[:, :, n], field[:, :, n], color_scale, cmap=cm
            )

            # Set axes labels and figure title
            ax.set(xlabel=x_label, ylabel=y_label)
            ax.set_title(title, loc="left", fontsize=fontsize - 1)

            # Set x-limits
            if x_lim is None:
                ax.set_xlim([x[0, 0, n], x[-1, 0, n]])
            else:
                ax.set_xlim(x_lim)

            # Set y-limits
            if y_lim is None:
                ax.set_ylim([y[0, 0, n], y[0, -1, n]])
            else:
                ax.set_ylim(y_lim)

            # Add text
            if text is not None:
                ax.add_artist(AnchoredText(text, loc=text_loc))

            if n == 0:
                # Set colorbar
                cb = plt.colorbar(orientation=cbar_orientation)
                cb.set_ticks(
                    0.5
                    * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step]
                )
                cb.ax.set_title(cbar_title)
                cb.ax.set_xlabel(cbar_x_label)
                cb.ax.set_ylabel(cbar_y_label)

            # Add time
            plt.title(
                str(
                    convert_datetime64_to_datetime(time[n])
                    - convert_datetime64_to_datetime(time[0])
                ),
                loc="right",
                fontsize=fontsize - 1,
            )

            # Set layout
            fig.tight_layout()

            # Let the writer grab the frame
            writer.grab_frame()

    # Bring axes and field back to original units
    x /= x_factor
    y /= y_factor
    field = field / field_factor + field_bias
