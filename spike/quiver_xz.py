# -*- coding: utf-8 -*-
#
# Tasmania
#
# Copyright (c) 2018-2024, ETH Zurich
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
import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np

from tasmania.conf import datatype
import python.utils.utils as utils


def make_plot_quiver_xz(
    grid, state, field_to_plot, y_level, fig=None, **kwargs
):
    """
	Given an input model state, generate the quiver plot of a specified field at a cross-section
	parallel to the :math:`xz`-plane.

	Parameters
	----------
	grid : obj
		The underlying grid, as an instance of :class:`~tasmania.grids.grid_xyz.GridXYZ`,
		or one of its derived classes.
	state : dict
		A model state dictionary.
	field_to_plot : str
		String specifying the field to plot. This might be:

		* 'velocity', for the velocity field of a two-dimensional, isentropic_prognostic, steady-state flow; \
			the current object must contain the following variables:

			- `air_isentropic_density`;
			- `x_momentum_isentropic`;
			- `height_on_interface_levels`.

	y_level : int
		:math:`y`-index identifying the cross-section.
	fig : `obj`, optional
		A :class:`matplotlib.pyplot.figure`. If not given, it will be instantiated within the function.
	**kwargs :
		Keyword arguments to specify different plotting settings.
		See :func:`~tasmania.plot.quiver_xz.plot_quiver_xz` for the complete list.

	Return
	------
	obj :
		A :class:`matplotlib.pyplot.figure` with the desired plot.

	Raise
	-----
	ValueError :
		If neither the grid, nor the model state, contains `height` or `height_on_interface_levels`.
	"""
    # Extract, compute, or interpolate the field to plot
    if field_to_plot == "velocity":
        assert (
            grid.ny == 1
        ), "The input grid should consist of only one point in the y-direction."
        assert (
            y_level == 0
        ), "As the grid consists of only one point in the y-direction, y_level must be 0."

        # Extract the variables which are needed
        s, U, h = utils.get_numpy_arrays(
            state,
            (slice(0, None), y_level, slice(0, None)),
            "air_isentropic_density",
            "x_momentum_isentropic",
            "height_on_interface_levels",
        )

        # Compute the x-velocity
        vx = U / s

        # Compute the (Cartesian) vertical velocity
        if h.shape[1] == grid.nz + 1:
            h = 0.5 * (h[:, :-1] + h[:, 1:])
        z_ = 0.5 * (h[:-1, :] + h[1:, :])
        z = np.concatenate((z_[0:1, :], z_, z_[-1:, :]), axis=0)
        vz = vx * (z[1:, :] - z[:-1, :]) / grid.dx

        # Compute the velocity magnitude
        scalar = np.sqrt(vx ** 2 + vz ** 2)
    else:
        raise RuntimeError("Unknown field to plot.")

    # Shortcuts
    nx, nz = grid.nx, grid.nz
    ni, nk = scalar.shape

    # The underlying x-grid
    x = grid.x[:] if ni == nx else grid.x_at_u_locations[:]
    xv = np.repeat(x[:, np.newaxis], nk, axis=1)

    # Extract the height of the main or interface levels
    try:
        z = get_numpy_arrays(
            state,
            (slice(0, None), y_level, slice(0, None)),
            ("height_on_interface_levels", "height"),
        )
    except KeyError:
        try:
            z = grid.height_on_interface_levels[:]
        except AttributeError:
            try:
                z = grid.height[:]
            except AttributeError:
                print(
                    """Neither the grid, nor the state, contains either ''height''
						 or ''height_on_interface_levels''."""
                )

    # Reshape the extracted height of the vertical levels
    if z.shape[1] < nk:
        raise ValueError(
            """As the field to plot is vertically staggered,
							''height_on_interface_levels'' is needed."""
        )
    topo_ = z[:, -1]
    if z.shape[1] > nk:
        z = 0.5 * (z[:, :-1] + z[:, 1:])

    # The underlying z-grid
    zv = np.zeros((ni, nk), dtype=datatype)
    if ni == nx:
        zv[:, :] = z[:, :]
    else:
        zv[1:-1, :] = 0.5 * (z[:-1, :] + z[1:, :])
        zv[0, :], zv[-1, :] = zv[1, :], zv[-2, :]

    # The underlying topography
    if ni == nx:
        topo = topo_
    else:
        topo = np.zeros((nx + 1), dtype=datatype)
        topo[1:-1] = 0.5 * (topo_[:-1] + topo_[1:])
        topo[0], topo[-1] = topo[1], topo[-2]

    # Plot
    utils_plot.quiver_xz(xv, zv, topography, vx, vz, scalar, **kwargs)


def make_plot_quiver_xz(
    grid, state, field_to_plot, y_level, fig=None, **kwargs
):
    """
    Generate the quiver plot of a gridded vectorial field at a cross-section parallel to the :math:`xz`-plane.

    Parameters
    ----------
    x : array_like
            Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`x`-grid.
    z : array_like
            Two-dimensional :class:`numpy.ndarray` representing the underlying :math:`z`-grid.
    topography : array_like
            One-dimensional :class:`numpy.ndarray` representing the underlying topography height.
    vx : array_like
            :class:`numpy.ndarray` representing the :math:`x`-component of the field to plot.
    vz : array_like
            :class:`numpy.ndarray` representing the :math:`z`-component of the field to plot.
    scalar : `array_like`, optional
            :class:`numpy.ndarray` representing a scalar field associated with the vectorial field.
            The arrows will be colored based on the associated scalar value.
            If not specified, the arrows will be colored based on their magnitude.

    Keyword arguments
    -----------------
    show : bool
            :obj:`True` if the plot should be showed, :obj:`False` otherwise. Default is :obj:`True`.
    destination : str
            String specify the path to the location where the plot will be saved. Default is :obj:`None`,
            meaning that the plot will not be saved. Note that the plot may be saved only if :data:`show`
            is set to :obj:`False`.
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
            Sequence representing the interval of the :math:`x`-axis to visualize.
            By default, the entire domain is shown.
    x_step : int
            Maximum distance between the :math:`x`-index of a drawn point, and the :math:`x`-index of any
            of its neighbours. Default is 2, i.e., only half of the points will be drawn.
    z_label : str
            Label for the :math:`z`-axis. Default is 'z'.
    z_factor : float
            Scaling factor for the :math:`z`-axis. Default is 1.
    z_lim : sequence
            Sequence representing the interval of the :math:`z`-axis to visualize.
            By default, the entire domain is shown.
    z_step : int
            Maximum distance between the :math:`z`-index of a drawn point, and the :math:`z`-index of any
            of its neighbours. Default is 2, i.e., only half of the points will be drawn.
    field_factor : float
            Scaling factor for the field. Default is 1.
    cmap_name : str
            Name of the Matplotlib's color map to be used. All the color maps provided by Matplotlib,
            as well as the corresponding inverted versions, are available. If not specified, no color map
            will be used, and the arrows will draw black.
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
    text : str
            Text to be added to the figure as anchored text. By default, no extra text is shown.
    text_loc : str
            String specifying the location where the text box should be placed. Default is 'upper right';
            please see :class:`matplotlib.offsetbox.AnchoredText` for all the available options.
    """
    # Get keyword arguments
    show = kwargs.get("show", True)
    destination = kwargs.get("destination", None)
    fontsize = kwargs.get("fontsize", 12)
    figsize = kwargs.get("figsize", [8, 8])
    title = kwargs.get("title", "$xz$-quiver")
    x_label = kwargs.get("x_label", "x")
    x_factor = kwargs.get("x_factor", 1.0)
    x_lim = kwargs.get("x_lim", None)
    x_step = kwargs.get("x_step", 2)
    z_label = kwargs.get("z_label", "y")
    z_factor = kwargs.get("z_factor", 1.0)
    z_lim = kwargs.get("z_lim", None)
    z_step = kwargs.get("z_step", 2)
    field_factor = kwargs.get("field_factor", 1.0)
    cmap_name = kwargs.get("cmap_name", None)
    cbar_levels = kwargs.get("cbar_levels", 14)
    cbar_ticks_step = kwargs.get("cbar_ticks_step", 1)
    cbar_center = kwargs.get("cbar_center", None)
    cbar_half_width = kwargs.get("cbar_half_width", None)
    cbar_x_label = kwargs.get("cbar_x_label", "")
    cbar_y_label = kwargs.get("cbar_y_label", "")
    cbar_title = kwargs.get("cbar_title", "")
    cbar_orientation = kwargs.get("cbar_orientation", "vertical")

    # Shortcuts
    ni, nj = scalar.shape

    # Global settings
    mpl.rcParams["font.size"] = fontsize

    # Rescale the axes and the field for visualization purposes
    x *= x_factor
    z *= z_factor
    topography *= z_factor
    scalar *= field_factor

    # Instantiate figure and axis objects
    fig, ax = plt.subplots(figsize=figsize)

    # Draw topography
    plt.plot(x[:, -1], topography, color="black")

    if cmap_name is not None:
        # Create color bar for colormap
        if scalar is None:
            scalar = np.sqrt(vx ** 2 + vz ** 2)
        scalar_min, scalar_max = np.nanmin(scalar), np.nanmax(scalar)
        if cbar_center is None or not (
            lt(scalar_min, cbar_center) and lt(cbar_center, scalar_max)
        ):
            cbar_lb, cbar_ub = scalar_min, scalar_max
        else:
            half_width = (
                max(cbar_center - scalar_min, scalar_max - cbar_center)
                if cbar_half_width is None
                else cbar_half_width
            )
            cbar_lb, cbar_ub = (
                cbar_center - half_width,
                cbar_center + half_width,
            )
        color_scale = np.linspace(cbar_lb, cbar_ub, cbar_levels, endpoint=True)

        # Create colormap
        if cmap_name == "BuRd":
            cm = _reverse_colormap(plt.get_cmap("RdBu"), "BuRd")
        else:
            cm = plt.get_cmap(cmap_name)

    # Generate quiver-plot
    if cmap_name is None:
        q = plt.quiver(
            x[::x_step, ::z_step],
            z[::x_step, ::z_step],
            vx[::x_step, ::z_step],
            vz[::x_step, ::z_step],
        )
    else:
        q = plt.quiver(
            x[::x_step, ::z_step],
            z[::x_step, ::z_step],
            vx[::x_step, ::z_step],
            vz[::x_step, ::z_step],
            scalar[::x_step, ::z_step],
            cmap=cm,
        )

    # Set plot settings
    ax.set(xlabel=x_label, ylabel=z_label)
    ax.set_title(title, loc="left", fontsize=fontsize - 1)
    if x_lim is not None:
        ax.set_xlim(x_lim)
    if z_lim is not None:
        ax.set_ylim(z_lim)

    # Set colorbar
    if cmap_name is not None:
        cb = plt.colorbar(orientation=cbar_orientation)
        cb.set_ticks(
            0.5 * (color_scale[:-1] + color_scale[1:])[::cbar_ticks_step]
        )
        cb.ax.set_title(cbar_title)
        cb.ax.set_xlabel(cbar_x_label)
        cb.ax.set_ylabel(cbar_y_label)

    # Show
    fig.tight_layout()
    if show or (destination is None):
        plt.show()
    else:
        plt.savefig(destination + ".eps", format="eps", dpi=1000)
