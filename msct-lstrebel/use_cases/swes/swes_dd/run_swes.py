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
#!/usr/bin/python3
# -*- coding: utf-8 -*-

"""
This module contains
"""
import numpy as np
import math
import sys
import argparse
import copy
from mpi4py import MPI

import gridtools as gt
import stencils
import timer as ti

from domain_decomposition import DomainDecomposition
from dd_postprocess import DomainPostprocess


class LaxWendroffSWES:
    """
    Implementation of the finite difference Lax-Wendroff scheme
    for the shallow water equations defined on a sphere.
    """

    def __init__(self, planet, t_final, m, n, ic, cfl, diff, backend,
                 dtype=np.float64, nparts=0, path="", prefix=""):
        """
        Constructor.

        Parameters
        ----------
        planet : int
            Integer denoting the planet on which set the equations.
            Available options are:

                * 0, for the Earth;
                * 1, for Saturn.

        t_final : float
            Simulation length in days.
        m : int
            Number of grid points in the longitude direction.
        n : int
            Number of grid points in the latitude direction.
        ic : tuple
            Tuple storing the identifier of the initial condition,
            followed by optional parameters. Available options for
            the identifier are:

                * 0, for the test case #1 by Williamson et al.;
                * 1, for the test case #2 by Williamson et al.;
                * 2, for the test case #6 by Williamson et al..

        cfl : float
            CFL number.
        diff : bool
            :obj:`True` to switch on numerical diffusion, :obj:`False` otherwise.
        backend : obj
            GT4Py backend.
        dtype : `obj`, float
            Type to be used for all NumPy arrays used with this object.
            Defaults tp :obj:`numpy.float64`.
        """
        # Set planet constants
        # For Saturn, all the values are retrieved from:
        # http://nssdc.gsfc.nasa.gov/planetary/factsheet/saturnfact.html

        timer.start(name="Initialization", level=2)

        self.planet = planet
        if self.planet == 0:  # Earth
            self.g = 9.80616
            self.a = 6.37122e6
            self.omega = 7.292e-5
            self.scale_height = 8e3
            self.nu = 5e5
        elif self.planet == 1:  # Saturn
            self.g = 10.44
            self.a = 5.8232e7
            self.omega = 2. * math.pi / (10.656 * 3600.)
            self.scale_height = 60e3
            self.nu = 5e6
        else:
            raise ValueError("Unknown planet {}.".format(planet))
        #
        # Lat-lon grid
        #
        assert (m > 1) and (n > 1), \
            "Number of grid points along each direction must be greater than one."
        # Discretize longitude
        self.m = m
        # Discretize latitude
        self.n = n

        # Time discretization
        #
        assert (t_final >= 0.), "Final time must be non-negative."

        # Convert simulation length from days to seconds
        self.t_final = 24. * 3600. * t_final

        # Set maximum number of iterations
        if self.t_final == 0.:
            self.nmax = 50
            self.t_final = 3600.
        else:
            self.nmax = sys.maxsize

        # CFL number
        self.cfl = cfl

        # Initialize timestep size
        self.dt = gt.Global(0.)

        #
        # Flat terrain height
        #
        self.hs = np.zeros((self.m + 3, self.n), dtype=dtype)

        #
        # Numerical settings
        #
        assert ic[0] in range(3), 'Invalid problem identifier. ' \
                                  'See code documentation for supported initial conditions.'

        self.ic = ic
        self.diffusion = diff

        if self.ic[0] == 0:
            self.only_advection = True
        else:
            self.only_advection = False

        self.backend = backend

        # Register fields and stencils to the DomainDecomposition class:
        self.prepared_domain = DomainDecomposition("subdomains_pymetis.dat.part." + str(nparts), "metis",
                                              path=path, prefix=prefix)

        self.halo = [1, 1, 1, 1, 0, 0]

        if self.only_advection:
            self.prepared_domain.register_field(fieldname="h",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_h.npy")

            self.prepared_domain.register_field(fieldname="dx",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dx.npy")

            self.prepared_domain.register_field(fieldname="dxc",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dxc.npy",
                                                haloincluded=True)

            self.prepared_domain.register_field(fieldname="dy",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dy.npy",
                                                staggered=(0, 1, 0))

            self.prepared_domain.register_field(fieldname="dy1",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dy1.npy",
                                                staggered=(0, 1, 0),
                                                haloincluded=True)

            self.prepared_domain.register_field(fieldname="dy1c",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_dy1c.npy",
                                                haloincluded=True)

            self.prepared_domain.register_field(fieldname="c",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_c.npy")

            self.prepared_domain.register_field(fieldname="c_midy",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_c_midy.npy",
                                                staggered=(0, 1, 0))

            self.prepared_domain.register_field(fieldname="u",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_u.npy")

            self.prepared_domain.register_field(fieldname="u_midx",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_u_midx.npy")

            self.prepared_domain.register_field(fieldname="v",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_v.npy")

            self.prepared_domain.register_field(fieldname="v_midy",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_v_midy.npy",
                                                staggered=(0, 1, 0))

            self.prepared_domain.register_field(fieldname="h_new",
                                                halo=self.halo,
                                                field_ic_file=path + prefix + "swes_ic" + str(ic[0]) + "_h.npy")

        # TODO maybe this belongs to preprocess?
        # Compute minimum longitudinal and latitudinal distance between
        # adjacent grid points, needed to compute time step size through
        # CFL condition
        dx_min = []
        dy_min = []
        for sd in self.prepared_domain.subdivisions:
            # print(sd.get_interior_field("dx"))
            dx_min.append(sd.get_interior_field("dx")[:, 1:-1].min())
            dy_min.append(sd.get_interior_field("dy").min())
        self.dx_min = min(dx_min)
        self.dy_min = min(dy_min)
        # print(dx_min, self.dx_min, dy_min, self.dy_min)


        #
        # Initialize stencils
        #
        if self.only_advection:
            self.stencil = self.prepared_domain.register_stencil(
                definitions_func=stencils.definitions_advection,
                inputs={"in_h": "h", "dx": "dx", "dxc": "dxc",
                        "dy1": "dy1", "dy1c": "dy1c",
                        "c": "c", "c_midy": "c_midy",
                        "u": "u", "u_midx": "u_midx",
                        "v": "v", "v_midy": "v_midy"},
                global_inputs={"dt": self.dt},
                outputs={"out_h": "h_new"},
                mode=self.backend
            )

        timer.stop(name="Initialization")

    def solve(self, verbose, save, nparts=0, path="", prefix=""):
            """
            Perform the time marching.

            Parameters
            ----------
            verbose	: int
                If positive, print to screen information about the solution
                every :obj:`verbose` timesteps. If negative, print to screen
                information only about the initial and final state.
            save : int
                If positive, store the solution every :obj:`save` timesteps.
                If negative, store only the initial and final state.

            Returns
            -------
            t_save : array_like
                1-D :class:`numpy.ndarray` collecting the time instants (in seconds)
                at which solution has been stored.
            phi : array_like
                2-D :class:`numpy.ndarray` collecting the longitudinal coordinates
                of all grid points.
            theta : array_like
                2-D :class:`numpy.ndarray` collecting the latitudinal coordinates
                of all grid points.
            h_save : array_like
                3-D :class:`numpy.ndarray` whose k-th (i,j)-slice stores the
                fluid height at time :obj:`t_save[k]`.
            u_save : array_like
                3-D :class:`numpy.ndarray` whose k-th (i,j)-slice stores the
                u-velocity at time :obj:`t_save[k]`.
            v_save : array_like
                3-D :class:`numpy.ndarray` whose k-th (i,j)-slice stores the
                v-velocity at time :obj:`t_save[k]`.
            """
            verbose = int(verbose)
            save = int(save)
            #
            # Time marching
            #
            n, t = 0, 0.

            # Communicate partition boundaries
            self.prepared_domain.communicate("h")
            self.prepared_domain.communicate("h_new")
            self.prepared_domain.communicate("dx")
            self.prepared_domain.communicate("dxc")
            self.prepared_domain.communicate("dy1")
            self.prepared_domain.communicate("dy1c")
            self.prepared_domain.communicate("c")
            self.prepared_domain.communicate("c_midy")
            self.prepared_domain.communicate("u")
            self.prepared_domain.communicate("u_midx")
            self.prepared_domain.communicate("v")
            self.prepared_domain.communicate("v_midy")

            # for sd in self.prepared_domain.subdivisions:
            #     h_north_now, h_south_now = sd.get_interior_field("h")[0, -1], sd.get_interior_field("h")[0, 0]

            hnew_north = np.zeros((self.m, self.halo[2], 1))
            hnew_south = np.zeros((self.m, self.halo[3], 1))

            if MPI.COMM_WORLD.Get_rank() == 0:
                tsave = [0.0]

            timer.start(name="Time integration", level=2)
            self.prepared_domain.save_fields(["h"], postfix="t_" + str(0))

            while t < self.t_final and n < self.nmax:
                n += 1

                # Compute timestep through CFL condition
                # Keep track of the old timestep for leap-frog scheme at the poles
                # dtold = copy.deepcopy(self.dt)

                dtnew = []
                # Compute flux Jacobian eigenvalues
                for sd in self.prepared_domain.subdivisions:
                    eigenx = (np.maximum(
                        np.absolute(sd.get_interior_field("u")[1:-1, :]
                                    - np.sqrt(self.g * np.absolute(sd.get_interior_field("h")[1:-1, :]))),
                        np.maximum(np.absolute(
                            sd.get_interior_field("u")[1:-1, :]), np.absolute(
                            sd.get_interior_field("u")[1:-1, :]
                            + np.sqrt(self.g * np.absolute(sd.get_interior_field("h")[1:-1, :])))))).max()
                    eigeny = (np.maximum(
                        np.absolute(sd.get_interior_field("v")[1:-1, :]
                                    - np.sqrt(self.g * np.absolute(sd.get_interior_field("h")[1:-1, :]))),
                                         np.maximum(np.absolute(
                                             sd.get_interior_field("v")[1:-1, :]), np.absolute(
                                             sd.get_interior_field("v")[1:-1, :]
                                             + np.sqrt(self.g
                                                       * np.absolute(sd.get_interior_field("h")[1:-1, :])))))).max()

                    # Compute timestep
                    dtmax = np.minimum(self.dx_min / eigenx, self.dy_min / eigeny)
                    dtnew.append(self.cfl * dtmax)
                # Select local minimum time step
                # print(dtnew)
                self.dt.value = min(dtnew)
                # If run with mpi collect all local minimum, choose the global minimum and send it to everybody
                if MPI.COMM_WORLD.Get_size() > 1:
                    self.dt.value = MPI.COMM_WORLD.allreduce(sendobj=self.dt.value, op=MPI.MIN)
                # print(self.dt.value)

                # If needed, adjust time step
                if t + self.dt > self.t_final:
                    self.dt = gt.Global(self.t_final - t)
                    t = self.t_final
                else:
                    t += self.dt.value

                # print(t)

                # Update height and stereographic components at the poles
                # This is needed for pole treatment
                # h_north_old, h_south_old = h_north_now, h_south_now
                #
                # for sd in self.prepared_domain.subdivisions:
                #     h_north_now, h_south_now = sd.get_interior_field("h")[0, -1], sd.get_interior_field("h")[0, 0]

                # Communicate partition boundaries
                timer.start(name="Communication during time integration", level=3)
                self.prepared_domain.communicate("h")
                timer.stop(name="Communication during time integration")

                # for sd in self.prepared_domain.subdivisions:
                    # print(np.where(sd.fields["dy1"] == 0.0))
                    # print(np.where(sd.fields["dxc"] == 0.0))

                #
                # Update solution at the internal grid points
                #
                if self.only_advection:
                    self.stencil.compute()

                #
                # Apply boundary conditions
                #
                for sd in self.prepared_domain.subdivisions:
                    hnew_north = sd.get_interior_field("h_new")[:, 1, 0]
                    hnew_south = sd.get_interior_field("h_new")[:, -2, 0]
                    sd.set_boundary_condition("h_new", 2, hnew_north.reshape((sd.size[0], self.halo[2], sd.size[2])))
                    sd.set_boundary_condition("h_new", 3, hnew_south.reshape((sd.size[0], self.halo[3], sd.size[2])))

                self.prepared_domain.apply_boundary_condition("h_new")

                # North pole treatment
                #
                # dtv = self.dt.value
                # dtoldv = dtold.value
                # for sd in self.prepared_domain.subdivisions:
                #     h_north_new = (h_north_old + (dtv + dtoldv) * 2. / (self.dxp * self.m_north * self.m)
                #                    * np.sum(sd.get_interior_field("h")[1:-2, -2]
                #                             * sd.get_interior_field("v")[1:-2, -2]))

                #
                # South pole treatment
                #
                # for sd in self.prepared_domain.subdivisions:
                #     h_south_new = (h_south_old - (dtv + dtoldv) * 2. / (self.dxp * self.m_south * self.m) *
                #                   np.sum(sd.get_interior_field("h")[1:-2, 1]
                #                          * sd.get_interior_field("v")[1:-2, 1]))

                self.prepared_domain.swap_fields("h", "h_new")

                timer.start(name="Saving fields during time integration", level=3)
                if (save > 0 and n % save == 0) or t == self.t_final:
                    self.prepared_domain.save_fields(["h"], postfix="t_" + str(n))

                if MPI.COMM_WORLD.Get_rank() == 0:
                    if (save > 0 and n % save == 0) or t == self.t_final:
                        tsave.append(t)

                    if t == self.t_final:
                        tsave.append(n)
                        np.save("t.npy", tsave)
                timer.stop(name="Saving fields during time integration")

            timer.stop(name="Time integration")


def postprocess_swes(nx, ny, nz, nic, save_freq, path="", prefix=""):
    postproc = DomainPostprocess(path=path, prefix=prefix)

    hsave = postproc.combine_output_files(size=[nx, ny, nz], fieldname="h",
                                      path=path, prefix=prefix,
                                      postfix="t_" + str(0), save=False, cleanup=True)
    tsave = np.load("t.npy")
    nt = int(tsave[-1])
    for n in range(1, nt + 1):
        # Save
        if ((save_freq > 0) and (n % save_freq == 0)) or (n == nt):
            hnew = postproc.combine_output_files(size=[nx, ny, nz], fieldname="h",
                                             path=path, prefix=prefix,
                                             postfix="t_" + str(n), save=False, cleanup=True)

            hsave = np.concatenate((hsave, hnew), axis=2)

    phi = np.load("swes_ic0_phi.npy")
    theta = np.load("swes_ic0_theta.npy")
    usave = np.load("swes_ic0_u.npy")
    vsave = np.load("swes_ic0_v.npy")
    filename = 'swes_no_poles_ic{}.npz'.format(nic)
    np.savez(filename, t=tsave[:-1], phi=phi[0:nx, 0:ny], theta=theta[0:nx, 0:ny],
             h=hsave, u=usave[0:nx, 0:ny], v=vsave[0:nx, 0:ny])


if __name__ == "__main__":
    timer = ti.Timings(name="Shallow Water Equation on a Sphere")
    timer.start(name="Overall SWES time", level=1)

    # Suggested values for $\alpha$ for first and second
    # test cases from Williamson's suite:
    # * 0
    # * 0.05
    # * pi/2 - 0.05
    # * pi/2
    ic = (0, 0) #math.pi / 2)

    # Suggested simulation's length for Williamson's test cases:
    # * IC 0: 12 days
    # * IC 1: 14 days
    # t_final = 12
    t_final = 12

    # Let's go!
    solver = LaxWendroffSWES(planet=0, t_final=t_final, m=180, n=90, ic=ic,
                             cfl=1, diff=True, backend=gt.mode.NUMPY, dtype=np.float64, nparts=2)
    solver.solve(verbose=100, save=100)
    # t, phi, theta, h, u, v = solver.solve(verbose=100, save=10)

    timer.stop(name="Overall SWES time")

    if MPI.COMM_WORLD.Get_rank() == 0:
        timer.list_timings()

        # Save data

        postprocess_swes(180, 88, 1, ic[0], 100)

        # postprocess_swes(nx=180, ny=88, nz=1, postfix="t_0")
        # postprocess_swes(nx=180, ny=88, nz=1, postfix="t_5226")