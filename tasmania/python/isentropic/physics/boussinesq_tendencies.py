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
"""
This module contains:
	NonconservativeFlux
	UpwindFlux
	CenteredFlux
	ThirdOrderUpwindFlux
	FifthOrderUpwindFlux
	IsentropicBoussinesqMetric
"""
import abc
import numpy as np

import gridtools as gt
from sympl import DataArray
from tasmania.python.framework.base_components import TendencyComponent

try:
    from tasmania.conf import datatype
except ImportError:
    datatype = np.float64


# convenient aliases
mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"


class NonconservativeFlux:
    __metaclass__ = abc.ABCMeta

    nb = None

    @staticmethod
    @abc.abstractmethod
    def get_flux_x(i, j, dx, u, h):
        pass

    @staticmethod
    @abc.abstractmethod
    def get_flux_y(i, j, dy, v, h):
        pass

    @staticmethod
    def factory(scheme):
        if scheme == "upwind":
            return Upwind()
        elif scheme == "centered":
            return Centered()
        elif scheme == "third_order_upwind":
            return ThirdOrderUpwind()
        elif scheme == "fifth_order_upwind":
            return FifthOrderUpwind()
        else:
            raise ValueError("Unsupported advection scheme.")


class Upwind(NonconservativeFlux):
    nb = 1

    @staticmethod
    def get_flux_x(i, j, dx, u, h):
        flux_x = gt.Equation()
        flux_x[i, j] = u[i, j] * (
            (u[i, j] > 0.0) * (h[i, j] - h[i - 1, j]) / dx
            + (u[i, j] < 0.0) * (h[i + 1, j] - h[i, j]) / dx
        )
        return flux_x

    @staticmethod
    def get_flux_y(i, j, dy, v, h):
        flux_y = gt.Equation()
        flux_y[i, j] = v[i, j] * (
            (v[i, j] > 0.0) * (h[i, j] - h[i, j - 1]) / dy
            + (v[i, j] < 0.0) * (h[i, j + 1] - h[i, j]) / dy
        )
        return flux_y


class Centered(NonconservativeFlux):
    nb = 1

    @staticmethod
    def get_flux_x(i, j, dx, u, h):
        flux_x = gt.Equation()
        flux_x[i, j] = u[i, j] * (h[i + 1, j] - h[i - 1, j]) / (2.0 * dx)
        return flux_x

    @staticmethod
    def get_flux_y(i, j, dy, v, h):
        flux_y = gt.Equation()
        flux_y[i, j] = v[i, j] * (h[i, j + 1] - h[i, j - 1]) / (2.0 * dy)
        return flux_y


class ThirdOrderUpwind(NonconservativeFlux):
    nb = 2

    @staticmethod
    def get_flux_x(i, j, dx, u, h):
        flux_x = gt.Equation()
        flux_x[i, j] = u[i] / (12.0 * dx) * (
            8.0 * (h[i + 1] - h[i - 1]) - (h[i + 2] - h[i - 2])
        ) + (u[i] * (u[i] > 0.0) - u[i] * (u[i] < 0.0)) / (12.0 * dx) * (
            h[i + 2] - 4.0 * h[i + 1] + 6.0 * h[i] - 4.0 * h[i - 1] + h[i - 2]
        )
        return flux_x

    @staticmethod
    def get_flux_y(i, j, dy, v, h):
        flux_y = gt.Equation()
        flux_y[i, j] = v[j] / (12.0 * dy) * (
            8.0 * (h[j + 1] - h[j - 1]) - (h[j + 2] - h[j - 2])
        ) + (v[j] * (v[j] > 0.0) - v[j] * (v[j] < 0.0)) / (12.0 * dy) * (
            h[j + 2] - 4.0 * h[j + 1] + 6.0 * h[j] - 4.0 * h[j - 1] + h[j - 2]
        )
        return flux_y


class FifthOrderUpwind(NonconservativeFlux):
    nb = 3

    @staticmethod
    def get_flux_x(i, j, dx, u, h):
        flux_x = gt.Equation()
        flux_x[i, j] = u[i] / (60.0 * dx) * (
            45.0 * (h[i + 1] - h[i - 1])
            - 9.0 * (h[i + 2] - h[i - 2])
            + (h[i + 3] - h[i - 3])
        ) - (u[i] * (u[i] > 0.0) - u[i] * (u[i] < 0.0)) / (60.0 * dx) * (
            (h[i + 3] + h[i - 3])
            - 6.0 * (h[i + 2] + h[i - 2])
            + 15.0 * (h[i + 1] + h[i - 1])
            - 20.0 * h[i]
        )
        return flux_x

    @staticmethod
    def get_flux_y(i, j, dy, v, h):
        flux_y = gt.Equation()
        flux_y[i, j] = v[j] / (60.0 * dy) * (
            45.0 * (h[j + 1] - h[j - 1])
            - 9.0 * (h[j + 2] - h[j - 2])
            + (h[j + 3] - h[j - 3])
        ) - (v[j] * (v[j] > 0.0) - v[j] * (v[j] < 0.0)) / (60.0 * dy) * (
            (h[j + 3] + h[j - 3])
            - 6.0 * (h[j + 2] + h[j - 2])
            + 15.0 * (h[j + 1] + h[j - 1])
            - 20.0 * h[j]
        )
        return flux_y


class IsentropicBoussinesqTendency(TendencyComponent):
    """
	Calculating the correction terms accounting for the incompressibility
	and Boussinesq approximations.
	The class is always instantiated over the numerical grid of the
	underlying domain.
	"""

    # default values for the physical constants used in the class
    _d_physical_constants = {
        "gravitational_acceleration": DataArray(9.80665, attrs={"units": "m s^-2"})
    }

    def __init__(
        self,
        domain,
        advection_scheme="upwind",
        moist=False,
        backend=gt.mode.NUMPY,
        dtype=datatype,
        **kwargs
    ):
        """
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		advection_scheme : `str`, optional
			Advection scheme to be used to discretize the metric term.
			Available options are:

				* 'upwind' (default), for the upwind scheme;
				* 'centered', for the centered scheme;
				* 'third_order_upwind', for the third-order upwind scheme;
				* 'fifth_order_upwind', for the fifth-order upwind scheme.

		moist : `bool`, optional
			:obj:`True` if water species are included in the model,
			:obj:`False` otherwise. Defaults to :obj:`False`.
		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` allocated and
			used within this class.
		**kwargs:
			Keyword arguments to be directly forwarded to the parent class.
		"""
        # the object calculating the advection fluxes
        self._fluxer = NonconservativeFlux.factory(advection_scheme)

        # safety checks
        nb = domain.horizontal_boundary.nb
        flux_nb = self._fluxer.nb
        assert nb >= flux_nb, (
            "The number of boundary layers is {} but should be no smaller "
            "than {}.".format(nb, flux_nb)
        )

        # store useful input arguments
        self._moist = moist

        # call parent's constructor
        super().__init__(domain, "numerical", **kwargs)

        # allocate globals for the stencils
        self._dx = gt.Global(self.grid.dx.to_units("m").values.item())
        self._dy = gt.Global(self.grid.dy.to_units("m").values.item())

        # allocate numpy arrays serving as stencils' inputs
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        self._in_s = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_ddmtg = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_h = np.zeros((nx, ny, nz + 1), dtype=dtype)
        self._in_su = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_sv = np.zeros((nx, ny, nz), dtype=dtype)
        if moist:
            self._in_qv = np.zeros((nx, ny, nz), dtype=dtype)
            self._in_qc = np.zeros((nx, ny, nz), dtype=dtype)
            self._in_qr = np.zeros((nx, ny, nz), dtype=dtype)

        # allocate temporary numpy arrays shared across the stencils
        self._tmp_u = np.zeros((nx, ny, nz + 1), dtype=dtype)
        self._tmp_v = np.zeros((nx, ny, nz + 1), dtype=dtype)
        self._tmp_b = np.zeros((nx, ny, nz + 1), dtype=dtype)
        self._diagnostics = {"metric_term": self._tmp_b}

        # allocate the numpy arrays serving as stencils' outputs
        self._out_s = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_su = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_sv = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_ddmtg = np.zeros((nx, ny, nz), dtype=dtype)
        self._tendencies = {
            "air_isentropic_density": self._out_s,
            "x_momentum_isentropic": self._out_su,
            "y_momentum_isentropic": self._out_sv,
            "dd_montgomery_potential": self._out_ddmtg,
        }
        if moist:
            self._out_qv = np.zeros((nx, ny, nz), dtype=dtype)
            self._out_qc = np.zeros((nx, ny, nz), dtype=dtype)
            self._out_qr = np.zeros((nx, ny, nz), dtype=dtype)
            self._tendencies.update(
                {mfwv: self._out_qv, mfcw: self._out_qc, mfpw: self._out_qr}
            )

        # instantiate the stencil interpolating the velocity components
        # at the interface levels
        self._stencil_velocity = gt.NGStencil(
            definitions_func=self._stencil_velocity_defs,
            inputs={"in_s": self._in_s, "in_su": self._in_su, "in_sv": self._in_sv},
            outputs={"out_u": self._tmp_u, "out_v": self._tmp_v},
            domain=gt.domain.Rectangle((0, 0, 1), (nx - 1, ny - 1, nz - 1)),
            mode=backend,
        )

        # instantiate the stencil calculating the metric term
        self._stencil_metric_term = gt.NGStencil(
            definitions_func=self._stencil_metric_term_defs,
            inputs={"in_u": self._tmp_u, "in_v": self._tmp_v, "in_h": self._in_h},
            global_inputs={"dx": self._dx, "dy": self._dy},
            outputs={"out_b": self._tmp_b},
            domain=gt.domain.Rectangle(
                (flux_nb, flux_nb, 1), (nx - flux_nb - 1, ny - flux_nb - 1, nz - 1)
            ),
            # TODO: domain=gt.domain.Rectangle((nb, nb, 1), (nx-nb-1, ny-nb-1, nz-1)),
            mode=backend,
        )

        # instantiate the stencil calculating the tendencies
        inputs = {
            "in_b": self._tmp_b,
            "in_s": self._in_s,
            "in_ddmtg": self._in_ddmtg,
            "in_h": self._in_h,
            "in_su": self._in_su,
            "in_sv": self._in_sv,
        }
        outputs = {
            "out_s": self._out_s,
            "out_ddmtg": self._out_ddmtg,
            "out_su": self._out_su,
            "out_sv": self._out_sv,
        }
        if moist:
            inputs.update(
                {"in_qv": self._in_qv, "in_qc": self._in_qc, "in_qr": self._in_qr}
            )
            outputs.update(
                {"out_qv": self._out_qv, "out_qc": self._out_qc, "out_qr": self._out_qr}
            )
        self._stencil_tendencies = gt.NGStencil(
            definitions_func=self._stencil_tendencies_defs,
            inputs=inputs,
            outputs=outputs,
            domain=gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
            # TODO: domain=gt.domain.Rectangle((nb, nb, 0), (nx-nb-1, ny-nb-1, nz-1)),
            mode=backend,
        )

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1"},
            "dd_montgomery_potential": {"dims": dims, "units": "m^2 K^-2 s^-2"},
            "height_on_interface_levels": {"dims": dims_z, "units": "m"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-1"},
        }
        if self._moist:
            return_dict.update(
                {
                    mfwv: {"dims": dims, "units": "g g^-1"},
                    mfcw: {"dims": dims, "units": "g g^-1"},
                    mfpw: {"dims": dims, "units": "g g^-1"},
                }
            )

        return return_dict

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return_dict = {
            "air_isentropic_density": {"dims": dims, "units": "kg m^-2 K^-1 s^-1"},
            "dd_montgomery_potential": {"dims": dims, "units": "m^2 K^-2 s^-3"},
            "x_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
            "y_momentum_isentropic": {"dims": dims, "units": "kg m^-1 K^-1 s^-2"},
        }
        if self._moist:
            return_dict.update(
                {
                    mfwv: {"dims": dims, "units": "g g^-1 s^-1"},
                    mfcw: {"dims": dims, "units": "g g^-1 s^-1"},
                    mfpw: {"dims": dims, "units": "g g^-1 s^-1"},
                }
            )

        return return_dict

    @property
    def diagnostic_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])
        return {"metric_term": {"dims": dims, "units": "m s^-1"}}

    def array_call(self, state):
        self._in_s[...] = state["air_isentropic_density"][...]
        self._in_ddmtg[...] = state["dd_montgomery_potential"][...]
        self._in_h[...] = state["height_on_interface_levels"][...]
        self._in_su[...] = state["x_momentum_isentropic"][...]
        self._in_sv[...] = state["y_momentum_isentropic"][...]
        if self._moist:
            self._in_qv[...] = state[mfwv][...]
            self._in_qc[...] = state[mfcw][...]
            self._in_qr[...] = state[mfpw][...]

        self._stencil_velocity.compute()
        self._stencil_metric_term.compute()
        self._stencil_tendencies.compute()

        return self._tendencies, self._diagnostics

    @staticmethod
    def _stencil_velocity_defs(in_s, in_su, in_sv):
        # index
        k = gt.Index(axis=2)

        # output fields
        out_u = gt.Equation()
        out_v = gt.Equation()

        # calculations
        out_u[k] = 0.5 * (in_su[k - 1] / in_s[k - 1] + in_su[k] / in_s[k])
        out_v[k] = 0.5 * (in_sv[k - 1] / in_s[k - 1] + in_sv[k] / in_s[k])

        return out_u, out_v

    def _stencil_metric_term_defs(self, dx, dy, in_u, in_v, in_h):
        # indices
        i = gt.Index(axis=0)
        j = gt.Index(axis=1)

        # output field
        out_b = gt.Equation()

        # calculations
        flux_h_x = self._fluxer.get_flux_x(i, j, dx, in_u, in_h)
        flux_h_y = self._fluxer.get_flux_y(i, j, dy, in_v, in_h)
        out_b[i, j] = flux_h_x[i, j] + flux_h_y[i, j]

        return out_b

    @staticmethod
    def _stencil_tendencies_defs(
        in_b, in_s, in_ddmtg, in_h, in_su, in_sv, in_qv=None, in_qc=None, in_qr=None
    ):
        # indices
        i = gt.Index(axis=0)
        j = gt.Index(axis=1)
        k = gt.Index(axis=2)

        # temporary field
        tmp_dbdz = gt.Equation()

        # output fields
        out_s = gt.Equation()
        out_ddmtg = gt.Equation()
        out_su = gt.Equation()
        out_sv = gt.Equation()
        if in_qv is not None:
            out_qv = gt.Equation()
            out_qc = gt.Equation()
            out_qr = gt.Equation()

        # calculations
        tmp_dbdz[k] = (in_b[k] - in_b[k + 1]) / (in_h[k] - in_h[k + 1])
        out_s[i, j, k] = in_s[i, j] * tmp_dbdz[k]
        out_ddmtg[i, j, k] = in_ddmtg[i, j] * tmp_dbdz[k]
        out_su[i, j, k] = in_su[i, j] * tmp_dbdz[k]
        out_sv[i, j, k] = in_sv[i, j] * tmp_dbdz[k]
        if in_qv is not None:
            out_qv[i, j, k] = in_qv[i, j] * tmp_dbdz[k]
            out_qc[i, j, k] = in_qc[i, j] * tmp_dbdz[k]
            out_qr[i, j, k] = in_qr[i, j] * tmp_dbdz[k]

        if in_qv is None:
            return out_s, out_ddmtg, out_su, out_sv
        else:
            return out_s, out_ddmtg, out_su, out_sv, out_qv, out_qc, out_qr
