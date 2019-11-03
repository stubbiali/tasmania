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
from sympl import DataArray

import gridtools as gt
from tasmania.python.framework.base_components import (
    DiagnosticComponent,
    ImplicitTendencyComponent,
    TendencyComponent,
)
from tasmania.python.physics.microphysics.utils import SedimentationFlux
from tasmania.python.utils.data_utils import get_physical_constants
from tasmania.python.utils.storage_utils import zeros
from tasmania.python.utils.meteo_utils import goff_gratch_formula, tetens_formula

try:
    from tasmania.conf import datatype
except ImportError:
    from numpy import float32 as datatype


mfwv = "mass_fraction_of_water_vapor_in_air"
mfcw = "mass_fraction_of_cloud_liquid_water_in_air"
mfpw = "mass_fraction_of_precipitation_water_in_air"
ndpw = "number_density_of_precipitation_water"


class PorzMicrophysics(TendencyComponent):
    """
	The microphysics scheme proposed by Porz et al. (2018).

	References
	----------
	Porz, N., Hanke, M., Baumgartner, M., and Spichtinger, P. (2018). \
		A model for warm clouds with implicit droplet activation, \
		avoiding saturation adjustment. *Math. Clim. Weather Forecast*, 4:50-78.
	"""

    # useful coefficients
    ae = 0.78
    alpha = 190.3
    ak = 0.002646
    av = 0.78
    beta = 4.0 / 15.0
    bk = 245.4
    bv = 0.308
    ck = -12.0
    D0 = 2.11e-5
    eps = 0.622
    k1 = 0.0041
    k2 = 0.8
    m0 = 4.0 / 3.0 * np.pi * 1000 * 0.5e-6 ** 3
    mt = 1.21e-5
    mu0 = 1.458e-6
    N0 = 1000.0
    p_star = 101325.0
    rho_star = 1.225
    t0 = 273.15
    t_mu = 110.4
    t_star = 288.0

    # default value for the activation parameter
    d_ninf = DataArray(8e8, attrs={"units": "kg^-1"})

    # default values for the physical constants used in the class
    _d_physical_constants = {
        "air_pressure_at_sea_level": DataArray(1e5, attrs={"units": "Pa"}),
        "density_of_liquid_water": DataArray(1e3, attrs={"units": "kg m^-3"}),
        "gas_constant_of_dry_air": DataArray(287.05, attrs={"units": "J K^-1 kg^-1"}),
        "gas_constant_of_water_vapor": DataArray(461.52, attrs={"units": "J K^-1 kg^-1"}),
        "latent_heat_of_vaporization_of_water": DataArray(
            2.5e6, attrs={"units": "J kg^-1"}
        ),
        "specific_heat_of_dry_air_at_constant_pressure": DataArray(
            1004.0, attrs={"units": "J K^-1 kg^-1"}
        ),
    }

    def __init__(
        self,
        domain,
        grid_type="numerical",
        air_pressure_on_interface_levels=True,
        tendency_of_air_potential_temperature_in_diagnostics=False,
        rain_evaporation=True,
        activation_parameter=d_ninf,
        saturation_water_vapor_formula="tetens",
        backend="numpy",
        dtype=datatype,
        physical_constants=None,
        **kwargs
    ):
        """
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		air_pressure_on_interface_levels : `bool`, optional
			:obj:`True` (respectively, :obj:`False`) if the input pressure
			field is defined at the interface (resp., main) levels.
			Defaults to :obj:`True`.
		tendency_of_air_potential_temperature_in_diagnostics : `bool`, optional
			:obj:`True` to include the tendency for the potential
			temperature in the output dictionary collecting the diagnostics,
			:obj:`False` otherwise. Defaults to :obj:`False`.
		rain_evaporation : `bool`, optional
			:obj:`True` if the evaporation of raindrops should be taken
			into account, :obj:`False` otherwise. Defaults to :obj:`True`.
		activation_parameter : `sympl.DataArray`, optional
			The free parameter appearing in Eq. (22) of Porz et al. (2018);
			in units compatible with [kg^-1].
		saturation_water_vapor_formula : `str`, optional
			The formula giving the saturation water vapor. Available options are:

				* 'tetens' (default) for the Tetens' equation;
				* 'goff_gratch' for the Goff-Gratch equation.

		backend : `obj`, optional
			TODO
		dtype : `numpy.dtype`, optional
			The data type for any :class:`numpy.ndarray` instantiated and
			used within this class.
		physical_constants : `dict`, optional
			Dictionary whose keys are strings indicating physical constants used
			within this object, and whose values are :class:`sympl.DataArray`\s
			storing the values and units of those constants. The constants might be:

				* 'air_pressure_at_sea_level', in units compatible with [Pa];
				* 'density_of_liquid_water', in units compatible with [kg m^-3];
				* 'gas_constant_of_dry_air', in units compatible with \
					[J K^-1 kg^-1];
				* 'gas_constant_of_water_vapor', in units compatible with \
					[J K^-1 kg^-1];
				* 'latent_heat_of_vaporization_of_water', in units compatible with \
					[J kg^-1];
				* 'specific_heat_of_dry_air_at_constant_pressure', in units \
					compatible with [J K^-1 kg^-1].

		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`tasmania.TendencyComponent`.
		"""
        # keep track of input arguments
        self._pttd = tendency_of_air_potential_temperature_in_diagnostics
        self._air_pressure_on_interface_levels = air_pressure_on_interface_levels
        self._rain_evaporation = rain_evaporation
        self._ninf = activation_parameter.to_units("kg^-1").values.item()

        # call parent's constructor
        super().__init__(domain, grid_type, **kwargs)

        # set physical parameters values
        self._physical_constants = get_physical_constants(
            self._d_physical_constants, physical_constants
        )

        # set the formula calculating the saturation water vapor pressure
        self._swvf = (
            goff_gratch_formula
            if saturation_water_vapor_formula == "goff_gratch"
            else tetens_formula
        )

        # instantiate the underlying GT4Py stencil
        self._stencil_initialize(backend, dtype)

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_stgz = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return_dict = {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "air_temperature": {"dims": dims, "units": "K"},
            "mass_fraction_of_water_vapor_in_air": {"dims": dims, "units": "kg kg^-1"},
            "mass_fraction_of_cloud_liquid_water_in_air": {
                "dims": dims,
                "units": "kg kg^-1",
            },
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "kg kg^-1",
            },
            "number_density_of_precipitation_water": {"dims": dims, "units": "kg^-1"},
        }

        if self._air_pressure_on_interface_levels:
            return_dict["air_pressure_on_interface_levels"] = {
                "dims": dims_stgz,
                "units": "Pa",
            }
        else:
            return_dict["air_pressure"] = {"dims": dims, "units": "Pa"}

        return return_dict

    @property
    def tendency_properties(self):
        grid = self._grid
        dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])

        return_dict = {
            "mass_fraction_of_water_vapor_in_air": {
                "dims": dims,
                "units": "kg kg^-1 s^-1",
            },
            "mass_fraction_of_cloud_liquid_water_in_air": {
                "dims": dims,
                "units": "kg kg^-1 s^-1",
            },
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "kg kg^-1 s^-1",
            },
            "number_density_of_precipitation_water": {
                "dims": dims,
                "units": "kg^-1 s^-1",
            },
        }

        if not self._pttd:
            return_dict["air_potential_temperature"] = {"dims": dims, "units": "K s^-1"}

        return return_dict

    @property
    def diagnostic_properties(self):
        if self._pttd:
            grid = self._grid
            dims = (grid.x.dims[0], grid.y.dims[0], grid.z.dims[0])
            return {
                "tendency_of_air_potential_temperature": {"dims": dims, "units": "K s^-1"}
            }
        else:
            return {}

    def array_call(self, state):
        # retrieve needed quantities from input state
        self._in_rho[...] = state["air_density"][...]
        self._in_p[...] = (
            state["air_pressure"][...]
            if not self._air_pressure_on_interface_levels
            else state["air_pressure_on_interface_levels"][...]
        )
        self._in_t[...] = state["air_temperature"][...]
        self._in_qv[...] = state["mass_fraction_of_water_vapor_in_air"][...]
        self._in_qc[...] = state["mass_fraction_of_cloud_liquid_water_in_air"][...]
        self._in_qr[...] = state["mass_fraction_of_precipitation_water_in_air"][...]
        self._in_nr[...] = state["number_density_of_precipitation_water"][...]

        # calculate the saturation concentration for water vapor
        self._in_ps[...] = self._swvf(self._in_t)

        # calculate the number density of cloud droplets
        qc, Ninf = self._in_qc, self._ninf
        self._in_nc[...] = (
            qc[...]
            * Ninf
            / ((qc[...] + Ninf * self.m0) * np.tanh(qc[...] / (self.N0 * self.m0)))
        )

        if False:
            # evaluate the stencil
            self._stencil.compute()
        else:
            # shortcuts
            pref = self._physical_constants["air_pressure_at_sea_level"]
            rhol = self._physical_constants["density_of_liquid_water"]
            rd = self._physical_constants["gas_constant_of_dry_air"]
            rv = self._physical_constants["gas_constant_of_water_vapor"]
            l = self._physical_constants["latent_heat_of_vaporization_of_water"]
            cp = self._physical_constants["specific_heat_of_dry_air_at_constant_pressure"]
            rho = self._in_rho
            p = (
                self._in_p
                if not self._air_pressure_on_interface_levels
                else 0.5 * (self._in_p[:, :, :-1] + self._in_p[:, :, 1:])
            )
            ps = self._in_ps
            t = self._in_t
            qv = self._in_qv
            qc = self._in_qc
            qr = self._in_qr
            nc = self._in_nc
            nr = self._in_nr

            # calculate some of the coefficients needed to compute the tendencies
            D = self.D0 * (t / self.t0) ** 1.94 * self.p_star / p
            K = self.ak * t ** 1.5 / (t + self.bk * 10 ** (self.ck / t))
            G = 1.0 / ((l / (rv * t) - 1.0) * l * ps * D / (rv * K * t ** 2) + 1.0)
            d = 4.0 * np.pi * (3.0 / (4.0 * np.pi * rhol)) ** (1.0 / 3.0) * D * G
            mu = self.mu0 * t ** 1.5 / (t + self.t_mu)
            be = (
                self.bv
                * (mu / (rho * D)) ** (1.0 / 3.0)
                * (2.0 * rho / mu) ** 0.5
                * (3.0 / (4.0 * np.pi * rhol)) ** (1.0 / 6.0)
            )

            # calculate the terminal velocity of water particles
            vt = (
                self.alpha
                * qr ** self.beta
                * (self.mt / (qr + self.mt * nr)) ** self.beta
                * (self.rho_star / rho) ** 0.5
            )

            # calculate the saturation mixing ratio of water vapor
            qvs = self.eps * ps / p

            # calculate the tendencies due to autoconversion, accretion and condensation
            A1 = self.k1 * rho * qc ** 2 / rhol
            A1p = 0.5 * self.k1 * rho * nc * qc / rhol
            A2 = (
                self.k2
                * np.pi
                * (3.0 / (4.0 * np.pi * rhol)) ** (2.0 / 3.0)
                * vt
                * rho
                * qc
                * qr ** (2.0 / 3.0)
                * nr ** (1.0 / 3.0)
            )
            C = d * rho * (qv - qvs) * nc ** (2.0 / 3.0) * qc ** (1.0 / 3.0)

            if self._rain_evaporation:
                # calculate the tendencies due to evaporation
                E = (
                    d
                    * rho
                    * ((qvs - qv) > 0.0)
                    * (qvs - qv)
                    * (
                        self.ae * qr ** (1.0 / 3.0) * nr ** (2.0 / 3.0)
                        + be * vt ** 0.5 * qr ** 0.5 * nr ** 0.5
                    )
                )
                Ep = E * nr / qr
                Ep[qr <= 0.0] = 0.0

            # calculate the overall tendencies
            if self._rain_evaporation:
                self._out_qv[...] = -C[...] + E[...]
                self._out_qc[...] = C[...] - A1[...] - A2[...]
                self._out_qr[...] = A1[...] + A2[...] - E[...]
                self._out_nr[...] = A1p[...] - Ep[...]
                self._out_theta[...] = (
                    (pref / p[...]) ** (rd / cp) * l * (C[...] - E[...]) / cp
                )
            else:
                self._out_qv[...] = -C[...]
                self._out_qc[...] = C[...] - A1[...] - A2[...]
                self._out_qr[...] = A1[...] + A2[...]
                self._out_nr[...] = A1p[...]
                self._out_theta[...] = (pref / p[...]) ** (rd / cp) * l * C[...] / cp

        # set the outputs
        tendencies = {
            "mass_fraction_of_water_vapor_in_air": self._out_qv,
            "mass_fraction_of_cloud_liquid_water_in_air": self._out_qc,
            "mass_fraction_of_precipitation_water_in_air": self._out_qr,
            "number_density_of_precipitation_water": self._out_nr,
        }
        if self._pttd:
            diagnostics = {"tendency_of_air_potential_temperature": self._out_theta}
        else:
            tendencies["air_potential_temperature"] = self._out_theta
            diagnostics = {}

        return tendencies, diagnostics

    def _stencil_initialize(self, backend, dtype):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        # allocate the numpy arrays serving as stencil inputs
        self._in_rho = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_p = np.zeros(
            (nx, ny, nz + 1 if self._air_pressure_on_interface_levels else nz),
            dtype=dtype,
        )
        self._in_ps = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_t = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_qv = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_qc = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_qr = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_nc = np.zeros((nx, ny, nz), dtype=dtype)
        self._in_nr = np.zeros((nx, ny, nz), dtype=dtype)

        # allocate the numpy arrays serving as stencil outputs
        self._out_qv = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_qc = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_qr = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_nr = np.zeros((nx, ny, nz), dtype=dtype)
        self._out_theta = np.zeros((nx, ny, nz), dtype=dtype)

        # instantiate the stencil
        self._stencil = gt.NGStencil(
            definitions_func=self._stencil_defs,
            inputs={
                "in_rho": self._in_rho,
                "in_p": self._in_p,
                "in_ps": self._in_ps,
                "in_t": self._in_t,
                "in_qv": self._in_qv,
                "in_qc": self._in_qc,
                "in_qr": self._in_qr,
                "in_nc": self._in_nc,
                "in_nr": self._in_nr,
            },
            outputs={
                "out_qv": self._out_qv,
                "out_qc": self._out_qc,
                "out_qr": self._out_qr,
                "out_nr": self._out_nr,
                "out_theta": self._out_theta,
            },
            domain=gt.domain.Rectangle((0, 0, 0), (nx - 1, ny - 1, nz - 1)),
            mode=backend,
        )

    def _stencil_defs(self, in_rho, in_p, in_ps, in_t, in_qv, in_qc, in_qr, in_nc, in_nr):
        # shortcuts
        pref = self._physical_constants["air_pressure_at_sea_level"]
        rhol = self._physical_constants["density_of_liquid_water"]
        rd = self._physical_constants["gas_constant_of_dry_air"]
        rw = self._physical_constants["gas_constant_of_water_vapor"]
        l = self._physical_constants["latent_heat_of_vaporization_of_water"]
        cp = self._physical_constants["specific_heat_of_dry_air_at_constant_pressure"]

        # instantiate the indices
        k = gt.Index(axis=2)

        # allocate the temporary fields
        p = gt.Equation()
        qvs = gt.Equation()
        D = gt.Equation()
        K = gt.Equation()
        G = gt.Equation()
        d = gt.Equation()
        mu = gt.Equation()
        be = gt.Equation()
        vt = gt.Equation()
        A1 = gt.Equation()
        A1p = gt.Equation()
        A2 = gt.Equation()
        C = gt.Equation()
        if self._rain_evaporation:
            E = gt.Equation()
            Ep = gt.Equation()

        # allocate the output fields
        out_qv = gt.Equation()
        out_qc = gt.Equation()
        out_qr = gt.Equation()
        out_nr = gt.Equation()
        out_theta = gt.Equation()

        # computations
        ## TODO


class PorzFallVelocity(DiagnosticComponent):
    """
	Calculate the effective fall velocity for the mass and number
	concentration of raindrops as prescribed by the microphysics scheme
	proposed by Porz et al. (2018).

	References
	----------
	Porz, N., Hanke, M., Baumgartner, M., and Spichtinger, P. (2018). \
		A model for warm clouds with implicit droplet activation, \
		avoiding saturation adjustment. *Math. Clim. Weather Forecast*, 4:50-78.
	"""

    # useful coefficients
    alpha = 190.3
    beta = 4.0 / 15.0
    cn = 0.58
    cq = 1.84
    mt = 1.21e-5
    rho_star = 1.225

    def __init__(
        self,
        domain,
        grid_type="numerical",
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False
    ):
        """
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		backend : `str`, optional
			TODO
		backend_opts : `dict`, optional
			TODO
		build_info : `dict`, optional
			TODO
		dtype : `numpy.dtype`, optional
			TODO
		exec_info : `dict`, optional
			TODO
		default_origin : `tuple`, optional
			TODO
		rebuild : `bool`, optional
			TODO
		"""
        super().__init__(domain, grid_type)

        self._exec_info = exec_info

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        self._in_rho = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._in_qr = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._in_nr = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._out_vq = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._out_vn = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)

        decorator = gt.stencil(
            backend,
            backend_opts=backend_opts,
            build_info=build_info,
            rebuild=rebuild,
            module="porz_fall_velocity",
            externals={
                "alpha": self.__class__.alpha,
                "beta": self.__class__.beta,
                "cn": self.__class__.cn,
                "cq": self.__class__.cq,
                "mt": self.__class__.mt,
                "rho_star": self.__class__.rho_star,
            },
        )
        self._stencil = decorator(self._stencil_defs)

    @property
    def input_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "kg kg^-1",
            },
            "number_density_of_precipitation_water": {"dims": dims, "units": "kg^-1"},
        }

        return return_dict

    @property
    def diagnostic_properties(self):
        dims = (self.grid.x.dims[0], self.grid.y.dims[0], self.grid.z.dims[0])

        return_dict = {
            "raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"},
            "number_density_of_raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"},
        }

        return return_dict

    def array_call(self, state):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz

        # extract the needed model variables
        self._in_rho.data[...] = state["air_density"][...]
        self._in_qr.data[...] = state[mfpw][...]
        self._in_nr.data[...] = state[ndpw][...]

        # run the stencil
        self._stencil(
            in_rho=self._in_rho,
            in_qr=self._in_qr,
            in_nr=self._in_nr,
            out_vq=self._out_vq,
            out_vn=self._out_vn,
            origin={"_all_": (0, 0, 0)},
            domain=(nx, ny, nz),
            exec_info=self._exec_info,
        )

        # collect the diagnostics
        diagnostics = {
            "raindrop_fall_velocity": self._out_vq.data,
            "number_density_of_raindrop_fall_velocity": self._out_vn.data,
        }

        return diagnostics

    @staticmethod
    def _stencil_defs(
        in_rho: gt.storage.f64_sd,
        in_qr: gt.storage.f64_sd,
        in_nr: gt.storage.f64_sd,
        out_vq: gt.storage.f64_sd,
        out_vn: gt.storage.f64_sd,
    ):
        vt = (
            alpha
            * in_qr[0, 0, 0] ** beta
            * (mt / (in_qr[0, 0, 0] + mt * in_nr[0, 0, 0])) ** beta
            * (rho_star / in_rho[0, 0, 0]) ** 0.5
        )
        out_vq = cq * vt[0, 0, 0]
        out_vn = cn * vt[0, 0, 0]


class PorzSedimentation(ImplicitTendencyComponent):
    """
	Calculate the vertical derivative of the sedimentation flux for the mass
	and number density of precipitation water.
	"""

    def __init__(
        self,
        domain,
        grid_type="numerical",
        sedimentation_flux_scheme="first_order_upwind",
        maximum_vertical_cfl=0.975,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=datatype,
        exec_info=None,
        default_origin=None,
        rebuild=False,
        **kwargs
    ):
        """
		Parameters
		----------
		domain : tasmania.Domain
			The underlying domain.
		grid_type : `str`, optional
			The type of grid over which instantiating the class. Either:

				* 'physical';
				* 'numerical' (default).

		sedimentation_flux_scheme : `str`, optional
			The numerical sedimentation flux scheme. Please refer to
			:class:`~tasmania.SedimentationFlux` for the available options.
			Defaults to 'first_order_upwind'.
		maximum_vertical_cfl : `float`, optional
			Maximum allowed vertical CFL number. Defaults to 0.975.
		backend : `str`, optional
			TODO
		backend_opts : `dict`, optional
			TODO
		build_info : `dict`, optional
			TODO
		dtype : `numpy.dtype`, optional
			TODO
		exec_info : `dict`, optional
			TODO
		default_origin : `tuple`, optional
			TODO
		rebuild : `bool`, optional
			TODO
		**kwargs :
			Additional keyword arguments to be directly forwarded to the parent
			:class:`~tasmania.ImplicitTendencyComponent`.
		"""
        super().__init__(domain, grid_type, **kwargs)

        self._exec_info = exec_info

        sflux = SedimentationFlux.factory(sedimentation_flux_scheme)
        self._nb = sflux.nb

        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        self._in_rho = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._in_h = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._in_qr = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._in_nr = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._in_vq = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._in_vn = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._out_qr = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)
        self._out_nr = zeros((nx + 1, ny + 1, nz + 1), backend, dtype, default_origin)

        decorator = gt.stencil(
            backend,
            backend_opts=backend_opts,
            build_info=build_info,
            rebuild=rebuild,
            module="porz_sedimentation",
            externals={"sflux": sflux.__call__, "max_cfl": maximum_vertical_cfl},
        )
        self._stencil = decorator(self._stencil_defs)

    @property
    def input_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])
        dims_z = (g.x.dims[0], g.y.dims[0], g.z_on_interface_levels.dims[0])

        return {
            "air_density": {"dims": dims, "units": "kg m^-3"},
            "height_on_interface_levels": {"dims": dims_z, "units": "m"},
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "g g^-1",
            },
            "number_density_of_precipitation_water": {"dims": dims, "units": "kg^-1"},
            "raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"},
            "number_density_of_raindrop_fall_velocity": {"dims": dims, "units": "m s^-1"},
        }

    @property
    def tendency_properties(self):
        g = self.grid
        dims = (g.x.dims[0], g.y.dims[0], g.z.dims[0])

        return {
            "mass_fraction_of_precipitation_water_in_air": {
                "dims": dims,
                "units": "g g^-1 s^-1",
            },
            "number_density_of_precipitation_water": {
                "dims": dims,
                "units": "kg^-1 s^-1",
            },
        }

    @property
    def diagnostic_properties(self):
        return {}

    def array_call(self, state, timestep):
        nx, ny, nz = self.grid.nx, self.grid.ny, self.grid.nz
        nb = self._nb

        dt = timestep.total_seconds()
        self._in_rho.data[:, :, :nz] = state["air_density"]
        self._in_h.data[...] = state["height_on_interface_levels"]
        self._in_qr.data[:, :, :nz] = state[mfpw]
        self._in_nr.data[:, :, :nz] = state[ndpw]
        self._in_vq.data[:, :, :nz] = state["raindrop_fall_velocity"]
        self._in_vn.data[:, :, :nz] = state["number_density_of_raindrop_fall_velocity"]

        self._stencil(
            in_rho=self._in_rho,
            in_h=self._in_h,
            in_qr=self._in_qr,
            in_nr=self._in_nr,
            in_vq=self._in_vq,
            in_vn=self._in_vn,
            out_qr=self._out_qr,
            out_nr=self._out_nr,
            dt=dt,
            origin={"_all_": (0, 0, nb)},
            domain=(nx, ny, nz - nb),
            exec_info=self._exec_info,
        )

        # dh = self._in_h[:, :, :-1] - self._in_h[:, :, 1:]
        # x = np.where(self._in_vq > self._max_cfl * dh / timestep.total_seconds())
        # if x[0].size > 0:
        #   print('Number of gps violating vertical CFL for qr: {:4d}'.format(x[0].size))
        # x = np.where(self._in_vn > self._max_cfl * dh / timestep.total_seconds())
        # if x[0].size > 0:
        #   print('Number of gps violating vertical CFL for nr: {:4d}'.format(x[0].size))

        tendencies = {
            mfpw: self._out_qr.data[:, :, :nz],
            ndpw: self._out_nr.data[:, :, :nz],
        }
        diagnostics = {}

        return tendencies, diagnostics

    @staticmethod
    def _stencil_defs(
        in_rho: gt.storage.f64_sd,
        in_h: gt.storage.f64_sd,
        in_qr: gt.storage.f64_sd,
        in_nr: gt.storage.f64_sd,
        in_vq: gt.storage.f64_sd,
        in_vn: gt.storage.f64_sd,
        out_qr: gt.storage.f64_sd,
        out_nr: gt.storage.f64_sd,
        *,
        dt: float
    ):
        # dh = in_h[0, 0, 0] - in_h[0, 0, 1]
        # tmp_vq = \
        # 	(in_vq[0, 0, 0] >  max_cfl * dh[0, 0, 0] / dt) * max_cfl * dh[0, 0, 0] / dt + \
        # 	(in_vq[0, 0, 0] <= max_cfl * dh[0, 0, 0] / dt) * in_vq[0, 0, 0]
        # tmp_vn =
        # 	(in_vn[0, 0, 0] >  max_cfl * dh[0, 0, 0] / dt) * max_cfl * dh[0, 0, 0] / dt + \
        # 	(in_vn[0, 0, 0] <= max_cfl * dh[0, 0, 0] / dt) * in_vn[0, 0, 0]

        dfdz_qr = sflux(rho=in_rho, h=in_h, q=in_qr, vt=in_vq)
        dfdz_nr = sflux(rho=in_rho, h=in_h, q=in_nr, vt=in_vn)

        out_qr = dfdz_qr[0, 0, 0] / in_rho[0, 0, 0]
        out_nr = dfdz_nr[0, 0, 0] / in_rho[0, 0, 0]
