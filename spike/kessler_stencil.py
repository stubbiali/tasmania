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
def defs_stencil_kessler(
    self, dt, in_rho, in_p, in_ps, in_exn, in_T, in_qv, in_qc, in_qr
):
    """
	GT4Py stencil carrying out the microphysical adjustments and computing the change over time
	in potential temperature.

	Parameters
	----------
	in_rho : obj
		:class:`gridtools.Equation` representing the air density.
	in_p : obj
		:class:`gridtools.Equation` representing the air pressure.
	in_ps : obj
		:class:`gridtools.Equation` representing the saturation vapor pressure.
	in_exn : obj
		:class:`gridtools.Equation` representing the Exner function.
	in_T : obj
		:class:`gridtools.Equation` representing the air temperature.
	in_qv : obj
		:class:`gridtools.Equation` representing the mass fraction of water vapor.
	in_qc : obj
		:class:`gridtools.Equation` representing the mass fraction of cloud liquid water.
	in_qr : obj
		:class:`gridtools.Equation` representing the mass fraction of precipitation water.

	Returns
	-------
	out_qv : obj
		:class:`gridtools.Equation` representing the adjusted mass fraction of water vapor.
	out_qc : obj
		:class:`gridtools.Equation` representing the adjusted mass fraction of cloud liquid water.
	out_qr : obj
		:class:`gridtools.Equation` representing the adjusted mass fraction of precipitation water.
	out_w : obj
		:class:`gridtools.Equation` representing the change over time in potential temperature.

	References
	----------
	Doms, G., et al. (2015). *A description of the nonhydrostatic regional COSMO-model. \
		Part II: Physical parameterization.* Retrieved from `http://www.cosmo-model.org`_. \
	Mielikainen, J., B. Huang, J. Wang, H. L. A. Huang, and M. D. Goldberg. (2013). \
		*Compute Unified Device Architecture (CUDA)-based parallelization of WRF Kessler \
		cloud microphysics scheme*. Computer \& Geosciences, 52:292-299.
	"""
    # Declare the indeces
    i = gt.Index()
    j = gt.Index()
    k = gt.Index()

    # Instantiate the temporary fields
    tmp_p = gt.Equation()
    tmp_p_mbar = gt.Equation()
    tmp_rho_gcm3 = gt.Equation()
    tmp_qvs = gt.Equation()
    tmp_Ar = gt.Equation()
    tmp_Cr = gt.Equation()
    tmp_C = gt.Equation()
    tmp_Er = gt.Equation()
    tmp_qv = gt.Equation()
    tmp_qc = gt.Equation()
    tmp_qc_ = gt.Equation()
    tmp_qr_ = gt.Equation()
    tmp_sat = gt.Equation()
    tmp_dlt = gt.Equation()

    # Instantiate the output fields
    out_qv = gt.Equation()
    out_qc = gt.Equation()
    out_qr = gt.Equation()
    out_w = gt.Equation()

    # Interpolate the pressure at the model main levels
    tmp_p[i, j, k] = 0.5 * (in_p[i, j, k] + in_p[i, j, k + 1])

    # Perform units conversion
    tmp_rho_gcm3[i, j, k] = 1.0e3 * in_rho[i, j, k]
    tmp_p_mbar[i, j, k] = 1.0e-2 * tmp_p[i, j, k]

    # Compute the saturation mixing ratio of water vapor
    tmp_qvs[i, j, k] = (
        self._beta
        * in_ps[i, j, k]
        / (tmp_p[i, j, k] - self._beta_c * in_ps[i, j, k])
    )

    # Compute the contribution of autoconversion to rain development
    tmp_Ar[i, j, k] = (
        self.k1 * (in_qc[i, j, k] > self.a) * (in_qc[i, j, k] - self.a)
    )

    # Compute the contribution of accretion to rain development
    tmp_Cr[i, j, k] = self.k2 * in_qc[i, j, k] * (in_qr[i, j, k] ** 0.875)

    if self._rain_evaporation_on:
        # Compute the contribution of evaporation to rain development
        tmp_C[i, j, k] = 1.6 + 124.9 * (
            (tmp_rho_gcm3[i, j, k] * in_qr[i, j, k]) ** 0.2046
        )
        tmp_Er[i, j, k] = (
            (1.0 - in_qv[i, j, k] / tmp_qvs[i, j, k])
            * tmp_C[i, j, k]
            * ((tmp_rho_gcm3[i, j, k] * in_qr[i, j, k]) ** 0.525)
            / (
                tmp_rho_gcm3[i, j, k]
                * (5.4e5 + 2.55e6 / (tmp_p_mbar[i, j, k] * tmp_qvs[i, j, k]))
            )
        )

    # Perform the adjustments, neglecting the evaporation of cloud liquid water
    if not self._rain_evaporation_on:
        tmp_qc_[i, j, k] = in_qc[i, j, k] - self.time_levels * dt * (
            tmp_Ar[i, j, k] + tmp_Cr[i, j, k]
        )
        tmp_qr_[i, j, k] = in_qr[i, j, k] + self.time_levels * dt * (
            tmp_Ar[i, j, k] + tmp_Cr[i, j, k]
        )
    else:
        tmp_qv[i, j, k] = (
            in_qv[i, j, k] + self.time_levels * dt * tmp_Er[i, j, k]
        )
        tmp_qc_[i, j, k] = in_qc[i, j, k] - self.time_levels * dt * (
            tmp_Ar[i, j, k] + tmp_Cr[i, j, k]
        )
        tmp_qr_[i, j, k] = in_qr[i, j, k] + self.time_levels * dt * (
            tmp_Ar[i, j, k] + tmp_Cr[i, j, k] - tmp_Er[i, j, k]
        )

    # Clipping
    tmp_qc[i, j, k] = (tmp_qc_[i, j, k] > 0.0) * tmp_qc_[i, j, k]
    out_qr[i, j, k] = (tmp_qr_[i, j, k] > 0.0) * tmp_qr_[i, j, k]

    # Compute the amount of latent heat released by the condensation of cloud liquid water
    if not self._rain_evaporation_on:
        tmp_sat[i, j, k] = (tmp_qvs[i, j, k] - in_qv[i, j, k]) / (
            1.0
            + self._kappa
            * in_ps[i, j, k]
            / (
                (in_T[i, j, k] - self._bw)
                * (in_T[i, j, k] - self._bw)
                * (in_p[i, j, k] - self._beta * in_ps[i, j, k])
                * (in_p[i, j, k] - self._beta * in_ps[i, j, k])
            )
        )
    else:
        tmp_sat[i, j, k] = (tmp_qvs[i, j, k] - tmp_qv[i, j, k]) / (
            1.0
            + self._kappa
            * in_ps[i, j, k]
            / (
                (in_T[i, j, k] - self._bw)
                * (in_T[i, j, k] - self._bw)
                * (in_p[i, j, k] - self._beta * in_ps[i, j, k])
                * (in_p[i, j, k] - self._beta * in_ps[i, j, k])
            )
        )

    # Compute the source term representing the evaporation of cloud liquid water
    tmp_dlt[i, j, k] = (tmp_sat[i, j, k] <= tmp_qc[i, j, k]) * tmp_sat[
        i, j, k
    ] + (tmp_sat[i, j, k] > tmp_qc[i, j, k]) * tmp_qc[i, j, k]

    # Perform the adjustments, accounting for the evaporation of cloud liquid water
    if not self._rain_evaporation_on:
        out_qv[i, j, k] = in_qv[i, j, k] + tmp_dlt[i, j, k]
        out_qc[i, j, k] = tmp_qc[i, j, k] - tmp_dlt[i, j, k]
    else:
        out_qv[i, j, k] = tmp_qv[i, j, k] + tmp_dlt[i, j, k]
        out_qc[i, j, k] = tmp_qc[i, j, k] - tmp_dlt[i, j, k]

    # Compute the change over time in potential temperature
    if not self._rain_evaporation_on:
        out_w[i, j, k] = (
            -L
            / (0.5 * (in_exn[i, j, k] + in_exn[i, j, k + 1]))
            * tmp_dlt[i, j, k]
        )
    else:
        out_w[i, j, k] = (
            -L
            / (0.5 * (in_exn[i, j, k] + in_exn[i, j, k + 1]))
            * (tmp_dlt[i, j, k] + tmp_Er[i, j, k])
        )

    return out_qv, out_qc, out_qr, out_w
