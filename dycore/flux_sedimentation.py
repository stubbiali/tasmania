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
import abc
import warnings

import gridtools as gt

class FluxSedimentation:
	"""
	Abstract base class whose derived classes discretize the vertical derivative of the sedimentation flux
	with different orders of accuracy.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def get_vertical_derivative_of_sedimentation_flux(self, i, j, k, in_rho, in_h, in_qr, in_vt):
		"""
		Get the vertical derivative of the sedimentation flux.
		As this method is marked as abstract, its implementation is delegated to the derived classes.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_rho : obj
			:class:`gridtools.Equation` representing the air density.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height of the model half-levels.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water in air.
		in_vt : obj
			:class:`gridtools.Equation` representing the raindrop fall velocity.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the vertical derivative of the sedimentation flux.
		"""

	@staticmethod
	def factory(sedimentation_flux_type):
		"""
		Static method returning an instance of the derived class which discretizes the vertical derivative of
		the sedimentation flux with the desired level of accuracy.

		Parameters
		----------
		sedimentation_flux_type : str
			String specifying the method used to compute the numerical sedimentation flux. Available options are:

			- 'first_order_upwind', for the first-order upwind scheme;
			- 'second_order_upwind', for the second-order upwind scheme.

		Return
		------
			Instance of the derived class implementing the desired method.
		"""
		if sedimentation_flux_type == 'first_order_upwind':
			return FluxSedimentationUpwindFirstOrder()
		elif sedimentation_flux_type == 'second_order_upwind':
			return FluxSedimentationUpwindSecondOrder()
		else:
			raise ValueError('Only first- and second-order upwind methods have been implemented.')


class FluxSedimentationUpwindFirstOrder(FluxSedimentation):
	"""
	This class inherits :class:`~tasmania.dycore.sedimentation_flux.SedimentationFlux` to implement a standard, first-order
	accurate upwind method to discretize the vertical derivative of the sedimentation flux.

	Attributes
	----------
	nb : int
		Extent of the stencil in the upward vertical direction.
	"""
	def __init__(self):
		"""
		Constructor.

		Note
		----
		To instantiate an object of this class, one should prefer the static method
		:meth:`~tasmania.dycore.sedimentation_flux.SedimentationFlux.factory` of
		:class:`~tasmania.dycore.sedimentation_flux.SedimentationFlux`.
		"""
		self.nb = 1

	def get_vertical_derivative_of_sedimentation_flux(self, i, j, k, in_rho, in_h, in_qr, in_vt):
		"""
		Get the vertical derivative of the sedimentation flux.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_rho : obj
			:class:`gridtools.Equation` representing the air density.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height of the model half-levels.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water in air.
		in_vt : obj
			:class:`gridtools.Equation` representing the raindrop fall velocity.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the vertical derivative of the sedimentation flux.
		"""
		# Interpolate the geometric height at the model main levels
		tmp_h = gt.Equation()
		tmp_h[i, j, k] = 0.5 * (in_h[i, j, k] + in_h[i, j, k+1])

		# Calculate the vertical derivative of the sedimentation flux via the upwind method
		out_dfdz = gt.Equation(name = 'tmp_dfdz')
		out_dfdz[i, j, k] = (in_rho[i, j, k-1] * in_qr[i, j, k-1] * in_vt[i, j, k-1] - 
							 in_rho[i, j, k  ] * in_qr[i, j, k  ] * in_vt[i, j, k  ]) / \
							(tmp_h[i, j, k-1] - tmp_h[i, j, k])

		return out_dfdz


class FluxSedimentationUpwindSecondOrder(FluxSedimentation):
	"""
	This class inherits :class:`~tasmania.dycore.sedimentation_flux.SedimentationFlux` to implement a second-order
	accurate upwind method to discretize the vertical derivative of the sedimentation flux.

	Attributes
	----------
	nb : int
		Extent of the stencil in the upward vertical direction.
	"""
	def __init__(self):
		"""
		Constructor.

		Note
		----
		To instantiate an object of this class, one should prefer the static method
		:meth:`~tasmania.dycore.sedimentation_flux.SedimentationFlux.factory` of
		:class:`~tasmania.dycore.sedimentation_flux.SedimentationFlux`.
		"""
		self.nb = 2

	def get_vertical_derivative_of_sedimentation_flux(self, i, j, k, in_rho, in_h, in_qr, in_vt):
		"""
		Get the vertical derivative of the sedimentation flux.

		Parameters
		----------
		i : obj
			:class:`gridtools.Index` representing the index running along the :math:`x`-axis.
		j : obj
			:class:`gridtools.Index` representing the index running along the :math:`y`-axis.
		k : obj
			:class:`gridtools.Index` representing the index running along the :math:`\\theta`-axis.
		in_rho : obj
			:class:`gridtools.Equation` representing the air density.
		in_h : obj
			:class:`gridtools.Equation` representing the geometric height of the model half-levels.
		in_qr : obj
			:class:`gridtools.Equation` representing the mass fraction of precipitation water in air.
		in_vt : obj
			:class:`gridtools.Equation` representing the raindrop fall velocity.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the vertical derivative of the sedimentation flux.
		"""
		# Instantiate temporary and output field
		tmp_h    = gt.Equation()
		tmp_a    = gt.Equation()
		tmp_b    = gt.Equation()
		tmp_c    = gt.Equation()
		out_dfdz = gt.Equation(name = 'tmp_dfdz')

		# Interpolate the geometric height at the model main levels
		tmp_h[i, j, k] = 0.5 * (in_h[i, j, k] + in_h[i, j, k+1])

		# Evaluate the space-dependent coefficients occurring in the second-order, upwind finite difference 
		# approximation of the vertical derivative of the flux
		tmp_a[i, j, k] = (2. * tmp_h[i, j, k] - tmp_h[i, j, k-1] - tmp_h[i, j, k-2]) / \
						 ((tmp_h[i, j, k-1] - tmp_h[i, j, k]) * (tmp_h[i, j, k-2] - tmp_h[i, j, k]))
		tmp_b[i, j, k] = (tmp_h[i, j, k-2] - tmp_h[i, j, k]) / \
						 ((tmp_h[i, j, k-1] - tmp_h[i, j, k]) * (tmp_h[i, j, k-2] - tmp_h[i, j, k-1]))
		tmp_c[i, j, k] = (tmp_h[i, j, k] - tmp_h[i, j, k-1]) / \
						 ((tmp_h[i, j, k-2] - tmp_h[i, j, k]) * (tmp_h[i, j, k-2] - tmp_h[i, j, k-1]))

		# Calculate the vertical derivative of the sedimentation flux via the upwind method
		out_dfdz[i, j, k] = tmp_a[i, j, k] * in_rho[i, j, k  ] * in_qr[i, j, k  ] * in_vt[i, j, k  ] + \
							tmp_b[i, j, k] * in_rho[i, j, k-1] * in_qr[i, j, k-1] * in_vt[i, j, k-1] + \
							tmp_c[i, j, k] * in_rho[i, j, k-2] * in_qr[i, j, k-2] * in_vt[i, j, k-2]

		return out_dfdz
