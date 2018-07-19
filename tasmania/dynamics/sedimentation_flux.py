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
Classes:
    SedimentationFlux
    _{First, Second}OrderUpwind(SedimentationFlux)
"""
import abc

import gridtools as gt


class SedimentationFlux:
	"""
	Abstract base class whose derived classes discretize the
	vertical derivative of the sedimentation flux with different
	orders of accuracy.
	"""
	# Make the class abstract
	__metaclass__ = abc.ABCMeta

	@abc.abstractmethod
	def __call__(self, k, rho, h_on_interface_levels, qr, vt):
		"""
		Get the vertical derivative of the sedimentation flux.
		As this method is marked as abstract, its implementation
		is delegated to the derived classes.

		Parameters
		----------
		k : obj
			:class:`gridtools.Index` representing the index running
			along the :math:`\\theta`-axis.
		rho : obj
			:class:`gridtools.Equation` representing the air density.
		h_on_interface_levels : obj
			:class:`gridtools.Equation` representing the geometric
			height of the model half-levels.
		qr : obj
			:class:`gridtools.Equation` representing the mass fraction
			of precipitation water in air.
		vt : obj
			:class:`gridtools.Equation` representing the raindrop
			fall velocity.

		Return
		------
		obj :
			:class:`gridtools.Equation` representing the vertical
			derivative of the sedimentation flux.
		"""

	@staticmethod
	def factory(sedimentation_flux_type):
		"""
		Static method returning an instance of the derived class
		which discretizes the vertical derivative of the
		sedimentation flux with the desired level of accuracy.

		Parameters
		----------
		sedimentation_flux_type : str
			String specifying the method used to compute the numerical
			sedimentation flux. Available options are:

			- 'first_order_upwind', for the first-order upwind scheme;
			- 'second_order_upwind', for the second-order upwind scheme.

		Return
		------
			Instance of the derived class implementing the desired method.
		"""
		if sedimentation_flux_type == 'first_order_upwind':
			return _FirstOrderUpwind()
		elif sedimentation_flux_type == 'second_order_upwind':
			return _SecondOrderUpwind()
		else:
			raise ValueError('Only first- and second-order upwind methods '
							 'have been implemented.')


class _FirstOrderUpwind(SedimentationFlux):
	"""
	This class inherits
	:class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`
	to implement a standard, first-order accurate upwind method
	to discretize the vertical derivative of the sedimentation flux.

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
		To instantiate an object of this class, one should prefer
		the static method
		:meth:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux.factory`
		of :class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`.
		"""
		self.nb = 1

	def __call__(self, k, rho, h_on_interface_levels, qr, vt):
		# Interpolate the geometric height at the model main levels
		tmp_h = gt.Equation()
		tmp_h[k] = 0.5 * (h_on_interface_levels[k] + h_on_interface_levels[k+1])

		# Calculate the vertical derivative of the sedimentation flux
		# via the upwind method
		out_dfdz = gt.Equation(name='tmp_dfdz')
		out_dfdz[k] = (rho[k-1] * qr[k-1] * vt[k-1] -
					   rho[k  ] * qr[k  ] * vt[k  ]) / \
					  (tmp_h[k-1] - tmp_h[k])

		return out_dfdz


class _SecondOrderUpwind(SedimentationFlux):
	"""
	This class inherits
	:class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`
	to implement a second-order accurate upwind method to discretize
	the vertical derivative of the sedimentation flux.

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
		To instantiate an object of this class, one should prefer
		the static method
		:meth:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux.factory`
		of :class:`~tasmania.dynamics.sedimentation_flux.SedimentationFlux`.
		"""
		self.nb = 2

	def __call__(self, k, rho, h_on_interface_levels, qr, vt):
		# Instantiate temporary and output fields
		tmp_h    = gt.Equation()
		tmp_a    = gt.Equation()
		tmp_b    = gt.Equation()
		tmp_c    = gt.Equation()
		out_dfdz = gt.Equation(name='tmp_dfdz')

		# Interpolate the geometric height at the model main levels
		tmp_h[k] = 0.5 * (h_on_interface_levels[k] + h_on_interface_levels[k+1])

		# Evaluate the space-dependent coefficients occurring in the
		# second-order, upwind finite difference approximation of the
		# vertical derivative of the flux
		tmp_a[k] = (2. * tmp_h[k] - tmp_h[k-1] - tmp_h[k-2]) / \
				   ((tmp_h[k-1] - tmp_h[k]) * (tmp_h[k-2] - tmp_h[k]))
		tmp_b[k] = (tmp_h[k-2] - tmp_h[k]) / \
				   ((tmp_h[k-1] - tmp_h[k]) * (tmp_h[k-2] - tmp_h[k-1]))
		tmp_c[k] = (tmp_h[k] - tmp_h[k-1]) / \
				   ((tmp_h[k-2] - tmp_h[k]) * (tmp_h[k-2] - tmp_h[k-1]))

		# Calculate the vertical derivative of the sedimentation flux
		# via the upwind method
		out_dfdz[k] = tmp_a[k] * rho[k  ] * qr[k  ] * vt[k  ] + \
					  tmp_b[k] * rho[k-1] * qr[k-1] * vt[k-1] + \
					  tmp_c[k] * rho[k-2] * qr[k-2] * vt[k-2]

		return out_dfdz
