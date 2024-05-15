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
import numpy as np
from sympl import DataArray


def change_dims(paxis, dims):
    ndims = dims if dims is not None else paxis.dims[0]
    return DataArray(
        paxis.values,
        coords=[paxis.values],
        dims=ndims,
        attrs={"units": paxis.attrs["units"]},
    )


def extend_axis(paxis, nb, dims):
    pvalues = paxis.values
    cdims = dims if dims is not None else paxis.dims[0]
    mi, dtype = pvalues.shape[0], pvalues.dtype

    cvalues = np.zeros(mi + 2 * nb, dtype=dtype)
    cvalues[nb:-nb] = pvalues[...]
    cvalues[:nb] = np.array(
        [pvalues[0] - i * (pvalues[1] - pvalues[0]) for i in range(nb, 0, -1)],
        dtype=dtype,
    )
    cvalues[-nb:] = np.array(
        [pvalues[-1] + (i + 1) * (pvalues[1] - pvalues[0]) for i in range(nb)],
        dtype=dtype,
    )

    return DataArray(
        cvalues,
        coords=[cvalues],
        dims=cdims,
        name=paxis.name,
        attrs={"units": paxis.attrs["units"]},
    )


def repeat_axis(paxis, nb, dims):
    pvalues = paxis.values
    dims = dims if dims is not None else paxis.dims[0]
    name = paxis.name
    attrs = paxis.attrs
    dtype = paxis.dtype

    if pvalues[0] <= pvalues[-1]:
        padneg = np.array(
            tuple(pvalues[0] - nb + i for i in range(nb)), dtype=dtype
        )
        padpos = np.array(
            tuple(pvalues[-1] + i + 1 for i in range(nb)), dtype=dtype
        )
    else:
        padneg = np.array(
            tuple(pvalues[0] + nb - i for i in range(nb)), dtype=dtype
        )
        padpos = np.array(
            tuple(pvalues[-1] - i - 1 for i in range(nb)), dtype=dtype
        )

    cvalues = np.concatenate((padneg, pvalues, padpos), axis=0)

    return DataArray(
        cvalues, coords=[cvalues], dims=dims, name=name, attrs=attrs
    )


def shrink_axis(naxis, nb, dims):
    nvalues = naxis.values
    dims = dims if dims is not None else naxis.dims[0]
    name = naxis.name
    attrs = naxis.attrs

    pvalues = nvalues[nb:-nb]

    return DataArray(
        pvalues, coords=[pvalues], dims=dims, name=name, attrs=attrs
    )
