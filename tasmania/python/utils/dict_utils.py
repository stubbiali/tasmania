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

from gt4py import gtscript

from tasmania.python.utils.gtscript_utils import (
    set_annotations,
    stencil_copy_defs,
    stencil_copychange_defs,
    stencil_add_defs,
    stencil_iadd_defs,
    stencil_sub_defs,
    stencil_isub_defs,
    stencil_scale_defs,
    stencil_iscale_defs,
    stencil_addsub_defs,
    stencil_iaddsub_defs,
    stencil_fma_defs,
    stencil_sts_rk2_0_defs,
    stencil_sts_rk3ws_0_defs,
)
from tasmania.python.utils.storage_utils import deepcopy_dataarray, zeros


class DataArrayDictOperator:
    """ Operate on multiple dictionaries of :class:`sympl.DataArray`. """

    def __init__(
        self,
        gt_powered=False,
        *,
        backend="numpy",
        backend_opts=None,
        build_info=None,
        dtype=np.float64,
        rebuild=False,
        **kwargs
    ):
        """
        Parameters
        ----------
        gt_powered : `bool`, optional
            `True` to perform all the intensive math operations harnessing GT4Py.
        backend : `str`, optional
            The GT4Py backend.
        backend_opts : `dict`, optional
            Dictionary of backend-specific options.
        build_info : `dict`, optional
            Dictionary of building options.
        dtype : `data-type`, optional
            Data type of the storages.
        rebuild : `bool`, optional
            `True` to trigger the stencils compilation at any class instantiation,
            `False` to rely on the caching mechanism implemented by GT4Py.
        **kwargs :
            Catch-all for unused keyword arguments.
        """
        self._gt_powered = gt_powered
        self._dtype = dtype
        self._gt_kwargs = {
            "backend": backend,
            "build_info": build_info,
            "rebuild": rebuild,
        }
        self._gt_kwargs.update(backend_opts or {})

        self._stencil_copy = None
        self._stencil_copychange = None
        self._stencil_add = None
        self._stencil_iadd = None
        self._stencil_sub = None
        self._stencil_isub = None
        self._stencil_scale = None
        self._stencil_iscale = None
        self._stencil_addsub = None
        self._stencil_iaddsub = None
        self._stencil_fma = None
        self._stencil_sts_rk2_0 = None
        self._stencil_sts_rk3ws_0 = None

    def copy(self, dst, src, unshared_variables_in_output=False):
        """
        Overwrite the :class:`sympl.DataArray` in one dictionary using the
        :class:`sympl.DataArray` contained in another dictionary.

        Parameters
        ----------
        dst : dict[str, sympl.DataArray]
            The destination dictionary.
        src : dict[str, sympl.DataArray]
            The source dictionary.
        unshared_variables_in_output : `bool`, optional
            `True` to include in the destination dictionary the variables not
            originally shared with the source dictionary.
        """
        if "time" in src:
            dst["time"] = src["time"]

        shared_keys = tuple(key for key in src if key in dst and key != "time")
        unshared_keys = tuple(key for key in src if key not in dst and key != "time")

        if self._gt_powered:
            if self._stencil_copy is None:
                set_annotations(stencil_copy_defs, self._dtype)
                self._stencil_copy = gtscript.stencil(
                    definition=stencil_copy_defs, **self._gt_kwargs
                )

            for key in shared_keys:
                assert "units" in dst[key].attrs
                src_field = src[key].to_units(dst[key].attrs["units"]).values
                dst_field = dst[key].values
                self._stencil_copy(
                    src=src_field, dst=dst_field, origin=(0, 0, 0), domain=src_field.shape
                )
        else:
            for key in shared_keys:
                assert "units" in dst[key].attrs
                dst[key].values[...] = src[key].to_units(dst[key].attrs["units"]).values

        if unshared_variables_in_output:
            for key in unshared_keys:
                dst[key] = src[key]

    def add(
        self,
        dict1,
        dict2,
        out=None,
        field_properties=None,
        unshared_variables_in_output=False,
    ):
        """
        Add up two dictionaries item-wise.

        Parameters
        ----------
        dict1 : dict[str, sympl.DataArray]
            First addend.
        dict2 : dict[str, sympl.DataArray]
            Second addend.
        out : `dict[str, sympl.DataArray]`, optional
            The output dictionary.
        field_properties : `dict[str, dict]`, optional
            Dictionary whose keys are strings indicating the variables
            included in the output dictionary, and values are dictionaries
            gathering fundamental properties (units) for those variables.
            If not specified, a variable is included in the output dictionary
            in the same units used in the first input dictionary, or the second
            dictionary if the variable is not present in the first one.
        unshared_variables_in_output : `bool`, optional
            `True` if the output dictionary should contain those variables
            included in only one of the two input dictionaries.

        Return
        ------
        dict[str, sympl.DataArray] :
            The item-wise sum.
        """
        field_properties = field_properties or {}

        out = out or {}
        if "time" in dict1 or "time" in dict2:
            out["time"] = dict1.get("time", dict2.get("time", None))

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.difference(("time",))
        unshared_keys = set(dict1.keys()).symmetric_difference(dict2.keys())
        unshared_keys = unshared_keys.difference(("time",))

        if self._gt_powered and self._stencil_add is None:
            set_annotations(stencil_add_defs, self._dtype)
            self._stencil_add = gtscript.stencil(
                definition=stencil_add_defs, **self._gt_kwargs
            )

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            if key in out:
                out_da = out[key]
            else:
                out_da = deepcopy_dataarray(dict1[key])

            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_da.attrs["units"] = units

            if self._gt_powered:
                self._stencil_add(
                    in_a=field1,
                    in_b=field2,
                    out_c=out_da.values,
                    origin=(0, 0, 0),
                    domain=field1.shape,
                )
            else:
                out_da.values[...] = field1 + field2

            out[key] = out_da

        if unshared_variables_in_output and len(unshared_keys) > 0:
            if self._gt_powered and self._stencil_copy is None:
                set_annotations(stencil_copy_defs, self._dtype)
                self._stencil_copy = gtscript.stencil(
                    definition=stencil_copy_defs, **self._gt_kwargs
                )

            for key in unshared_keys:
                _dict = dict1 if key in dict1 else dict2

                props = field_properties.get(key, {})
                units = props.get("units", _dict[key].attrs["units"])
                if key in out:
                    if self._gt_powered:
                        self._stencil_copy(
                            src=_dict[key].to_units(units).values,
                            dst=out[key].values,
                            origin=(0, 0, 0),
                            domain=_dict[key].shape,
                        )
                    else:
                        out[key].values[...] = _dict[key].to_units(units).values

                    out[key].attrs["units"] = units
                else:
                    out[key] = _dict[key].to_units(units)

        return out

    def iadd(
        self,
        dict1,
        dict2,
        field_properties=None,
        unshared_variables_in_output=False,
        deepcopy_unshared_variables=False,
    ):
        """
        In-place variant of `add`.

        Parameters
        ----------
        dict1 : dict[str, sympl.DataArray]
            First addend, modified in-place.
        dict2 : dict[str, sympl.DataArray]
            Second addend.
        out : `dict[str, sympl.DataArray]`, optional
            The output dictionary.
        field_properties : `dict[str, dict]`, optional
            Dictionary whose keys are strings indicating the variables
            included in the output dictionary, and values are dictionaries
            gathering fundamental properties (units) for those variables.
            If not specified, a variable is included in the output dictionary
            in the same units used in the first input dictionary, or the second
            dictionary if the variable is not present in the first one.
        unshared_variables_in_output : `bool`, optional
            `True` if the output dictionary should contain those variables
            included in only one of the two input dictionaries.
        deepcopy_unshared_variables : `bool`, optional
        """
        field_properties = field_properties or {}

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.difference(("time",))
        unshared_keys = set(dict1.keys()).symmetric_difference(dict2.keys())
        unshared_keys = unshared_keys.difference(("time",))
        # shared_keys = tuple(key for key in dict1 if key in dict2 and key != "time")
        # unshared_keys = list(key for key in dict1 if key not in dict2 and key != "time")
        # unshared_keys += list(key for key in dict2 if key not in dict1 and key != "time")

        if self._gt_powered and self._stencil_iadd is None:
            set_annotations(stencil_iadd_defs, self._dtype)
            self._stencil_iadd = gtscript.stencil(
                definition=stencil_iadd_defs, **self._gt_kwargs
            )

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])
            dict1[key] = dict1[key].to_units(units)
            field1 = dict1[key].values
            field2 = dict2[key].to_units(units).values

            if self._gt_powered:
                self._stencil_iadd(
                    inout_a=field1, in_b=field2, origin=(0, 0, 0), domain=field1.shape
                )
            else:
                field1 += field2

        if unshared_variables_in_output and len(unshared_keys) > 0:
            for key in unshared_keys:
                _dict = dict1 if key in dict1 else dict2
                props = field_properties.get(key, {})
                units = props.get("units", _dict[key].attrs["units"])
                dict1[key] = (
                    deepcopy_dataarray(_dict[key].to_units(units))
                    if deepcopy_unshared_variables
                    else _dict[key].to_units(units)
                )

    def sub(
        self,
        dict1,
        dict2,
        out=None,
        field_properties=None,
        unshared_variables_in_output=False,
    ):
        """
        Compute the item-wise difference between two dictionaries.

        Parameters
        ----------
        dict1 : dict[str, sympl.DataArray]
            The minuend.
        dict2 : dict[str, sympl.DataArray]
            The subtrahend.
        out : `dict[str, sympl.DataArray]`, optional
            The output dictionary.
        field_properties : `dict[str, dict]`, optional
            Dictionary whose keys are strings indicating the variables
            included in the output dictionary, and values are dictionaries
            gathering fundamental properties (units) for those variables.
            If not specified, a variable is included in the output dictionary
            in the same units used in the first input dictionary, or the second
            dictionary if the variable is not present in the first one.
        unshared_variables_in_output : `bool`, optional
            `True` if the output dictionary should contain those variables
            included in only one of the two input dictionaries.

        Return
        ------
        dict[str, sympl.DataArray] :
            The item-wise difference.
        """
        field_properties = field_properties or {}

        out = out or {}
        if "time" in dict1 or "time" in dict2:
            out["time"] = dict1.get("time", dict2.get("time", None))

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.difference(("time",))
        unshared_keys = set(dict1.keys()).symmetric_difference(dict2.keys())
        unshared_keys = unshared_keys.difference(("time",))

        if self._gt_powered and self._stencil_sub is None:
            set_annotations(stencil_sub_defs, self._dtype)
            self._stencil_sub = gtscript.stencil(
                definition=stencil_sub_defs, **self._gt_kwargs
            )

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            if key in out:
                out_da = out[key]
            else:
                out_da = deepcopy_dataarray(dict1[key])

            field1 = dict1[key].to_units(units).values
            field2 = dict2[key].to_units(units).values
            out_da.attrs["units"] = units

            if self._gt_powered:
                self._stencil_sub(
                    in_a=field1,
                    in_b=field2,
                    out_c=out_da.values,
                    origin=(0, 0, 0),
                    domain=field1.shape,
                )
            else:
                out_da.values[...] = field1 - field2

            out[key] = out_da

        if unshared_variables_in_output and len(unshared_keys) > 0:
            if self._gt_powered:
                if self._stencil_copy is None:
                    set_annotations(stencil_copy_defs, self._dtype)
                    self._stencil_copy = gtscript.stencil(
                        definition=stencil_copy_defs, **self._gt_kwargs
                    )
                if self._stencil_copychange is None:
                    set_annotations(stencil_copychange_defs, self._dtype)
                    self._stencil_copychange = gtscript.stencil(
                        definition=stencil_copychange_defs, **self._gt_kwargs
                    )

            for key in unshared_keys:
                props = field_properties.get(key, {})

                if key in dict1:
                    units = props.get("units", dict1[key].attrs["units"])

                    if key in out:
                        if self._gt_powered:
                            self._stencil_copy(
                                src=dict1[key].to_units(units).values,
                                dst=out[key].values,
                                origin=(0, 0, 0),
                                domain=dict1[key].shape,
                            )
                        else:
                            out[key].values[...] = dict1[key].to_units(units).values

                        out[key].attrs["units"] = units
                    else:
                        out[key] = dict1[key].to_units(units)
                else:
                    units = props.get("units", dict2[key].attrs["units"])

                    if key in out:
                        if self._gt_powered:
                            self._stencil_copychange(
                                src=dict2[key].to_units(units).values,
                                dst=out[key].values,
                                origin=(0, 0, 0),
                                domain=dict2[key].shape,
                            )
                        else:
                            out[key].values[...] = -dict2[key].to_units(units).values

                        out[key].attrs["units"] = units
                    else:
                        out[key] = dict2[key].to_units(units)
                        out[key].values *= -1

        return out

    def isub(
        self, dict1, dict2, field_properties=None, unshared_variables_in_output=False
    ):
        """
        In-place variant of `add`.

        Parameters
        ----------
        dict1 : dict[str, sympl.DataArray]
            The minuend, modified in-place.
        dict2 : dict[str, sympl.DataArray]
            The subtrahend.
        out : `dict[str, sympl.DataArray]`, optional
            The output dictionary.
        field_properties : `dict[str, dict]`, optional
            Dictionary whose keys are strings indicating the variables
            included in the output dictionary, and values are dictionaries
            gathering fundamental properties (units) for those variables.
            If not specified, a variable is included in the output dictionary
            in the same units used in the first input dictionary, or the second
            dictionary if the variable is not present in the first one.
        unshared_variables_in_output : `bool`, optional
            `True` if the output dictionary should contain those variables
            included in only one of the two input dictionaries.
        """
        field_properties = field_properties or {}

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.difference(("time",))
        unshared_keys = set(dict1.keys()).symmetric_difference(dict2.keys())
        unshared_keys = unshared_keys.difference(("time",))

        if self._gt_powered and self._stencil_isub is None:
            set_annotations(stencil_isub_defs, self._dtype)
            self._stencil_isub = gtscript.stencil(
                definition=stencil_isub_defs, **self._gt_kwargs
            )

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])
            dict1[key] = dict1[key].to_units(units)
            field1 = dict1[key].values
            field2 = dict2[key].to_units(units).values

            if self._gt_powered:
                self._stencil_isub(
                    inout_a=field1, in_b=field2, origin=(0, 0, 0), domain=field1.shape
                )
            else:
                field1 -= field2

        if unshared_variables_in_output and len(unshared_keys) > 0:
            if self._gt_powered and self._stencil_iscale is None:
                set_annotations(stencil_iscale_defs, self._dtype)
                self._stencil_iscale = gtscript.stencil(
                    definition=stencil_iscale_defs, **self._gt_kwargs
                )

            for key in unshared_keys:
                props = field_properties.get(key, {})

                if key in dict1:
                    units = props.get("units", dict1[key].attrs["units"])
                    dict1[key] = dict1[key].to_units(units)
                else:
                    units = props.get("units", dict2[key].attrs["units"])
                    dict1[key] = deepcopy_dataarray(dict2[key].to_units(units))

                    if self._gt_powered:
                        self._stencil_iscale(
                            inout_a=dict1[key].values,
                            f=-1,
                            origin=(0, 0, 0),
                            domain=dict1[key].shape,
                        )
                    else:
                        dict1[key].values *= -1

    def scale(self, dictionary, factor, out=None, field_properties=None):
        """ TODO """

        field_properties = field_properties or {}

        out = out or {}
        if "time" in dictionary:
            out["time"] = dictionary["time"]

        if self._gt_powered and self._stencil_scale is None:
            set_annotations(stencil_scale_defs, self._dtype)
            self._stencil_scale = gtscript.stencil(
                definition=stencil_scale_defs, **self._gt_kwargs
            )

        for key in dictionary:
            if key == "time":
                out["time"] = dictionary["time"]
            else:
                props = field_properties.get(key, {})
                units = props.get("units", dictionary[key].attrs["units"])
                field = dictionary[key].to_units(units)
                rfield = field.values

                if key in out:
                    out[key].attrs["units"] = units
                else:
                    out[key] = deepcopy_dataarray(field)

                rout = out[key].values

                if self._gt_powered:
                    self._stencil_scale(
                        in_a=rfield,
                        out_a=rout,
                        f=factor,
                        origin=(0, 0, 0),
                        domain=rout.shape,
                    )
                else:
                    rout[...] = factor * rfield

        return out

    def iscale(self, dictionary, factor, field_properties=None):
        """ TODO """

        field_properties = field_properties or {}

        if self._gt_powered and self._stencil_iscale is None:
            set_annotations(stencil_iscale_defs, self._dtype)
            self._stencil_iscale = gtscript.stencil(
                definition=stencil_iscale_defs, **self._gt_kwargs
            )

        for key in dictionary:
            if key != "time":
                props = field_properties.get(key, {})
                units = props.get("units", dictionary[key].attrs["units"])
                dictionary[key] = dictionary[key].to_units(units)
                rfield = dictionary[key].values

                if self._gt_powered:
                    self._stencil_iscale(
                        inout_a=rfield, f=factor, origin=(0, 0, 0), domain=rfield.shape
                    )
                else:
                    rfield[...] *= factor

    def addsub(self, dict1, dict2, dict3, out=None, field_properties=None):
        """ TODO """

        field_properties = field_properties or {}

        out = out or {}
        if "time" in dict1 or "time" in dict2 or "time" in dict3:
            out["time"] = dict1.get("time", dict2.get("time", dict3.get("time", None)))

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.intersection(dict3.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._gt_powered and self._stencil_addsub is None:
            set_annotations(stencil_addsub_defs, self._dtype)
            self._stencil_addsub = gtscript.stencil(
                definition=stencil_addsub_defs, **self._gt_kwargs
            )

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            field1 = dict1[key].to_units(units)
            rfield1 = field1.values
            rfield2 = dict2[key].to_units(units).values
            rfield3 = dict3[key].to_units(units).values

            if key in out:
                out[key].attrs["units"] = units
            else:
                out[key] = deepcopy_dataarray(field1)

            rout = out[key].values

            if self._gt_powered:
                self._stencil_addsub(
                    in_a=rfield1,
                    in_b=rfield2,
                    in_c=rfield3,
                    out_d=rout,
                    origin=(0, 0, 0),
                    domain=rout.shape,
                )
            else:
                rout[...] = rfield1 + rfield2 - rfield3

        return out

    def iaddsub(self, dict1, dict2, dict3, field_properties=None):
        """ TODO """

        field_properties = field_properties or {}

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.intersection(dict3.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._gt_powered and self._stencil_iaddsub is None:
            set_annotations(stencil_iaddsub_defs, self._dtype)
            self._stencil_iaddsub = gtscript.stencil(
                definition=stencil_iaddsub_defs, **self._gt_kwargs
            )

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            dict1[key] = dict1[key].to_units(units)
            rfield1 = dict1[key].values
            rfield2 = dict2[key].to_units(units).values
            rfield3 = dict3[key].to_units(units).values

            if self._gt_powered:
                self._stencil_iaddsub(
                    inout_a=rfield1,
                    in_b=rfield2,
                    in_c=rfield3,
                    origin=(0, 0, 0),
                    domain=rfield1.shape,
                )
            else:
                rfield1 += rfield2 - rfield3

    def fma(self, dict1, dict2, factor, out=None, field_properties=None):
        """ TODO """

        field_properties = field_properties or {}

        out = out or {}
        if "time" in dict1 or "time" in dict2:
            out["time"] = dict1.get("time", dict2.get("time", None))

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._gt_powered and self._stencil_fma is None:
            set_annotations(stencil_fma_defs, self._dtype)
            self._stencil_fma = gtscript.stencil(
                definition=stencil_fma_defs, **self._gt_kwargs
            )

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            field1 = dict1[key].to_units(units)
            rfield1 = field1.values
            rfield2 = dict2[key].to_units(units).values

            if key in out:
                out[key].attrs["units"] = units
            else:
                out[key] = deepcopy_dataarray(field1)

            rout = out[key].values

            if self._gt_powered:
                self._stencil_fma(
                    in_a=rfield1,
                    in_b=rfield2,
                    out_c=rout,
                    f=factor,
                    origin=(0, 0, 0),
                    domain=rout.shape,
                )
            else:
                rout[...] = rfield1 + factor * rfield2

        return out

    def sts_rk2_0(self, dt, state, state_prv, tnd, out=None, field_properties=None):
        """ TODO """

        field_properties = field_properties or {}

        out = out or {}
        if "time" in state or "time" in state_prv:
            out["time"] = state.get("time", state_prv.get("time", None))

        shared_keys = set(state.keys()).intersection(state_prv.keys())
        shared_keys = shared_keys.intersection(tnd.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._gt_powered and self._stencil_sts_rk2_0 is None:
            set_annotations(stencil_sts_rk2_0_defs, self._dtype)
            self._stencil_sts_rk2_0 = gtscript.stencil(
                definition=stencil_sts_rk2_0_defs, **self._gt_kwargs
            )

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", state[key].attrs["units"])

            field = state[key].to_units(units)
            r_field = field.values
            r_field_prv = state_prv[key].to_units(units).values
            r_tnd = tnd[key].to_units(units).values

            if key in out:
                out[key].attrs["units"] = units
            else:
                out[key] = deepcopy_dataarray(field)

            r_out = out[key].values

            if self._gt_powered:
                self._stencil_sts_rk2_0(
                    in_field=r_field,
                    in_field_prv=r_field_prv,
                    in_tnd=r_tnd,
                    out_field=r_out,
                    dt=dt,
                    origin=(0, 0, 0),
                    domain=r_out.shape,
                )
            else:
                r_out[...] = 0.5 * (r_field + r_field_prv + dt * r_tnd)

        return out

    def sts_rk3ws_0(self, dt, state, state_prv, tnd, out=None, field_properties=None):
        """ TODO """

        field_properties = field_properties or {}

        out = out or {}
        if "time" in state or "time" in state_prv:
            out["time"] = state.get("time", state_prv.get("time", None))

        shared_keys = set(state.keys()).intersection(state_prv.keys())
        shared_keys = shared_keys.intersection(tnd.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._gt_powered and self._stencil_sts_rk3ws_0 is None:
            set_annotations(stencil_sts_rk3ws_0_defs, self._dtype)
            self._stencil_sts_rk3ws_0 = gtscript.stencil(
                definition=stencil_sts_rk3ws_0_defs, **self._gt_kwargs
            )

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", state[key].attrs["units"])

            field = state[key].to_units(units)
            r_field = field.values
            r_field_prv = state_prv[key].to_units(units).values
            r_tnd = tnd[key].to_units(units).values

            if key in out:
                out[key].attrs["units"] = units
            else:
                out[key] = deepcopy_dataarray(field)

            r_out = out[key].values

            if self._gt_powered:
                self._stencil_sts_rk3ws_0(
                    in_field=r_field,
                    in_field_prv=r_field_prv,
                    in_tnd=r_tnd,
                    out_field=r_out,
                    dt=dt,
                    origin=(0, 0, 0),
                    domain=r_out.shape,
                )
            else:
                r_out[...] = (2.0 * r_field + r_field_prv + dt * r_tnd) / 3.0

        return out
