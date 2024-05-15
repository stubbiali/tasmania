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
from typing import Optional, TYPE_CHECKING

# from sympl._core.time import FakeTimer as Timer

from sympl._core.time import Timer

from tasmania.python.framework.stencil import StencilFactory
from tasmania.python.utils import typingx
from tasmania.python.utils.storage import deepcopy_dataarray

if TYPE_CHECKING:
    from sympl._core.typingx import DataArrayDict

    from tasmania.python.framework.options import (
        BackendOptions,
        StorageOptions,
    )


class DataArrayDictOperator(StencilFactory):
    """Operate on multiple dictionaries of :class:`sympl.DataArray`."""

    def __init__(
        self,
        *,
        backend: str = "numpy",
        backend_options: Optional["BackendOptions"] = None,
        storage_options: Optional["StorageOptions"] = None
    ) -> None:
        """
        Parameters
        ----------
        backend : `str`, optional
            The backend.
        backend_options : `BackendOptions`, optional
            Backend-specific options.
        storage_options : `StorageOptions`, optional
            Storage-related options.
        """
        super().__init__(backend, backend_options, storage_options)

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

    def copy(
        self,
        dst: typingx.DataArrayDict,
        src: typingx.DataArrayDict,
        unshared_variables_in_output: bool = False,
    ) -> None:
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
            ``True`` to include in the destination dictionary the variables not
            originally shared with the source dictionary.
        """
        Timer.start(label="copy")

        if "time" in src:
            dst["time"] = src["time"]

        shared_keys = tuple(key for key in src if key in dst and key != "time")
        unshared_keys = tuple(
            key for key in src if key not in dst and key != "time"
        )

        if self._stencil_copy is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_copy = self.compile_stencil("copy")

        for key in shared_keys:
            assert "units" in dst[key].attrs
            src_field = src[key].to_units(dst[key].attrs["units"]).data
            dst_field = dst[key].data
            Timer.start(label="stencil")
            self._stencil_copy(
                src=src_field,
                dst=dst_field,
                origin=(0, 0, 0),
                domain=src_field.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()

        if unshared_variables_in_output:
            for key in unshared_keys:
                dst[key] = src[key]

        Timer.stop(label="copy")

    def add(
        self,
        dict1: typingx.DataArrayDict,
        dict2: typingx.DataArrayDict,
        out: Optional[typingx.DataArrayDict] = None,
        field_properties: Optional[typingx.PropertiesDict] = None,
        unshared_variables_in_output: bool = False,
    ) -> typingx.DataArrayDict:
        """Add up two dictionaries item-wise.

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
            ``True`` if the output dictionary should contain those variables
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

        if self._stencil_add is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_add = self.compile_stencil("add")

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            if key in out:
                out_da = out[key]
            else:
                out_da = deepcopy_dataarray(dict1[key])

            field1 = dict1[key].to_units(units).data
            field2 = dict2[key].to_units(units).data
            out_da.attrs["units"] = units
            Timer.start(label="stencil")
            self._stencil_add(
                in_a=field1,
                in_b=field2,
                out_c=out_da.data,
                origin=(0, 0, 0),
                domain=field1.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()
            out[key] = out_da

        if unshared_variables_in_output and len(unshared_keys) > 0:
            if self._stencil_copy is None:
                dtype = self.storage_options.dtype
                self.backend_options.dtypes = {"dtype": dtype}
                self._stencil_copy = self.compile_stencil("copy")

            for key in unshared_keys:
                _dict = dict1 if key in dict1 else dict2
                props = field_properties.get(key, {})
                units = props.get("units", _dict[key].attrs["units"])
                if key in out:
                    Timer.start(label="stencil")
                    self._stencil_copy(
                        src=_dict[key].to_units(units).data,
                        dst=out[key].data,
                        origin=(0, 0, 0),
                        domain=_dict[key].shape,
                        exec_info=self.backend_options.exec_info,
                        validate_args=self.backend_options.validate_args,
                    )
                    Timer.stop()
                    out[key].attrs["units"] = units
                else:
                    out[key] = _dict[key].to_units(units)

        return out

    def iadd(
        self,
        dict1: typingx.DataArrayDict,
        dict2: typingx.DataArrayDict,
        field_properties: Optional[typingx.PropertiesDict] = None,
        unshared_variables_in_output: bool = False,
        deepcopy_unshared_variables: bool = False,
    ) -> None:
        """In-place variant of `add`.

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
            ``True`` if the output dictionary should contain those variables
            included in only one of the two input dictionaries.
        deepcopy_unshared_variables : `bool`, optional
        """
        field_properties = field_properties or {}

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.difference(("time",))
        unshared_keys = set(dict1.keys()).symmetric_difference(dict2.keys())
        unshared_keys = unshared_keys.difference(("time",))

        if self._stencil_iadd is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_iadd = self.compile_stencil("iadd")

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])
            dict1[key] = dict1[key].to_units(units)
            field1 = dict1[key].data
            field2 = dict2[key].to_units(units).data
            Timer.start(label="stencil")
            self._stencil_iadd(
                inout_a=field1,
                in_b=field2,
                origin=(0, 0, 0),
                domain=field1.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()

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
        dict1: typingx.DataArrayDict,
        dict2: typingx.DataArrayDict,
        out: Optional[typingx.DataArrayDict] = None,
        field_properties: Optional[typingx.PropertiesDict] = None,
        unshared_variables_in_output: bool = False,
    ) -> typingx.DataArrayDict:
        """Compute the item-wise difference between two dictionaries.

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
            ``True`` if the output dictionary should contain those variables
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

        if self._stencil_sub is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_sub = self.compile_stencil("sub")

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            if key in out:
                out_da = out[key]
            else:
                out_da = deepcopy_dataarray(dict1[key])

            field1 = dict1[key].to_units(units).data
            field2 = dict2[key].to_units(units).data
            out_da.attrs["units"] = units
            Timer.start(label="stencil")
            self._stencil_sub(
                in_a=field1,
                in_b=field2,
                out_c=out_da.data,
                origin=(0, 0, 0),
                domain=field1.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()
            out[key] = out_da

        if unshared_variables_in_output and len(unshared_keys) > 0:
            if self._stencil_copy is None:
                dtype = self.storage_options.dtype
                self.backend_options.dtypes = {"dtype": dtype}
                self._stencil_copy = self.compile_stencil("copy")
            if self._stencil_copychange is None:
                dtype = self.storage_options.dtype
                self.backend_options.dtypes = {"dtype": dtype}
                self._stencil_copychange = self.compile_stencil("copychange")

            for key in unshared_keys:
                props = field_properties.get(key, {})

                if key in dict1:
                    units = props.get("units", dict1[key].attrs["units"])

                    if key in out:
                        Timer.start(label="stencil")
                        self._stencil_copy(
                            src=dict1[key].to_units(units).data,
                            dst=out[key].data,
                            origin=(0, 0, 0),
                            domain=dict1[key].shape,
                            exec_info=self.backend_options.exec_info,
                            validate_args=self.backend_options.validate_args,
                        )
                        Timer.stop()
                        out[key].attrs["units"] = units
                    else:
                        out[key] = dict1[key].to_units(units)
                else:
                    units = props.get("units", dict2[key].attrs["units"])

                    if key in out:
                        Timer.start(label="stencil")
                        self._stencil_copychange(
                            src=dict2[key].to_units(units).data,
                            dst=out[key].data,
                            origin=(0, 0, 0),
                            domain=dict2[key].shape,
                            exec_info=self.backend_options.exec_info,
                            validate_args=self.backend_options.validate_args,
                        )
                        Timer.stop()
                        out[key].attrs["units"] = units
                    else:
                        out[key] = dict2[key].to_units(units)
                        out[key].data *= -1

        return out

    def isub(
        self,
        dict1: typingx.DataArrayDict,
        dict2: typingx.DataArrayDict,
        field_properties: Optional[typingx.PropertiesDict] = None,
        unshared_variables_in_output: bool = False,
    ) -> None:
        """In-place variant of `add`.

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
            ``True`` if the output dictionary should contain those variables
            included in only one of the two input dictionaries.
        """
        field_properties = field_properties or {}

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.difference(("time",))
        unshared_keys = set(dict1.keys()).symmetric_difference(dict2.keys())
        unshared_keys = unshared_keys.difference(("time",))

        if self._stencil_isub is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_isub = self.compile_stencil("isub")

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])
            dict1[key] = dict1[key].to_units(units)
            field1 = dict1[key].data
            field2 = dict2[key].to_units(units).data
            Timer.start(label="stencil")
            self._stencil_isub(
                inout_a=field1,
                in_b=field2,
                origin=(0, 0, 0),
                domain=field1.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()

        if unshared_variables_in_output and len(unshared_keys) > 0:
            if self._stencil_iscale is None:
                dtype = self.storage_options.dtype
                self.backend_options.dtypes = {"dtype": dtype}
                self._stencil_iscale = self.compile_stencil("iscale")

            for key in unshared_keys:
                props = field_properties.get(key, {})

                if key in dict1:
                    units = props.get("units", dict1[key].attrs["units"])
                    dict1[key] = dict1[key].to_units(units)
                else:
                    units = props.get("units", dict2[key].attrs["units"])
                    dict1[key] = deepcopy_dataarray(dict2[key].to_units(units))
                    Timer.start(label="stencil")
                    self._stencil_iscale(
                        inout_a=dict1[key].data,
                        f=-1.0,
                        origin=(0, 0, 0),
                        domain=dict1[key].shape,
                        exec_info=self.backend_options.exec_info,
                        validate_args=self.backend_options.validate_args,
                    )
                    Timer.stop()

    def scale(
        self,
        dictionary: typingx.DataArrayDict,
        factor: float,
        out: Optional[typingx.DataArrayDict] = None,
        field_properties: Optional[typingx.PropertiesDict] = None,
    ) -> typingx.DataArrayDict:
        """TODO"""

        field_properties = field_properties or {}

        out = out or {}
        if "time" in dictionary:
            out["time"] = dictionary["time"]

        if self._stencil_scale is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_scale = self.compile_stencil("scale")

        for key in dictionary:
            if key == "time":
                out["time"] = dictionary["time"]
            else:
                props = field_properties.get(key, {})
                units = props.get("units", dictionary[key].attrs["units"])
                field = dictionary[key].to_units(units)
                rfield = field.data

                if key in out:
                    out[key].attrs["units"] = units
                else:
                    out[key] = deepcopy_dataarray(field)

                rout = out[key].data
                Timer.start(label="stencil")
                self._stencil_scale(
                    in_a=rfield,
                    out_a=rout,
                    f=factor,
                    origin=(0, 0, 0),
                    domain=rout.shape,
                    exec_info=self.backend_options.exec_info,
                    validate_args=self.backend_options.validate_args,
                )
                Timer.stop()

        return out

    def iscale(
        self,
        dictionary: typingx.DataArrayDict,
        factor: float,
        field_properties: Optional[typingx.PropertiesDict] = None,
    ) -> None:
        """TODO"""

        field_properties = field_properties or {}

        if self._stencil_iscale is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_iscale = self.compile_stencil("iscale")

        for key in dictionary:
            if key != "time":
                props = field_properties.get(key, {})
                units = props.get("units", dictionary[key].attrs["units"])
                dictionary[key] = dictionary[key].to_units(units)
                rfield = dictionary[key].data
                Timer.start(label="stencil")
                self._stencil_iscale(
                    inout_a=rfield,
                    f=factor,
                    origin=(0, 0, 0),
                    domain=rfield.shape,
                    exec_info=self.backend_options.exec_info,
                    validate_args=self.backend_options.validate_args,
                )
                Timer.stop()

    def addsub(
        self,
        dict1: typingx.DataArrayDict,
        dict2: typingx.DataArrayDict,
        dict3: typingx.DataArrayDict,
        out: Optional[typingx.DataArrayDict] = None,
        field_properties: Optional[typingx.PropertiesDict] = None,
    ) -> typingx.DataArrayDict:
        """TODO"""

        field_properties = field_properties or {}

        out = out or {}
        if "time" in dict1 or "time" in dict2 or "time" in dict3:
            out["time"] = dict1.get(
                "time", dict2.get("time", dict3.get("time", None))
            )

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.intersection(dict3.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._stencil_addsub is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_addsub = self.compile_stencil("addsub")

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            field1 = dict1[key].to_units(units)
            rfield1 = field1.data
            rfield2 = dict2[key].to_units(units).data
            rfield3 = dict3[key].to_units(units).data

            if key in out:
                out[key].attrs["units"] = units
            else:
                out[key] = deepcopy_dataarray(field1)

            rout = out[key].data
            Timer.start(label="stencil")
            self._stencil_addsub(
                in_a=rfield1,
                in_b=rfield2,
                in_c=rfield3,
                out_d=rout,
                origin=(0, 0, 0),
                domain=rout.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()

        return out

    def iaddsub(
        self,
        dict1: typingx.DataArrayDict,
        dict2: typingx.DataArrayDict,
        dict3: typingx.DataArrayDict,
        field_properties: Optional[typingx.PropertiesDict] = None,
    ) -> None:
        """TODO"""

        field_properties = field_properties or {}

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.intersection(dict3.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._stencil_iaddsub is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_iaddsub = self.compile_stencil("iaddsub")

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            dict1[key] = dict1[key].to_units(units)
            rfield1 = dict1[key].data
            rfield2 = dict2[key].to_units(units).data
            rfield3 = dict3[key].to_units(units).data

            Timer.start(label="stencil")
            self._stencil_iaddsub(
                inout_a=rfield1,
                in_b=rfield2,
                in_c=rfield3,
                origin=(0, 0, 0),
                domain=rfield1.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()

    def fma(
        self,
        dict1: typingx.DataArrayDict,
        dict2: typingx.DataArrayDict,
        factor: float,
        out: Optional[typingx.DataArrayDict] = None,
        field_properties: Optional[typingx.PropertiesDict] = None,
    ) -> typingx.DataArrayDict:
        """TODO"""

        field_properties = field_properties or {}

        out = out or {}
        if "time" in dict1 or "time" in dict2:
            out["time"] = dict1.get("time", dict2.get("time", None))

        shared_keys = set(dict1.keys()).intersection(dict2.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._stencil_fma is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_fma = self.compile_stencil("fma")

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", dict1[key].attrs["units"])

            field1 = dict1[key].to_units(units)
            rfield1 = field1.data
            rfield2 = dict2[key].to_units(units).data

            if key in out:
                out[key].attrs["units"] = units
            else:
                out[key] = deepcopy_dataarray(field1)

            rout = out[key].data
            Timer.start(label="stencil")
            self._stencil_fma(
                in_a=rfield1,
                in_b=rfield2,
                out_c=rout,
                f=factor,
                origin=(0, 0, 0),
                domain=rout.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()

        return out

    def sts_rk2_0(
        self,
        dt: float,
        state: typingx.DataArrayDict,
        state_prv: typingx.DataArrayDict,
        tnd: typingx.DataArrayDict,
        out: Optional[typingx.DataArrayDict] = None,
        field_properties: Optional[typingx.PropertiesDict] = None,
    ) -> typingx.DataArrayDict:
        """TODO"""

        field_properties = field_properties or {}

        out = out or {}
        if "time" in state or "time" in state_prv:
            out["time"] = state.get("time", state_prv.get("time", None))

        shared_keys = set(state.keys()).intersection(state_prv.keys())
        shared_keys = shared_keys.intersection(tnd.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._stencil_sts_rk2_0 is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_sts_rk2_0 = self.compile_stencil("sts_rk2_0")

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", state[key].attrs["units"])

            field = state[key].to_units(units)
            r_field = field.data
            r_field_prv = state_prv[key].to_units(units).data
            r_tnd = tnd[key].to_units(units).data

            if key in out:
                out[key].attrs["units"] = units
            else:
                out[key] = deepcopy_dataarray(field)

            r_out = out[key].data
            Timer.start(label="stencil")
            self._stencil_sts_rk2_0(
                in_field=r_field,
                in_field_prv=r_field_prv,
                in_tnd=r_tnd,
                out_field=r_out,
                dt=dt,
                origin=(0, 0, 0),
                domain=r_out.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()

        return out

    def sts_rk3ws_0(
        self,
        dt: float,
        state: typingx.DataArrayDict,
        state_prv: typingx.DataArrayDict,
        tnd: typingx.DataArrayDict,
        out: Optional[typingx.DataArrayDict] = None,
        field_properties: Optional[typingx.PropertiesDict] = None,
    ) -> typingx.DataArrayDict:
        """TODO"""

        field_properties = field_properties or {}

        out = out or {}
        if "time" in state or "time" in state_prv:
            out["time"] = state.get("time", state_prv.get("time", None))

        shared_keys = set(state.keys()).intersection(state_prv.keys())
        shared_keys = shared_keys.intersection(tnd.keys())
        shared_keys = shared_keys.difference(("time",))

        if self._stencil_sts_rk3ws_0 is None:
            dtype = self.storage_options.dtype
            self.backend_options.dtypes = {"dtype": dtype}
            self._stencil_sts_rk3ws_0 = self.compile_stencil("sts_rk3ws_0")

        for key in shared_keys:
            props = field_properties.get(key, {})
            units = props.get("units", state[key].attrs["units"])

            field = state[key].to_units(units)
            r_field = field.data
            r_field_prv = state_prv[key].to_units(units).data
            r_tnd = tnd[key].to_units(units).data

            if key in out:
                out[key].attrs["units"] = units
            else:
                out[key] = deepcopy_dataarray(field)

            r_out = out[key].data
            Timer.start(label="stencil")
            self._stencil_sts_rk3ws_0(
                in_field=r_field,
                in_field_prv=r_field_prv,
                in_tnd=r_tnd,
                out_field=r_out,
                dt=dt,
                origin=(0, 0, 0),
                domain=r_out.shape,
                exec_info=self.backend_options.exec_info,
                validate_args=self.backend_options.validate_args,
            )
            Timer.stop()

        return out

    @staticmethod
    def update_swap(dct1: "DataArrayDict", dct2: "DataArrayDict") -> None:
        for key in dct2:
            if key in dct1:
                dct1[key], dct2[key] = dct2[key], dct1[key]
            else:
                dct1[key] = dct2[key]
