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
import collections
from copy import deepcopy
import pytest

from tasmania.python.framework import protocol as prt
from tasmania.python.utils.exceptions import ProtocolError
from tasmania.python.utils.protocol import (
    Registry,
    filter_args_list,
    multiregister,
    singleregister,
    set_attribute,
    set_runtime_attribute,
)


class TestRegistry:
    def test_type_error(self):
        reg = Registry()

        keys = (1, 2.3, {"a", "b"}, {"a": 1, "b": 1})

        for key in keys:
            with pytest.raises(TypeError):
                tmp = reg[key]

        for key in keys:
            with pytest.raises(TypeError):
                reg[key] = ""

    def test_key_error(self):
        reg = Registry()

        keys = ("a", ("a", "b"), ["a", "b", "c"])

        for key in keys:
            with pytest.raises(KeyError):
                tmp = reg[key]

    def test_one_level(self):
        reg = Registry()
        reg["a"] = 1
        reg["b"] = 2

        assert "a" in reg
        assert reg["a"] == 1
        assert "b" in reg
        assert reg["b"] == 2
        assert len(reg) == 2

    def test_two_levels(self):
        reg = Registry()
        reg[("a", "b")] = 1
        reg[("a", "c")] = 2
        reg["c"] = 3

        assert "a" in reg
        assert "c" in reg
        assert len(reg) == 2

        assert isinstance(reg["a"], Registry)
        assert "b" in reg["a"]
        assert reg["a"]["b"] == 1
        assert reg[("a", "b")] == 1
        assert reg[("a",)]["b"] == 1
        assert reg["a"][["b"]] == 1
        assert reg[("a",)][["b"]] == 1
        assert "c" in reg["a"]
        assert reg["a"]["c"] == 2
        assert reg[("a", "c")] == 2
        assert len(reg["a"]) == 2
        assert reg["c"] == 3
        assert reg[("c",)] == 3

    def test_three_levels(self):
        reg = Registry()
        reg[("a", "b")] = 1
        reg[("a", "c")] = 2
        reg[("a", "d", "e")] = 3
        reg[("f", "g", "h")] = 3

        assert "a" in reg
        assert "f" in reg
        assert len(reg) == 2

        assert "b" in reg["a"]
        assert reg["a"]["b"] == 1
        assert "c" in reg["a"]
        assert reg["a"]["c"] == 2
        assert "d" in reg["a"]
        assert len(reg["a"]) == 3
        assert "g" in reg["f"]
        assert len(reg["f"]) == 1

        assert "e" in reg["a"]["d"]
        assert reg["a"]["d"]["e"] == 3
        assert len(reg["a"]["d"]) == 1
        assert "h" in reg["f"]["g"]
        assert reg["f"]["g"]["h"] == 3
        assert len(reg["f"]["g"]) == 1

    def test_regex(self):
        reg = Registry()
        reg["a[a-z]c"] = 1

        assert reg["abc"] == 1

        with pytest.raises(KeyError) as excinfo:
            assert reg["a9c"] == 1
        assert str(excinfo.value) == "\"Key 'a9c' not found.\""


class TestFilterArgsList:
    def test_do_nothing(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        args_val = deepcopy(args)

        out = tuple(filter_args_list(args))

        assert args == args_val
        assert out == args_val

    def test_add_default(self):
        args = ("function", "stencil_compiler", "backend", "numpy")
        args_val = deepcopy(args)

        out = tuple(filter_args_list(args))

        assert args == args_val

        assert len(out) == 6
        assert out[:4] == args_val
        assert out[4] == "stencil"
        assert out[5] == prt.wildcard

    def test_missing_attribute(self):
        args = ("function", "ones", "stencil", "bar")

        with pytest.raises(ProtocolError) as excinfo:
            out = filter_args_list(args)
        assert str(excinfo.value) == (
            "The non-default key 'backend' of the '__tasmania__' dictionary "
            "has not been provided."
        )

    def test_unknown_attribute(self):
        args = ("function", "zeros", "abcde", "fghil")

        with pytest.raises(ProtocolError) as excinfo:
            out = filter_args_list(args)
        assert (
            str(excinfo.value) == "Unknown key 'abcde' in the '__tasmania__' "
            "dictionary."
        )

    def test_invalid_master_attribute_value(self):
        args = ("function", "abcde", "__backend__", "fghil")

        with pytest.raises(ProtocolError) as excinfo:
            out = filter_args_list(args)
        assert (
            str(excinfo.value) == "Unknown value 'abcde' for the master key "
            "'function' of the '__tasmania__' dictionary."
        )


def check_attribute(handle, attr_name, val):
    try:
        attr = getattr(handle, attr_name, {})
    except AttributeError:
        attr = handle.__dict__[attr_name]

    assert isinstance(attr, dict)

    for key in attr:
        assert key in val
        if isinstance(attr[key], str):
            assert isinstance(val[key], str)
            assert attr[key] == val[key]
        elif isinstance(attr[key], collections.abc.Sequence):
            assert isinstance(val[key], collections.abc.Sequence)
            assert all(item in val[key] for item in attr[key])
            assert len(attr[key]) == len(val[key])
        else:
            assert False

    assert len(attr) == len(val)


class TestSetAttribute:
    @staticmethod
    def static_method():
        pass

    def test_static_method(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        val = {args[i]: args[i + 1] for i in range(0, len(args), 2)}

        out = set_attribute(self.static_method, *args)

        check_attribute(self.static_method, prt.attribute, val)
        check_attribute(out, prt.attribute, val)

    @classmethod
    def class_method(cls):
        pass

    def test_class_method(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        val = {args[i]: args[i + 1] for i in range(0, len(args), 2)}

        out = set_attribute(self.class_method, *args)

        check_attribute(self.class_method, prt.attribute, val)
        check_attribute(out, prt.attribute, val)

    def bound_method(self):
        pass

    def test_bound_method(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        val = {args[i]: args[i + 1] for i in range(0, len(args), 2)}

        out = set_attribute(self.bound_method, *args)

        check_attribute(self.bound_method, prt.attribute, val)
        check_attribute(out, prt.attribute, val)

    def name_conflict(self):
        pass

    def test_name_conflict(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )

        self.name_conflict.__dict__[prt.attribute] = "abcde"

        with pytest.raises(ProtocolError) as excinfo:
            out = set_attribute(self.name_conflict, *args)
        assert str(excinfo.value) == (
            f"The object 'name_conflict' already defines the attribute "
            f"'{prt.attribute}' as a non-mapping object."
        )

    def update(self):
        pass

    def test_update(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        val = {args[i]: args[i + 1] for i in range(0, len(args), 2)}

        self.update.__dict__[prt.attribute] = {"bar": "pippo"}
        val["bar"] = "pippo"

        out = set_attribute(self.update, *args)

        check_attribute(self.update, prt.attribute, val)
        check_attribute(out, prt.attribute, val)

    def multiple_calls(self):
        pass

    def test_multiple_calls(self):
        args1 = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        args2 = (
            "function",
            "stencil_compiler",
            "backend",
            "cupy",
            "stencil",
            "foo",
        )
        args3 = (
            "function",
            "stencil_compiler",
            "backend",
            "numba",
            "stencil",
            "bar",
        )
        val = {
            "function": "stencil_compiler",
            "backend": ("cupy", "numba", "numpy"),
            "stencil": ("bar", "foo"),
        }

        _ = set_attribute(self.multiple_calls, *args1)
        _ = set_attribute(self.multiple_calls, *args2)
        out = set_attribute(self.multiple_calls, *args3)

        check_attribute(self.multiple_calls, prt.attribute, val)
        check_attribute(out, prt.attribute, val)


class TestSetRuntimeAttribute:
    @staticmethod
    def static_method():
        pass

    def test_static_method(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        val = {args[i]: args[i + 1] for i in range(0, len(args), 2)}

        out = set_runtime_attribute(self.static_method, *args)

        check_attribute(self.static_method, prt.runtime_attribute, val)
        check_attribute(out, prt.runtime_attribute, val)

    @classmethod
    def class_method(cls):
        pass

    def test_class_method(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        val = {args[i]: args[i + 1] for i in range(0, len(args), 2)}

        out = set_runtime_attribute(self.class_method, *args)

        check_attribute(self.class_method, prt.runtime_attribute, val)
        check_attribute(out, prt.runtime_attribute, val)

    def bound_method(self):
        pass

    def test_bound_method(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        val = {args[i]: args[i + 1] for i in range(0, len(args), 2)}

        out = set_runtime_attribute(self.bound_method, *args)

        check_attribute(self.bound_method, prt.runtime_attribute, val)
        check_attribute(out, prt.runtime_attribute, val)

    def name_conflict(self):
        pass

    def test_name_conflict(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )

        self.name_conflict.__dict__[prt.runtime_attribute] = "abcde"

        with pytest.raises(ProtocolError) as excinfo:
            out = set_runtime_attribute(self.name_conflict, *args)
        assert str(excinfo.value) == (
            f"The object 'name_conflict' already defines the attribute "
            f"'{prt.runtime_attribute}' as a non-mapping object."
        )

    def update(self):
        pass

    def test_update(self):
        args = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        val = {args[i]: args[i + 1] for i in range(0, len(args), 2)}

        self.update.__dict__[prt.runtime_attribute] = {"bar": "pippo"}
        val["bar"] = "pippo"

        out = set_runtime_attribute(self.update, *args)

        check_attribute(self.update, prt.runtime_attribute, val)
        check_attribute(out, prt.runtime_attribute, val)

    def multiple_calls(self):
        pass

    def test_multiple_calls(self):
        args1 = (
            "function",
            "stencil_compiler",
            "backend",
            "numpy",
            "stencil",
            "foo",
        )
        args2 = (
            "function",
            "stencil_compiler",
            "backend",
            "cupy",
            "stencil",
            "foo",
        )
        args3 = (
            "function",
            "stencil_compiler",
            "backend",
            "numba",
            "stencil",
            "bar",
        )
        val = {args3[i]: args3[i + 1] for i in range(0, len(args3), 2)}

        _ = set_runtime_attribute(self.multiple_calls, *args1)
        _ = set_runtime_attribute(self.multiple_calls, *args2)
        out = set_runtime_attribute(self.multiple_calls, *args3)

        check_attribute(self.multiple_calls, prt.runtime_attribute, val)
        check_attribute(out, prt.runtime_attribute, val)


class TestSingleregister:
    registry_decorator = Registry()
    registry_function = Registry()

    @staticmethod
    @singleregister(
        registry=registry_decorator,
        args=("function", "stencil_compiler", "backend", "numpy"),
    )
    def compiler_numpy():
        """NumPy's stencil compiler."""
        return "compiler_numpy"

    @staticmethod
    @singleregister(
        registry=registry_decorator,
        args=("function", "stencil_compiler", "backend", "gt4py"),
    )
    def compiler_gt4py():
        """GT4Py's stencil compiler."""
        return "compiler_gt4py"

    @staticmethod
    @singleregister(
        registry=registry_decorator,
        args=("function", "zeros", "backend", "gt4py"),
    )
    def zeros_gt4py():
        """Allocate a storage filled with zeros for a GT4Py stencil."""
        return "zeros_gt4py"

    @staticmethod
    @singleregister(
        registry=registry_decorator,
        args=(
            "function",
            "stencil_definition",
            "backend",
            "gt4py",
            "stencil",
            "diffusion",
        ),
    )
    def diffusion_gt4py():
        """GT4Py diffusion stencil."""
        return "diffusion_gt4py"

    @staticmethod
    @singleregister(
        registry=registry_decorator,
        args=(
            "function",
            "stencil_definition",
            "backend",
            "numba",
            "stencil",
            "diffusion",
        ),
    )
    def diffusion_numba():
        """Numba diffusion stencil."""
        return "diffusion_numba"

    @staticmethod
    @singleregister(
        registry=registry_decorator,
        args=(
            "function",
            "stencil_definition",
            "backend",
            "numba",
            "stencil",
            "advection",
        ),
    )
    def advection_numba():
        """Numba advection stencil."""
        return "advection_numba"

    def check_register(self, reg):
        assert "stencil_compiler" in reg

        assert "zeros" in reg
        assert "stencil_definition" in reg
        assert len(reg) == 3

        assert "numpy" in reg["stencil_compiler"]
        assert "gt4py" in reg["stencil_compiler"]
        assert len(reg["stencil_compiler"]) == 2
        assert "gt4py" in reg["zeros"]
        assert len(reg["zeros"]) == 1
        assert "gt4py" in reg["stencil_definition"]
        assert "numba" in reg["stencil_definition"]
        assert len(reg["stencil_definition"]) == 2

        # compiler_numpy
        assert prt.wildcard in reg[("stencil_compiler", "numpy")]
        obj = reg[("stencil_compiler", "numpy", prt.wildcard)]
        assert obj.__doc__ == "NumPy's stencil compiler."
        assert obj() == "compiler_numpy"
        assert id(obj) == id(self.compiler_numpy)
        assert len(reg[("stencil_compiler", "numpy")]) == 1
        # compiler_gt4py
        assert prt.wildcard in reg[("stencil_compiler", "gt4py")]
        obj = reg[("stencil_compiler", "gt4py", prt.wildcard)]
        assert obj.__doc__ == "GT4Py's stencil compiler."
        assert obj() == "compiler_gt4py"
        assert id(obj) == id(self.compiler_gt4py)
        assert len(reg[("stencil_compiler", "gt4py")]) == 1
        # zeros_gt4py
        assert prt.wildcard in reg[("zeros", "gt4py")]
        obj = reg[("zeros", "gt4py", prt.wildcard)]
        assert (
            obj.__doc__
            == "Allocate a storage filled with zeros for a GT4Py stencil."
        )
        assert obj() == "zeros_gt4py"
        assert id(obj) == id(self.zeros_gt4py)
        assert len(reg[("zeros", "gt4py")]) == 1
        # diffusion_gt4py
        assert "diffusion" in reg[("stencil_definition", "gt4py")]
        obj = reg[("stencil_definition", "gt4py", "diffusion")]
        assert obj.__doc__ == "GT4Py diffusion stencil."
        assert obj() == "diffusion_gt4py"
        assert id(obj) == id(self.diffusion_gt4py)
        assert len(reg[("stencil_definition", "gt4py")]) == 1
        # diffusion_numba
        assert "diffusion" in reg[("stencil_definition", "numba")]
        obj = reg[("stencil_definition", "numba", "diffusion")]
        assert obj.__doc__ == "Numba diffusion stencil."
        assert obj() == "diffusion_numba"
        assert id(obj) == id(self.diffusion_numba)
        # advection_numba
        assert "advection" in reg[("stencil_definition", "numba")]
        obj = reg[("stencil_definition", "numba", "advection")]
        assert obj.__doc__ == "Numba advection stencil."
        assert obj() == "advection_numba"
        assert id(obj) == id(self.advection_numba)
        assert len(reg[("stencil_definition", "numba")]) == 2

    def test_function(self):
        reg = self.registry_function
        singleregister(
            self.compiler_numpy,
            reg,
            ("function", "stencil_compiler", "backend", "numpy"),
        )
        singleregister(
            self.compiler_gt4py,
            reg,
            ("function", "stencil_compiler", "backend", "gt4py"),
        )
        singleregister(
            self.zeros_gt4py, reg, ("function", "zeros", "backend", "gt4py")
        )
        singleregister(
            self.diffusion_gt4py,
            reg,
            (
                "function",
                "stencil_definition",
                "backend",
                "gt4py",
                "stencil",
                "diffusion",
            ),
        )
        singleregister(
            self.diffusion_numba,
            reg,
            (
                "function",
                "stencil_definition",
                "backend",
                "numba",
                "stencil",
                "diffusion",
            ),
        )
        singleregister(
            self.advection_numba,
            reg,
            (
                "function",
                "stencil_definition",
                "backend",
                "numba",
                "stencil",
                "advection",
            ),
        )
        self.check_register(reg)

    def test_decorator(self):
        self.check_register(self.registry_decorator)


class TestMultiregister:
    registry_decorator = Registry()
    registry_function = Registry()

    @staticmethod
    @multiregister(
        registry=registry_decorator,
        args=("function", "stencil_compiler", "backend", ("numpy", "cupy")),
    )
    def compiler_numpy():
        """NumPy's stencil compiler."""
        return "compiler_numpy"

    @staticmethod
    @multiregister(
        registry=registry_decorator,
        args=("function", "stencil_compiler", "backend", "gt4py"),
    )
    def compiler_gt4py():
        """GT4Py's stencil compiler."""
        return "compiler_gt4py"

    @staticmethod
    @singleregister(
        registry=registry_decorator,
        args=("function", "zeros", "backend", "numpy"),
    )
    @singleregister(
        registry=registry_decorator,
        args=("function", "zeros", "backend", "numba"),
    )
    def zeros_numpy():
        """Allocate a storage filled with zeros for a NumPy stencil."""
        return "zeros_numpy"

    @staticmethod
    @singleregister(
        registry=registry_decorator,
        args=("function", "zeros", "backend", "gt4py"),
    )
    def zeros_gt4py():
        """Allocate a storage filled with zeros for a GT4Py stencil."""
        return "zeros_gt4py"

    @staticmethod
    @multiregister(
        registry=registry_decorator,
        args=("function", "ones", "backend", "numpy"),
    )
    @multiregister(
        registry=registry_decorator,
        args=("function", "ones", "backend", "numba"),
    )
    def ones_numpy():
        """Allocate a storage filled with ones for a NumPy stencil."""
        return "ones_numpy"

    def check_register(self, reg):
        assert "stencil_compiler" in reg
        assert "zeros" in reg
        assert "ones" in reg
        assert len(reg) == 3

        assert "numpy" in reg["stencil_compiler"]
        assert "cupy" in reg["stencil_compiler"]
        assert "gt4py" in reg["stencil_compiler"]
        assert len(reg["stencil_compiler"]) == 3
        assert "numpy" in reg["zeros"]
        assert "numba" in reg["zeros"]
        assert "gt4py" in reg["zeros"]
        assert len(reg["zeros"]) == 3
        assert "numpy" in reg["ones"]
        assert "numba" in reg["ones"]
        assert len(reg["ones"]) == 2

        # compiler_numpy (numpy)
        assert prt.wildcard in reg[("stencil_compiler", "numpy")]
        obj = reg[("stencil_compiler", "numpy", prt.wildcard)]
        assert obj.__doc__ == "NumPy's stencil compiler."
        assert obj() == "compiler_numpy"
        assert id(obj) == id(self.compiler_numpy)
        assert len(reg[("stencil_compiler", "numpy")]) == 1
        # compiler_numpy (cupy)
        assert prt.wildcard in reg[("stencil_compiler", "cupy")]
        obj = reg[("stencil_compiler", "cupy", prt.wildcard)]
        assert obj.__doc__ == "NumPy's stencil compiler."
        assert obj() == "compiler_numpy"
        assert id(obj) == id(self.compiler_numpy)
        assert len(reg[("stencil_compiler", "cupy")]) == 1
        # compiler_gt4py
        assert prt.wildcard in reg[("stencil_compiler", "gt4py")]
        obj = reg[("stencil_compiler", "gt4py", prt.wildcard)]
        assert obj.__doc__ == "GT4Py's stencil compiler."
        assert obj() == "compiler_gt4py"
        assert id(obj) == id(self.compiler_gt4py)
        assert len(reg[("stencil_compiler", "gt4py")]) == 1
        # zeros_numpy (numpy)
        assert prt.wildcard in reg[("zeros", "numpy")]
        obj = reg[("zeros", "numpy", prt.wildcard)]
        assert (
            obj.__doc__
            == "Allocate a storage filled with zeros for a NumPy stencil."
        )
        assert obj() == "zeros_numpy"
        assert id(obj) == id(self.zeros_numpy)
        assert len(reg[("zeros", "numpy")]) == 1
        # zeros_numpy (numba)
        assert prt.wildcard in reg[("zeros", "numba")]
        obj = reg[("zeros", "numba", prt.wildcard)]
        assert (
            obj.__doc__
            == "Allocate a storage filled with zeros for a NumPy stencil."
        )
        assert obj() == "zeros_numpy"
        assert id(obj) == id(self.zeros_numpy)
        assert len(reg[("zeros", "numba")]) == 1
        # zeros_gt4py
        assert prt.wildcard in reg[("zeros", "gt4py")]
        obj = reg[("zeros", "gt4py", prt.wildcard)]
        assert (
            obj.__doc__
            == "Allocate a storage filled with zeros for a GT4Py stencil."
        )
        assert obj() == "zeros_gt4py"
        assert id(obj) == id(self.zeros_gt4py)
        assert len(reg[("zeros", "gt4py")]) == 1
        # ones_numpy (numpy)
        assert prt.wildcard in reg[("ones", "numpy")]
        obj = reg[("ones", "numpy", prt.wildcard)]
        assert (
            obj.__doc__
            == "Allocate a storage filled with ones for a NumPy stencil."
        )
        assert obj() == "ones_numpy"
        assert id(obj) == id(self.ones_numpy)
        assert len(reg[("ones", "numpy")]) == 1
        # ones_numpy (numba)
        assert prt.wildcard in reg[("ones", "numba")]
        obj = reg[("ones", "numba", prt.wildcard)]
        assert (
            obj.__doc__
            == "Allocate a storage filled with ones for a NumPy stencil."
        )
        assert obj() == "ones_numpy"
        assert id(obj) == id(self.ones_numpy)
        assert len(reg[("ones", "numba")]) == 1

    def test_function(self):
        reg = self.registry_function
        multiregister(
            self.compiler_numpy,
            reg,
            ("function", "stencil_compiler", "backend", ("cupy", "numpy")),
        )
        multiregister(
            self.compiler_gt4py,
            reg,
            ("function", "stencil_compiler", "backend", "gt4py"),
        )
        singleregister(
            self.zeros_gt4py, reg, ("function", "zeros", "backend", "gt4py")
        )
        singleregister(
            self.zeros_numpy, reg, ("function", "zeros", "backend", "numba")
        )
        singleregister(
            self.zeros_numpy, reg, ("function", "zeros", "backend", "numpy")
        )
        multiregister(
            self.ones_numpy, reg, ("function", "ones", "backend", "numpy",),
        )
        multiregister(
            self.ones_numpy, reg, ("function", "ones", "backend", "numba",),
        )
        self.check_register(reg)

    def test_decorator(self):
        self.check_register(self.registry_decorator)


if __name__ == "__main__":
    pytest.main([__file__])
