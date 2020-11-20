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
from copy import deepcopy
import pytest

from tasmania.python.framework import protocol as prt
from tasmania.python.utils.exceptions import ProtocolError
from tasmania.python.utils.protocol_utils import (
    Registry,
    filter_args_list,
    register,
    set_protocol_attributes,
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
            "__functionality__",
            "compiler",
            "__backend__",
            "numpy",
            "__stencil__",
            "foo",
        )
        args_val = deepcopy(args)

        out = tuple(filter_args_list(args))

        assert args == args_val
        assert out == args_val

    def test_add_default(self):
        args = ("__functionality__", "compiler", "__backend__", "numpy")
        args_val = deepcopy(args)

        out = tuple(filter_args_list(args))

        assert args == args_val

        assert len(out) == 6
        assert out[:4] == args_val
        assert out[4] == "__stencil__"
        assert out[5] == prt.catch_all

    def test_missing_attribute(self):
        args = ("__functionality__", "ones", "__stencil__", "bar")

        with pytest.raises(ProtocolError) as excinfo:
            out = filter_args_list(args)
        assert str(excinfo.value) == (
            "The non-default protocol attribute '__backend__' "
            "has not been provided."
        )

    def test_unknown_attribute(self):
        args = ("__functionality__", "zeros", "abcde", "fghil")

        with pytest.raises(ProtocolError) as excinfo:
            out = filter_args_list(args)
        assert str(excinfo.value) == "Unknown protocol attribute 'abcde'."

    def test_invalid_master_attribute_value(self):
        args = ("__functionality__", "abcde", "__backend__", "fghil")

        with pytest.raises(ProtocolError) as excinfo:
            out = filter_args_list(args)
        assert (
            str(excinfo.value) == "Unknown value 'abcde' for master attribute "
            "'__functionality__'."
        )


class TestSetProtocolMethods:
    @staticmethod
    def static_method():
        pass

    def test_static_method(self):
        args = (
            "__functionality__",
            "compiler",
            "__backend__",
            "numpy",
            "__stencil__",
            "foo",
        )

        out = set_protocol_attributes(self.static_method, args)

        assert hasattr(self.static_method, "__functionality__")
        assert self.static_method.__functionality__ == "compiler"
        assert hasattr(self.static_method, "__backend__")
        assert self.static_method.__backend__ == "numpy"
        assert hasattr(self.static_method, "__stencil__")
        assert self.static_method.__stencil__ == "foo"

        assert hasattr(out, "__functionality__")
        assert out.__functionality__ == "compiler"
        assert hasattr(out, "__backend__")
        assert out.__backend__ == "numpy"
        assert hasattr(out, "__stencil__")
        assert out.__stencil__ == "foo"

    @classmethod
    def class_method(cls):
        pass

    def test_class_method(self):
        args = (
            "__functionality__",
            "compiler",
            "__backend__",
            "numpy",
            "__stencil__",
            "foo",
        )

        out = set_protocol_attributes(self.class_method, args)

        assert hasattr(self.__class__.class_method, "__functionality__")
        assert self.__class__.class_method.__functionality__ == "compiler"
        assert hasattr(self.__class__.class_method, "__backend__")
        assert self.__class__.class_method.__backend__ == "numpy"
        assert hasattr(self.__class__.class_method, "__stencil__")
        assert self.__class__.class_method.__stencil__ == "foo"

        assert hasattr(out, "__functionality__")
        assert out.__functionality__ == "compiler"
        assert hasattr(out, "__backend__")
        assert out.__backend__ == "numpy"
        assert hasattr(out, "__stencil__")
        assert out.__stencil__ == "foo"

    def bound_method(self):
        pass

    def test_bound_method(self):
        args = (
            "__functionality__",
            "compiler",
            "__backend__",
            "numpy",
            "__stencil__",
            "foo",
        )

        out = set_protocol_attributes(self.bound_method, args)

        assert hasattr(self.bound_method, "__functionality__")
        assert self.bound_method.__functionality__ == "compiler"
        assert hasattr(self.bound_method, "__backend__")
        assert self.bound_method.__backend__ == "numpy"
        assert hasattr(self.bound_method, "__stencil__")
        assert self.bound_method.__stencil__ == "foo"

        assert hasattr(out, "__functionality__")
        assert out.__functionality__ == "compiler"
        assert hasattr(out, "__backend__")
        assert out.__backend__ == "numpy"
        assert hasattr(out, "__stencil__")
        assert out.__stencil__ == "foo"

    def name_conflict(self):
        pass

    def test_name_conflict(self):
        args = (
            "__functionality__",
            "compiler",
            "__backend__",
            "numpy",
            "__stencil__",
            "foo",
        )
        self.name_conflict.__dict__["__functionality__"] = "abcde"

        with pytest.raises(ProtocolError) as excinfo:
            out = set_protocol_attributes(self.name_conflict, args)
        assert str(excinfo.value) == (
            "Name conflict: Object 'name_conflict' already "
            "sets the attribute '__functionality__' to 'abcde'."
        )

    def test_harmless_name_conflict(self):
        args = (
            "__functionality__",
            "compiler",
            "__backend__",
            "numpy",
            "__stencil__",
            "foo",
        )
        self.name_conflict.__dict__["__functionality__"] = "compiler"

        out = set_protocol_attributes(self.name_conflict, args)

        assert self.name_conflict.__functionality__ == "compiler"
        assert out.__functionality__ == "compiler"

    def test_do_nothing(self):
        args = (
            "__functionality__",
            "compiler",
            "__backend__",
            "numpy",
            "__stencil__",
            "foo",
        )

        self.name_conflict.__dict__["__functionality__"] = "compiler"
        self.name_conflict.__dict__["__backend__"] = "numpy"
        self.name_conflict.__dict__["__stencil__"] = "foo"

        out = set_protocol_attributes(self.bound_method, args)

        assert hasattr(self.bound_method, "__functionality__")
        assert self.bound_method.__functionality__ == "compiler"
        assert hasattr(self.bound_method, "__backend__")
        assert self.bound_method.__backend__ == "numpy"
        assert hasattr(self.bound_method, "__stencil__")
        assert self.bound_method.__stencil__ == "foo"

        assert hasattr(out, "__functionality__")
        assert out.__functionality__ == "compiler"
        assert hasattr(out, "__backend__")
        assert out.__backend__ == "numpy"
        assert hasattr(out, "__stencil__")
        assert out.__stencil__ == "foo"


class TestRegisterFunction:
    @staticmethod
    def compiler_numpy():
        """NumPy's stencil compiler."""
        return "compiler_numpy"

    @staticmethod
    def compiler_gt4py():
        """GT4Py's stencil compiler."""
        return "compiler_gt4py"

    @staticmethod
    def zeros_gt4py():
        """Allocate a storage filled with zeros for a GT4Py stencil."""
        return "zeros_gt4py"

    @staticmethod
    def diffusion_gt4py():
        """GT4Py diffusion stencil."""
        return "diffusion_gt4py"

    @staticmethod
    def diffusion_numba():
        """Numba diffusion stencil."""
        return "diffusion_numba"

    @staticmethod
    def advection_numba():
        """Numba advection stencil."""
        return "advection_numba"

    def test(self):
        reg = Registry()
        register(reg, "__functionality__", "compiler", "__backend__", "numpy")(
            self.compiler_numpy
        )
        register(reg, "__functionality__", "compiler", "__backend__", "gt4py")(
            self.compiler_gt4py
        )
        register(reg, "__functionality__", "zeros", "__backend__", "gt4py")(
            self.zeros_gt4py
        )
        register(
            reg,
            "__functionality__",
            "definition",
            "__backend__",
            "gt4py",
            "__stencil__",
            "diffusion",
        )(self.diffusion_gt4py)
        register(
            reg,
            "__functionality__",
            "definition",
            "__backend__",
            "numba",
            "__stencil__",
            "diffusion",
        )(self.diffusion_numba)
        register(
            reg,
            "__functionality__",
            "definition",
            "__backend__",
            "numba",
            "__stencil__",
            "advection",
        )(self.advection_numba)

        assert "compiler" in reg
        assert "zeros" in reg
        assert "definition" in reg
        assert len(reg) == 3

        assert "numpy" in reg["compiler"]
        assert "gt4py" in reg["compiler"]
        assert len(reg["compiler"]) == 2
        assert "gt4py" in reg["zeros"]
        assert len(reg["zeros"]) == 1
        assert "gt4py" in reg["definition"]
        assert "numba" in reg["definition"]
        assert len(reg["definition"]) == 2

        # compiler_numpy
        assert prt.catch_all in reg[("compiler", "numpy")]
        assert (
            reg[("compiler", "numpy", prt.catch_all)].__doc__
            == "NumPy's stencil compiler."
        )
        assert reg[("compiler", "numpy", prt.catch_all)]() == "compiler_numpy"
        assert len(reg[("compiler", "numpy")]) == 1
        # compiler_gt4py
        assert prt.catch_all in reg[("compiler", "gt4py")]
        assert (
            reg[("compiler", "gt4py", prt.catch_all)].__doc__
            == "GT4Py's stencil compiler."
        )
        assert reg[("compiler", "gt4py", prt.catch_all)]() == "compiler_gt4py"
        assert len(reg[("compiler", "gt4py")]) == 1
        # zeros_gt4py
        assert prt.catch_all in reg[("zeros", "gt4py")]
        assert (
            reg[("zeros", "gt4py", prt.catch_all)].__doc__
            == "Allocate a storage filled with zeros for a GT4Py stencil."
        )
        assert reg[("zeros", "gt4py", prt.catch_all)]() == "zeros_gt4py"
        assert len(reg[("zeros", "gt4py")]) == 1
        # diffusion_gt4py
        assert "diffusion" in reg[("definition", "gt4py")]
        assert (
            reg[("definition", "gt4py", "diffusion")].__doc__
            == "GT4Py diffusion stencil."
        )
        assert reg[("definition", "gt4py", "diffusion")]() == "diffusion_gt4py"
        assert len(reg[("definition", "gt4py")]) == 1
        # diffusion_numba
        assert "diffusion" in reg[("definition", "numba")]
        assert (
            reg[("definition", "numba", "diffusion")].__doc__
            == "Numba diffusion stencil."
        )
        assert reg[("definition", "numba", "diffusion")]() == "diffusion_numba"
        # advection_numba
        assert "advection" in reg[("definition", "numba")]
        assert (
            reg[("definition", "numba", "advection")].__doc__
            == "Numba advection stencil."
        )
        assert reg[("definition", "numba", "advection")]() == "advection_numba"
        assert len(reg[("definition", "numba")]) == 2


reg = Registry()


class TestRegisterDecorator:
    @staticmethod
    @register(reg, "__functionality__", "compiler", "__backend__", "numpy")
    def compiler_numpy():
        """NumPy's stencil compiler."""
        return "compiler_numpy"

    @staticmethod
    @register(reg, "__functionality__", "compiler", "__backend__", "gt4py")
    def compiler_gt4py():
        """GT4Py's stencil compiler."""
        return "compiler_gt4py"

    @staticmethod
    @register(reg, "__functionality__", "zeros", "__backend__", "gt4py")
    def zeros_gt4py():
        """Allocate a storage filled with zeros for a GT4Py stencil."""
        return "zeros_gt4py"

    @staticmethod
    @register(
        reg,
        "__functionality__",
        "definition",
        "__backend__",
        "gt4py",
        "__stencil__",
        "diffusion",
    )
    def diffusion_gt4py():
        """GT4Py diffusion stencil."""
        return "diffusion_gt4py"

    @staticmethod
    @register(
        reg,
        "__functionality__",
        "definition",
        "__backend__",
        "numba",
        "__stencil__",
        "diffusion",
    )
    def diffusion_numba():
        """Numba diffusion stencil."""
        return "diffusion_numba"

    @staticmethod
    @register(
        reg,
        "__functionality__",
        "definition",
        "__backend__",
        "numba",
        "__stencil__",
        "advection",
    )
    def advection_numba():
        """Numba advection stencil."""
        return "advection_numba"

    def test(self):
        assert "compiler" in reg
        assert "zeros" in reg
        assert "definition" in reg
        assert len(reg) == 3

        assert "numpy" in reg["compiler"]
        assert "gt4py" in reg["compiler"]
        assert len(reg["compiler"]) == 2
        assert "gt4py" in reg["zeros"]
        assert len(reg["zeros"]) == 1
        assert "gt4py" in reg["definition"]
        assert "numba" in reg["definition"]
        assert len(reg["definition"]) == 2

        # compiler_numpy
        assert prt.catch_all in reg[("compiler", "numpy")]
        assert (
            reg[("compiler", "numpy", prt.catch_all)].__doc__
            == "NumPy's stencil compiler."
        )
        assert reg[("compiler", "numpy", prt.catch_all)]() == "compiler_numpy"
        assert len(reg[("compiler", "numpy")]) == 1
        # compiler_gt4py
        assert prt.catch_all in reg[("compiler", "gt4py")]
        assert (
            reg[("compiler", "gt4py", prt.catch_all)].__doc__
            == "GT4Py's stencil compiler."
        )
        assert reg[("compiler", "gt4py", prt.catch_all)]() == "compiler_gt4py"
        assert len(reg[("compiler", "gt4py")]) == 1
        # zeros_gt4py
        assert prt.catch_all in reg[("zeros", "gt4py")]
        assert (
            reg[("zeros", "gt4py", prt.catch_all)].__doc__
            == "Allocate a storage filled with zeros for a GT4Py stencil."
        )
        assert reg[("zeros", "gt4py", prt.catch_all)]() == "zeros_gt4py"
        assert len(reg[("zeros", "gt4py")]) == 1
        # diffusion_gt4py
        assert "diffusion" in reg[("definition", "gt4py")]
        assert (
            reg[("definition", "gt4py", "diffusion")].__doc__
            == "GT4Py diffusion stencil."
        )
        assert reg[("definition", "gt4py", "diffusion")]() == "diffusion_gt4py"
        assert len(reg[("definition", "gt4py")]) == 1
        # diffusion_numba
        assert "diffusion" in reg[("definition", "numba")]
        assert (
            reg[("definition", "numba", "diffusion")].__doc__
            == "Numba diffusion stencil."
        )
        assert reg[("definition", "numba", "diffusion")]() == "diffusion_numba"
        # advection_numba
        assert "advection" in reg[("definition", "numba")]
        assert (
            reg[("definition", "numba", "advection")].__doc__
            == "Numba advection stencil."
        )
        assert reg[("definition", "numba", "advection")]() == "advection_numba"
        assert len(reg[("definition", "numba")]) == 2


if __name__ == "__main__":
    pytest.main([__file__])
