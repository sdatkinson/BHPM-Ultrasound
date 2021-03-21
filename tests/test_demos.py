# File: test_demos.py
# Created Date: 2020-04-11
# Author: Steven Atkinson (steven@atkinson.mn)

import importlib
import inspect
import os
import sys

import pytest

base_path = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

jax_dirs = ["examples", "jax"]
torch_dirs = ["examples", "torch"]


def generate(base, dirs):
    @pytest.mark.demo
    def wrapped():
        path = os.path.join(base_path, *dirs)
        need_path = path not in sys.path
        if need_path:
            sys.path.append(path)

        m = importlib.import_module(".".join(dirs + [base]))
        assert hasattr(m, "main"), "Need a main()"
        main_kwargs = (
            {"testing": True}
            if "testing" in inspect.signature(m.main).parameters
            else {}
        )
        m.main(**main_kwargs)

        if need_path:
            sys.path.pop(-1)

    return wrapped


locals().update({"test_" + name: generate(name, jax_dirs) for name in ["mo_kernel"]})

# locals().update(
#     {
#         "test_" + name: generate(name, torch_dirs)
#         for name in ["multi_output", "1st_order_pde"]
#     }
# )
