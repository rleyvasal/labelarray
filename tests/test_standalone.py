"""labelarray needs NumPy only.

The library imports the standard library, NumPy, and itself when it is
imported; xarray (for ``to_xarray``) and Numba (the ``fast`` extra) only
inside the functions that use them. It runs without xarray and pandas.
"""

import ast
import inspect
import subprocess
import sys
import textwrap
from pathlib import Path

import labelarray

LIBRARY = Path(inspect.getfile(labelarray)).parent
OPTIONAL = {"xarray", "pandas", "numba"}


def _top_level_imports(tree: ast.Module) -> set[str]:
    names = set()
    for node in tree.body:
        if isinstance(node, ast.Import):
            names |= {alias.name.split(".")[0] for alias in node.names}
        elif isinstance(node, ast.ImportFrom) and node.module and not node.level:
            names.add(node.module.split(".")[0])
    return names


def test_the_library_imports_only_the_standard_library_numpy_and_itself_at_import_time():
    for path in LIBRARY.glob("*.py"):
        if path.name == "_compiled.py":
            continue  # the Numba kernels, imported only when Numba is installed
        outside = {name for name in _top_level_imports(ast.parse(path.read_text()))
                   if name not in sys.stdlib_module_names and name not in {"numpy", "labelarray"}}
        assert not outside, f"{path.name} imports {outside}"


def test_importing_the_library_imports_no_optional_dependency():
    code = f"import sys, labelarray; assert not {OPTIONAL!r} & set(sys.modules), sorted({OPTIONAL!r} & set(sys.modules))"
    result = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr


WITHOUT_XARRAY_OR_PANDAS = textwrap.dedent("""
    import importlib.abc, sys

    class Blocked(importlib.abc.MetaPathFinder):
        def find_spec(self, name, path=None, target=None):
            if name.split(".")[0] in ("xarray", "pandas"):
                raise ImportError(f"{name} is not installed")

    sys.meta_path.insert(0, Blocked())

    import numpy as np
    from labelarray import Array, Collection, Grouping, grouped

    a = Array(np.arange(6.0).reshape(3, 2), ("k", "band"), {"k": [3, 1, 2]})
    b = Array([30.0, 10.0, 20.0], ("k",), {"k": [3, 1, 2]})
    left = Collection({"a": a}, a.coords)
    right = Collection({"b": b.isel(k=[1, 2])}, {"k": [1, 2]})
    joined = left.join(right, "inner")
    assert joined.coords["k"].tolist() == [1, 2]
    means = grouped.reduce(a + b, "mean", Grouping(np.array([0, 1, 0]), ["even", "odd"]), "k", "parity")
    assert means.data.tolist() == [[27.0, 28.0], [12.0, 13.0]]
    assert not {"xarray", "pandas"} & {name.split(".")[0] for name in sys.modules}
""")


def test_the_library_runs_without_xarray_or_pandas(tmp_path):
    script = tmp_path / "alone.py"
    script.write_text(WITHOUT_XARRAY_OR_PANDAS)
    result = subprocess.run([sys.executable, str(script)], capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
