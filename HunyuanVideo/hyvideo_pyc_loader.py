"""
Meta-path finder for the hyvideo package when .py sources are absent but
__pycache__/*.cpython-311.pyc files are present.

Install by importing before any hyvideo import:
    import hyvideo_pyc_loader
"""
import sys
import os
import importlib.util
import importlib.machinery

_BASE = os.path.dirname(os.path.abspath(__file__))  # /workspace
_PYC_TAG = "cpython-311"


class _HyvideoSourcelessFinder:
    """Serve hyvideo.* from __pycache__/*.pyc when .py sources are missing."""

    def find_spec(self, fullname, path, target=None):
        if fullname != "hyvideo" and not fullname.startswith("hyvideo."):
            return None

        parts = fullname.split(".")
        pkg_dir = os.path.join(_BASE, *parts)
        init_pyc = os.path.join(pkg_dir, "__pycache__", f"__init__.{_PYC_TAG}.pyc")

        if os.path.isdir(pkg_dir) and os.path.exists(init_pyc):
            loader = importlib.machinery.SourcelessFileLoader(fullname, init_pyc)
            return importlib.util.spec_from_file_location(
                fullname,
                init_pyc,
                loader=loader,
                submodule_search_locations=[pkg_dir],
            )

        if len(parts) >= 2:
            parent_dir = os.path.join(_BASE, *parts[:-1])
            mod_pyc = os.path.join(
                parent_dir, "__pycache__", f"{parts[-1]}.{_PYC_TAG}.pyc"
            )
            if os.path.exists(mod_pyc):
                loader = importlib.machinery.SourcelessFileLoader(fullname, mod_pyc)
                return importlib.util.spec_from_file_location(
                    fullname, mod_pyc, loader=loader
                )

        return None


sys.meta_path.insert(0, _HyvideoSourcelessFinder())
