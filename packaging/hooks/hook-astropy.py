# Replaces the stock hook, which imports every astropy submodule (and trips over the optional matplotlib one).
# firstlight only reads and writes FITS files.
from PyInstaller.utils.hooks import collect_data_files, collect_submodules

hiddenimports = collect_submodules("astropy.io.fits") + collect_submodules("astropy.utils") + collect_submodules("astropy.constants") + collect_submodules("astropy.units") + collect_submodules("astropy.config")
datas = collect_data_files("astropy", include_py_files=True, includes=["CITATION", "*.cfg", "*.rst", "units/format/*parsetab.py", "units/format/*lextab.py"])
