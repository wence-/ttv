from Cython.Build import cythonize
from setuptools import setup, Extension
from setuptools import find_packages
import numpy
import os

extensions = [Extension("ttv.libttv",
                        sources=["ttv/libttv.pyx"],
                        include_dirs=[os.path.abspath("ttv"), numpy.get_include()],
                        libraries=["ttv_c"],
                        library_dirs=[os.path.abspath("ttv")],
                        extra_link_args=[f"-Wl,-rpath,{os.path.abspath('ttv')}"])]                                     

setup(name="libttv",
      version="0.0.1",
      packages=find_packages(),
      ext_modules=cythonize(extensions))
