from distutils.core import setup
from Cython.Build import cythonize
from distutils.extension import Extension
import numpy
import sys
import os

if sys.platform == 'Windows':
    c_options = ['/O2', '/fp:fast', '/arch:AVX2']
else:
    c_options = ['-Ofast', '-march=native']

extensions = [
    Extension("segment_tree", ["segment_tree.pyx"],
              extra_compile_args=c_options)
]

setup(
    ext_modules=cythonize(extensions,
                          compiler_directives={'language_level': sys.version_info[0]}),
    include_dirs=[numpy.get_include()]
)
