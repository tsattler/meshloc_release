"""
Based on the setup.py script from 
https://github.com/mihaidusmanu/pycolmap/blob/master/setup.py
BSD-3-Clause License 

Note that we never tried compiling it on Windows.
"""
import os
import re
import sys
import platform
import subprocess

from setuptools import setup, Extension
from setuptools.command.build_ext import build_ext
from distutils.version import LooseVersion

import multiprocessing

class CMakeExtension(Extension):
    def __init__(self, name, sourcedir=''):
        Extension.__init__(self, name, sources=[])
        self.sourcedir = os.path.abspath(sourcedir)


class CMakeBuild(build_ext):
    def run(self):
        try:
            out = subprocess.check_output(['cmake', '--version'])
        except OSError:
            raise RuntimeError("CMake must be installed to build the following extensions: " +
                               ", ".join(e.name for e in self.extensions))

        if platform.system() == "Windows":
            cmake_version = LooseVersion(re.search(r'version\s*([\d.]+)', out.decode()).group(1))
            if cmake_version < '3.1.0':
                raise RuntimeError("CMake >= 3.1.0 is required on Windows")

        for ext in self.extensions:
            self.build_extension(ext)

    def build_extension(self, ext):
        extdir = os.path.abspath(os.path.dirname(self.get_ext_fullpath(ext.name)))
        cmake_args = ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY=' + extdir,
                      '-DPYTHON_EXECUTABLE=' + sys.executable]

        cfg = 'Debug' if self.debug else 'Release'
        build_args = ['--config', cfg]

        if platform.system() == "Windows":
            if os.environ.get('CMAKE_TOOLCHAIN_FILE') is not None:
                cmake_toolchain_file = os.environ.get('CMAKE_TOOLCHAIN_FILE')
                print(f'-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file}')
                cmake_args += [f'-DCMAKE_TOOLCHAIN_FILE={cmake_toolchain_file}']
            if os.environ.get('CMAKE_PREFIX_PATH') is not None:
                cmake_prefix_path = os.environ.get('CMAKE_PREFIX_PATH')
                print(f'-DCMAKE_PREFIX_PATH={cmake_prefix_path}')
                cmake_args += [f'-DCMAKE_PREFIX_PATH={cmake_prefix_path}']
            cmake_args += ['-DCMAKE_LIBRARY_OUTPUT_DIRECTORY_{}={}'.format(cfg.upper(), extdir)]
            if sys.maxsize > 2**32:
                cmake_args += ['-DVCPKG_TARGET_TRIPLET=x64-windows']
                cmake_args += ['-A', 'x64']
            build_args += ['--', '/m']
        else:
            cmake_args += ['-DCMAKE_BUILD_TYPE=' + cfg]
            build_args += ['--', '-j{}'.format(multiprocessing.cpu_count() - 1)]

            if platform.system() == "Darwin":
                cmake_args += ['-DOpenMP_CXX_FLAGS="-Xclang -fopenmp"']
                cmake_args += ['-DOpenMP_CXX_LIB_NAMES=libomp']
                cmake_args += ['-DOpenMP_C_FLAGS="-Xclang -fopenmp"']
                cmake_args += ['-DOpenMP_C_LIB_NAMES=libomp']

        env = os.environ.copy()
        env['CXXFLAGS'] = '{} -DVERSION_INFO=\\"{}\\"'.format(
            env.get('CXXFLAGS', ''),
            self.distribution.get_version()
        )
        if not os.path.exists(self.build_temp):
            os.makedirs(self.build_temp)
        print(['cmake', ext.sourcedir] + cmake_args)
        subprocess.check_call(['cmake', ext.sourcedir] + cmake_args, cwd=self.build_temp, env=env)
        subprocess.check_call(['cmake', '--build', '.'] + build_args, cwd=self.build_temp)

setup(
    name='meshloc',
    version='0.1.0',
    author='Torsten Sattler',
    author_email='torsten.sattler@cvut.cz',
    description='Python bindings for localization helpers',
    ext_modules=[CMakeExtension('src')],
    cmdclass=dict(build_ext=CMakeBuild),
    zip_safe=False,
    install_requires=[
        "pyyaml",
        "numpy < 2.0.0",
    ],
)