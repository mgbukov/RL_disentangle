import os
import sys
import glob


def cython_files():
    import numpy
    from Cython.Build import cythonize

    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    cython_src = glob.glob(os.path.join(package_dir,"batch_transpose.pyx"))

    include_dirs = [numpy.get_include(),]
    cythonize(cython_src,include_path=include_dirs, language='c++')


def configuration(parent_package='', top_path=None):
    from numpy.distutils.misc_util import Configuration
    import numpy

    lib_name = 'batch_transpose' # name of library
    config = Configuration(lib_name, parent_package, top_path)
    cython_files()

    extra_compile_args=["-fno-strict-aliasing", "-O2"]

    #if sys.platform == "darwin": # can be used to specify the platform
    package_dir = os.path.dirname(os.path.realpath(__file__))
    package_dir = os.path.expandvars(package_dir)

    include_dirs = [numpy.get_include()]
    src = [os.path.join(package_dir, lib_name+".cpp")]

    config.add_extension(   '',
                            sources=src,
                            include_dirs=include_dirs,
                            extra_compile_args=extra_compile_args,
                            language="c++",
                        )

    return config

if __name__ == '__main__':
        from numpy.distutils.core import setup
        import sys
        try:
            instr = sys.argv[1]
            if instr == "build_templates":
                cython_files()
            else:
                setup(**configuration(top_path='').todict())
        except IndexError: pass