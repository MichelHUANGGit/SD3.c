from setuptools import setup, Extension
from torch.utils import cpp_extension

setup(name="custom_modules_cpp", 
      ext_modules=[cpp_extension.CppExtension('custom_modules_cpp', ['modules.cpp'])],
      cmdclass={"build_ext": cpp_extension.BuildExtension}
)

