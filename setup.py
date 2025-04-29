from setuptools import setup, Extension
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
      name="custom_modules", 
      ext_modules=[
          CUDAExtension("custom_modules", ["modules.cpp", "cuda/dit.cu"]),
      ],
      cmdclass={"build_ext": BuildExtension}
)

