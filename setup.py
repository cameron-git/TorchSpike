from setuptools import setup, find_packages, Extension
import torch
from torch.utils import cpp_extension
import os

setup(
    name="torchspike",
    version="0.0.1",
    packages=["torchspike"],
    install_requires=["torch"],
    ext_modules=[
        cpp_extension.CppExtension(
            "torchspike_cpu",
            [
                os.path.join("cpp", file)
                for file in [
                    "lif.cpp",
                ]
            ],
        ), 
        cpp_extension.CUDAExtension(
            "torchspike_cuda",
            [
                os.path.join("cpp", file)
                for file in [
                    "lif_cuda.cpp",
                    "lif_cuda_kernel.cu",
                ]
            ],
        ), 
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
