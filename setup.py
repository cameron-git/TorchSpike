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
            "torchspike_lib",
            [
                os.path.join("lib", file)
                for file in [
                    "lif.cpp",
                ]
            ],
        )
    ],
    cmdclass={"build_ext": cpp_extension.BuildExtension},
)
