from setuptools import setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension

setup(
    name="find_first_index_cuda",
    ext_modules=[
        CUDAExtension(
            "find_first_index_cuda",
            sources=["find_first.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)
