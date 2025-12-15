from setuptools import setup
from torch.utils.cpp_extension import CUDAExtension, BuildExtension

setup(
    name="custom_flash_attention",
    ext_modules=[
        CUDAExtension(
            name="custom_flash_attention",
            sources=["attention_kernel.cu"],
        ),
    ],
    cmdclass={"build_ext": BuildExtension},
)