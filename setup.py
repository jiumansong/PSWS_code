"""
Some code from "Vision Transformer with Progressive Sampling"
"""
import os
import glob
from setuptools import find_packages, setup
from torch.utils.cpp_extension import BuildExtension, CUDAExtension


def get_extensions():
    extensions = []
    ext_name = '_ext'
    op_files = glob.glob('./layers/csrc/*')
    print(op_files)
    include_path = os.path.abspath('./layers/cinclude')

    extensions.append(CUDAExtension(
        name=ext_name,
        sources=op_files,
        include_dirs=[include_path]
    ))

    return extensions


if __name__ == "__main__":
    setup(
        name='PS_VIT_extension',
        version='new.1',
        description='progressive sampling extension',
        packages=find_packages(),
        ext_modules=get_extensions(),
        cmdclass={'build_ext': BuildExtension},   
        zip_safe=False
    )
