#!/usr/bin/env python
"""
Modified setup.py — Cython extension removed so insightface 0.7.3 installs
on Windows without requiring Microsoft Visual C++ Build Tools.

The removed extension (mesh_core_cython) is only used by the 3D face
reconstruction module, which is NOT required by this project.
"""
import os
import io
import glob
import re
import sys
from setuptools import setup, find_packages

def read(*names, **kwargs):
    with io.open(os.path.join(os.path.dirname(__file__), *names),
                 encoding=kwargs.get("encoding", "utf8")) as fp:
        return fp.read()

def find_version(*file_paths):
    version_file = read(*file_paths)
    version_match = re.search(r"^__version__ = ['\"]([^'\"]*)['\"]\",",
                              version_file, re.M)
    if version_match:
        return version_match.group(1)
    # Fallback: read directly
    try:
        return read(*file_paths).split("__version__")[1].split("'")[1]
    except Exception:
        return "0.7.3"

try:
    import pypandoc
    long_description = pypandoc.convert_file('README.md', 'rst')
except (IOError, ImportError, ModuleNotFoundError):
    long_description = open('README.md').read()

VERSION = "0.7.3"

requirements = [
    'numpy',
    'onnx',
    'tqdm',
    'requests',
    'matplotlib',
    'Pillow',
    'scipy',
    'scikit-learn',
    'scikit-image',
    'easydict',
]

data_images  = list(glob.glob('insightface/data/images/*.jpg'))
data_images += list(glob.glob('insightface/data/images/*.png'))
data_objects = list(glob.glob('insightface/data/objects/*.pkl'))

data_files  = [('insightface/data/images',  data_images)]
data_files += [('insightface/data/objects', data_objects)]

setup(
    name='insightface',
    version=VERSION,
    author='InsightFace Contributors',
    author_email='contact@insightface.ai',
    url='https://github.com/deepinsight/insightface',
    description='InsightFace Python Library',
    long_description=long_description,
    long_description_content_type='text/markdown',
    license='MIT',
    packages=find_packages(exclude=('docs', 'tests', 'scripts')),
    data_files=data_files,
    zip_safe=True,
    include_package_data=True,
    entry_points={"console_scripts": ["insightface-cli=insightface.commands.insightface_cli:main"]},
    install_requires=requirements,
    # NOTE: ext_modules (mesh_core_cython) intentionally omitted —
    # requires MSVC and is only used for 3D face reconstruction.
)
