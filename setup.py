#! /usr/bin/env python
from setuptools import setup, find_packages

setup(
    name="anesi",
    version="0.0.1",
    description="A-NeSI: Approximate Neurosymbolic Inference",
    url="https://github.com/HEmile/a-nesi",
    author="Emile van Krieken",
    author_email="e.van.Krieken@vu.nl",
    license='MIT',
    classifiers=[
        "License :: OSI Approved :: Apache Software License",
        "Intended Audience :: Science/Research",
        "Programming Language :: Python :: 3.8",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    keywords="",
    package_dir={"": "anesi"},
    packages=find_packages(where="anesi"),
    python_requires='>=3.8',
    zip_safe=False,
    include_package_data=True,
    install_requires=[
        "torch",
        "torchvision",
        "wandb"
    ],
    extras_require={
        "examples": [],
        "tests": ["pytest"],
    },
)
