import os

import pkg_resources
from setuptools import setup, find_packages


setup(
    name="panda",
    py_modules=["panda"],
    version="1.0",
    description="",
    author="Samuel Young",
    packages=find_packages(exclude=["notebooks*"]),
    include_package_data=True,
)
