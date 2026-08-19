from setuptools import find_packages, setup

setup(
    name="panda",
    py_modules=["panda"],
    version="1.0",
    description="",
    author="Samuel Young",
    packages=find_packages(exclude=["notebooks*"]),
    include_package_data=True,
)
