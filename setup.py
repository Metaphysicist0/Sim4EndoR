import os

import pkg_resources
from setuptools import setup, find_packages

setup(
    name="SofaGW",
    py_modules=["SofaGW"],
    version="1.0.8",
    description="Use SOFA framework for guidewire navigation.",
    url="https://github.com/Metaphysicist0/Sim4EndoR.git",
    author="Metaphysicist0",
    packages=find_packages(),
    install_requires=[
        str(r)
        for r in pkg_resources.parse_requirements(
            open(os.path.join(os.path.dirname(__file__), "requirements.txt"))
        )
    ],
    # include_package_data=True,
    package_data={
        'SofaGW': ['vessel/*'],
    },
)
