# File: setup.py
# Created Date: 2020-04-08
# Author: Steven Atkinson (steven@atkinson.mn)

from setuptools import setup, find_packages
import sys

requirements = []

setup(
    name="bhpm",
    version="0.0.0",
    description="Bayesian Hidden Physics Models",
    author="Steven Atkinson",
    author_email="steven@atkinson.mn",
    url="https://github.com/sdatkinson/BHPM-Ultrasound",
    install_requires=requirements,
    packages=find_packages(),
)
