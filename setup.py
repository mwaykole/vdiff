#!/usr/bin/env python
"""Setup script for dfastllm."""

from setuptools import setup, find_packages

if __name__ == "__main__":
    setup(
        packages=find_packages(exclude=["tests", "tests.*", "examples", "docs"]),
    )
