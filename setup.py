"""Setup configuration for banana-ripeness-detection package."""

from setuptools import setup, find_packages

setup(
    name="banana-ripeness-detection",
    version="1.0.0",
    packages=find_packages(include=["src", "src.*"]),
    python_requires=">=3.10",
)
