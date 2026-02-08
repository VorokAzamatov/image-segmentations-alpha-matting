from setuptools import setup, find_packages

setup(
    name="image_segmentation",
    version="0.1",
    packages=find_packages(where="src"),
    package_dir={"": "src"},
)