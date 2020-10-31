from setuptools import find_packages, setup

with open("README.md", "r") as fh:
    long_description = fh.readlines()[3]

setup(
    name='src',
    packages=find_packages(),
    version='0.1.0',
    description='Mapintel project',
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/DavidSilva98/mapintel_project",
    author='davids98',
    license='MIT',
    python_requires='>=3.7'
)
