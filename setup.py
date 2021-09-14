import setuptools
from cr3bp import __version__

with open("README.md") as f:
    long_desc = f.read()

setuptools.setup(
    name="cr3bp",
    version=__version__,
    author="Nickolai Belakovski",
    description="A small library for exploring the Circular Restricted 3-Body Problem (CR3BP)",
    long_description=long_desc,
    long_description_content_type="text/markdown",
    url="https://github.com/nbelakovski/cr3bp",
    packages=['cr3bp'],
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: Free For Educational Use",
        "Operating System :: OS Independent"
    ],
    python_requires='>=3.6'
)