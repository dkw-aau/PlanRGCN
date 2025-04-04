from setuptools import setup, find_packages

VERSION = "0.0.1"
DESCRIPTION = "Python package with scripts and utility QPP features"
LONG_DESCRIPTION = "Python package with scripts and utility for qpp features construction for baselines"

# Setting up
setup(
    # the name must match the folder name 'verysimplemodule'
    name="qpp_features",
    version=VERSION,
    author="Abiram Mohanaraj",
    author_email="<abiramm@cs.aau.dk>",
    description=DESCRIPTION,
    long_description=LONG_DESCRIPTION,
    packages=find_packages(),
    install_requires=[
        "configparser",
        "numpy",
        "pandas",
        "matplotlib",
    ],  # add any additional packages that
    keywords=["python", "SPARQL"],
    classifiers=[],
)