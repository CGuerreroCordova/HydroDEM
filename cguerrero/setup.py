import os
import sys
from setuptools import (setup, find_packages, )

NAME = "hydrodem"
SCRIPT = [os.path.join('hydrodem', '__main__.py')]
DESCRIPTION = "HydroDEM"
here = os.path.abspath(os.path.dirname(__file__))
INSTALL_REQUIREMENTS = []


def read(fname):
    return open(os.path.join(os.path.dirname(__file__), fname)).read()


setup(
    name=NAME,
    version="0.1",
    license='CGGC',
    author="CGGC",
    keywords="",
    setup_requires=["pytest-runner"] if 'test' in sys.argv else [],
    tests_require=["pytest==3.8.0", "pytest-cov", "coverage"],
    install_requires=INSTALL_REQUIREMENTS,
    author_email="cguerrerocordova@gmail.com",
    description=DESCRIPTION,
    long_description='Hydrodem',
    packages=find_packages(exclude=['tests*']),
    package_data={},
    include_package_data=True,
    platforms='any',
    classifiers=[],
    scripts=SCRIPT,
)
