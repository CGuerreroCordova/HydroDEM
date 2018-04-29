import os
import shutil

from setuptools import setup, find_packages
from setuptools.command.sdist import sdist as _sdist

__author__ = "Cristian Guerrero Cordova"
__copyright__ = "Copyright (c) 2018 Cristian Guerrero Cordova"

NAME = "hydrodem"
PACKAGE_NAME = "hydrodem"
DESCRIPTION = "Processor to conditioner DEM to hydrologic uses."
README_FILE = "README.md"
SCRIPT = ["bin/" + NAME]

VERSION = __import__(PACKAGE_NAME + '.version', globals(), locals(),
                     ['__version__']).__version__

INSTALL_REQUERIMENTS = ["numpy==1.11.0", "configparser==3.5.0", "scipy==1.0.1"]


@staticmethod
def get_readme_file():
    """
    Gets readme file content.
    :return: readme file content, "" if readme file is not present
    :rtype: str
    """
    readme = ""
    readme_path = os.path.join(os.path.dirname(__file__), README_FILE)
    with open(readme_path, 'r') as readme_file:
        readme = readme_file.read()
    return readme


class HydroDEMSdist(_sdist):
    """
    Creates hydrodem distribution package
    """

    def run(self):
        """
        creates hydrodem distribution package
        """
        print "Creating hydrodem Package"
        if os.path.exists("dist"):
            shutil.rmtree("dist")
        egg_folder = NAME + ".egg-info"
        if os.path.exists(egg_folder):
            shutil.rmtree(egg_folder)
        _sdist.run(self)


setup(
    name=NAME,
    version=VERSION,
    license="",
    author="Cristian Guerrero Cordova",
    keywords="DEM Conditioner for hydrologic purposes",
    install_requires=INSTALL_REQUERIMENTS,
    cmdclass={'sdist': HydroDEMSdist},
    author_email="cguerrerocordova@gmail.com",
    description=DESCRIPTION,
    long_description=get_readme_file,
    packages=find_packages(exclude='tests'),
    include_package_data=True,
    platforms='Windows',
    classifiers=[
        'Development Status :: Beta',
        'Programming Language :: Python :: 2.7',
        'Environment :: Console',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    scripts=SCRIPT,
)
