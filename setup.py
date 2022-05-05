from setuptools import setup, find_packages
import re
import platform
 
## read and format version from file
VERSIONFILE = "phenomorph/_version.py"
verstrline = open(VERSIONFILE, "rt").read()
VSRE = r"^__version__ = ['\"]([^'\"]*)['\"]"
mo = re.search(VSRE, verstrline, re.M)
if mo:
    verstr = mo.group(1)
else:
    raise RuntimeError("Unable to find version string in %s." % (VERSIONFILE,))

## conditional deps
if platform.system() == "Windows":
    dlib_source = "dlib@https://github.com/phenopype/phenopype-dependencies/blob/main/wheels/dlib-19.23.99-cp37-cp37m-win_amd64.whl?raw=true"
else:
    dlib_source = "dlib"

## setup
setup(
    name="phenomorph",
    url="https://www.phenopype.org",
    author="Arthur Porto",
    author_email="agporto@gmail.com",
    packages=find_packages(),
    install_requires=[
        "phenopype>=3.3.*",
        dlib_source,
    ],
    version=verstr,
    license="LGPL",
    description="A mlmorph module for phenopype",
    entry_points={
        'phenopype.plugins':[
            'phenomorph = phenomorph',
            ],
    }
)