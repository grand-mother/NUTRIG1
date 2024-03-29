from setuptools import setup
import sys

sys.path.append('shower_radio/src')

import sradio
from pip._internal.req import parse_requirements

# requirements = parse_requirements("shower_radio/requirements_novers.txt", session="hack")
# requires = [str(item.req) for item in requirements]
# print (requires)

def parse_requirements(filename):
    """ load requirements from a pip requirements file """
    lineiter = (line.strip() for line in open(filename))
    return [line for line in lineiter if line and not line.startswith("#")]

#print (parse_requirements("shower_radio/requirements_novers.txt"))
setup(
    name="ShowerRadio",
    description="Tools for radio traces from air shower",
    version=sradio.__version__,
    author=sradio.__author__,
    classifiers=[
        "Topic :: Scientific/Engineering :: Astronomy",
        "Topic :: Scientific/Engineering :: Physics",
    ],
    url="https://github.com/grand-mother/NUTRIG1",
    package_dir={"sradio": "shower_radio/src/sradio"},
    scripts=["shower_radio/src/scripts/zhaires_view.py"],
    license='MIT', 
    python_requires='>=3.4', 
    #install_requires=["numpy","scipy","matplotlib","asdf","h5py"]
    install_requires=parse_requirements("shower_radio/requirements_novers.txt")
)
