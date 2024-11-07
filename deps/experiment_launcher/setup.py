from setuptools import setup, find_packages
from codecs import open
from os import path

from experiment_launcher import __version__

here = path.abspath(path.dirname(__file__))

requires_list = []
with open(path.join(here, 'requirements.txt'), encoding='utf-8') as f:
    for line in f:
        requires_list.append(str(line))

long_description = 'experiment_launcher is a Python library used to run experiments.' \
                   ' Supports Joblib and slurm jobs.'

setup(
    name='experiment-launcher',
    version=__version__,
    description='A Python toolkit for launching experiments.',
    long_description=long_description,
    url='https://git.ias.informatik.tu-darmstadt.de/common/experiment_launcher',
    author="Davide Tateo, Joao Carvalho",
    author_email='davide@robot-learning.de, joao@robot-learning.de',
    license='MIT',
    packages=[package for package in find_packages()
              if package.startswith('experiment_launcher')],
    zip_safe=False,
    install_requires=requires_list,
    extras_require={},
    classifiers=["Programming Language :: Python :: 3",
                 "License :: OSI Approved :: MIT License",
                 "Operating System :: OS Independent",
                 ]
)
