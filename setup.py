#!/usr/bin/env python
import os

try:
    from setuptools import setup, Extension
except ImportError:
    from distutils.core import setup, Extension


def read(fname):
    try:
        return open(os.path.join(os.path.dirname(__file__), fname)).read()
    except:
        return ''


setup(
    name='cosmic',
    version='0.2',
    description='COSMIC package containing tools and analysis',
    long_description=read('README.md'),
    author='Mark Muetzelfeldt',
    author_email='mark.muetzelfeldt@reading.ac.uk',
    maintainer='Mark Muetzelfeldt',
    maintainer_email='mark.muetzelfeldt@reading.ac.uk',
    packages=[
        'cosmic.datasets.UM_N1280',
        'cosmic.datasets.cmorph',
        'cosmic.processing',
        ],
    scripts=[
        'bin/cosmic-rsync-jasmin-data',
        'bin/cosmic-retrieve-from-mass',
        'bin/cosmic-bsub-submit',
        'bin/cosmic-bsub-task-submit',
        ],
    python_requires='>=3.6',
    install_requires=[
        ],
    # url='https://github.com/markmuetz/cosmic',
    classifiers=[
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Natural Language :: English',
        'Operating System :: POSIX :: Linux',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering :: Atmospheric Science',
        ],
    keywords=[''],
    )
