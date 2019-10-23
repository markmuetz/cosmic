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
    version='0.1',
    description='COSMIC package containing tools and analysis',
    long_description=read('README.md'),
    author='Mark Muetzelfeldt',
    author_email='mark.muetzelfeldt@reading.ac.uk',
    maintainer='Mark Muetzelfeldt',
    maintainer_email='mark.muetzelfeldt@reading.ac.uk',
    packages=[
        'cosmic.datasets.UM',
        'cosmic.datasets.cmorph',
        'cosmic.processing',
        ],
    scripts=[
        'bin/cosmic_retrieve_from_mass',
        'bin/cosmic_bsub_submit',
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
