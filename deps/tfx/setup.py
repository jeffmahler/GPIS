#!/usr/bin/env python

from distutils.core import setup
#from catkin_pkg.python_setup import generate_distutils_setup

"""
d = generate_distutils_setup(
   ##  don't do this unless you want a globally visible script
   #scripts=['scripts/tf_echo', 'scripts/save_tf', 'scripts/load_tf', 'scripts/topic_echo', 'scripts/publisher', 'scripts/pose_publisher', 'scripts/tf_publisher'], 
   packages=['tfx'],
   package_dir={'': 'src'}
)

setup(**d)
"""

setup(
    packages=['tfx'],
    package_dir={'':'src'}
)
