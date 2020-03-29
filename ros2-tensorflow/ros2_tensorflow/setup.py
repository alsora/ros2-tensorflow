from setuptools import find_packages
from setuptools import setup

package_name='ros2_tensorflow'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    install_requires=['setuptools', 'numpy', 'python-opencv'],
    author='Alberto Soragna',
    author_email='alberto.soragna@gmail.com',
    maintainer='Alberto Soragna',
    maintainer_email='alberto.soragna@gmail.com',
    keywords=['ROS'],
    classifiers=[
        'Intended Audience :: Developers',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python',
        'Topic :: Software Development',
    ],
    description='Utilities for working with ROS 2 and Tensorflow',
    license='Apache License, Version 2.0',
    test_suite='test',
    entry_points={
        'console_scripts': [
        ],
    },
)
