from setuptools import find_packages
from setuptools import setup

package_name = 'ros2_tf_core'

setup(
    name=package_name,
    version='0.0.2',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
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
    tests_require=['pytest'],
    test_suite='test',
    entry_points={
        'console_scripts': [
        ],
    },
)
