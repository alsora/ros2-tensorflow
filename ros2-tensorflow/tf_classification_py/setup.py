from setuptools import find_packages
from setuptools import setup

package_name = 'tf_classification_py'

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
    zip_safe=True,
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
    description=(
        'Python nodes for image classification tasks using Tensorflow.'
    ),
    license='Apache License, Version 2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'server = tf_classification_py.examples.server:main',
            'client_test = tf_classification_py.examples.client_test:main'
        ],
    },
)
