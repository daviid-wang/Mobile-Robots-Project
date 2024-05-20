from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'cv_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        # ('share/' + package_name, ['number_contour.py']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='naveennimalan',
    maintainer_email='naveen.nimalan@gmail.com',
    description='TODO: Package description',
    license='Apache-2.0',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'cv_node = cv_package.cv_node:main',
            'number_contour = cv_package.number_contour:main',
            'col_detection = cv_package.col_detection:main'
        ],
    },
)
