from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'master_package'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share/', package_name, 'launch'), glob('launch/*launch.py')),
        (os.path.join('share/', package_name, 'config'), glob('config/*.yaml')),
        (os.path.join('share/', package_name, 'robots'), glob('robots/*.urdf'))
        # (os.path.join('share', package_name, 'src'), glob('src/*imu.py')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='group7',
    maintainer_email='group7@todo.todo',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'master_node = master_package.master_node:main',
            'remapping_imu = master_package.remapping_imu:main',
            'remapping_cv = master_package.remapping_cv:main',
            'estop = master_package.estop:main',
            'automatic_estop = master_package.automatic_estop:main',
            'main = master_package.main:main'
        ],
    },
)
