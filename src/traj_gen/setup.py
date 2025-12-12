from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'traj_gen'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
        (os.path.join('share', package_name, 'launch'), glob('launch/*.launch.py')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.urdf')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.xacro')),
        (os.path.join('share', package_name, 'urdf'), glob('urdf/*.DAE')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ee106a-adc',
    maintainer_email='jared.haertel@berkeley.edu',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'sim = traj_gen.sim:main',
            'trajectory_planner = traj_gen.trajectory_planner:main',
        ],
    },
)
