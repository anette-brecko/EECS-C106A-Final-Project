from setuptools import find_packages, setup
from glob import glob
import os

package_name = 'planning'

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
    maintainer='ee106a-tah',
    maintainer_email='danielmunicio360@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'main = planning.main:main',
            'tf = planning.static_tf_transform:main',
            'ik = planning.ik:main',
            'transform_cube_pose = planning.transform_cube_pose:main',
            'test = planning.test_launch:main',
            'tickle_balls = planning.test_grab:main',
            'replay = planning.replay_test:main',
            'sim = planning.sim:main'
        ],
    },
)
