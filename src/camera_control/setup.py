from setuptools import find_packages, setup

package_name = 'camera_control'

setup(
    name=package_name,
    version='0.0.0',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='ubuntu',
    maintainer_email='vasyap8262@gmail.com',
    description='TODO: Package description',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [ 
            'camera = camera_control.camera_node:main',
            'depth_camera = camera_control.depth_camera_node:main',
            'cam = camera_control.cameras:main',
            'runtime = camera_control.camera_process_pipe:main'
        ],
    },
)
