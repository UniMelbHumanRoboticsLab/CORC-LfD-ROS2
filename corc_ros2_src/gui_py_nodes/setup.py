from setuptools import find_packages, setup

package_name = 'gui_py_nodes'

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
    maintainer='jqloh',
    maintainer_email='jqloh@student.unimelb.edu.au',
    description='TODO: Package description',
    license='Apache-2.0',
    extras_require={
        'test': [
            'pytest',
        ],
    },
    entry_points={
        'console_scripts': [
            'simple_viz = gui_py_nodes.simple_viz:main',
            'corc_viz = gui_py_nodes.corc_viz:main'
        ],
    },
)