from setuptools import find_packages, setup

setup(
    name='pamo_safe_project',
    version='0.0.0',
    packages=find_packages(where='src'),
    package_dir={'':'src'},
    install_requires=[
        'numpy>=1.26.3',
        'scipy>=1.12.0',
        'trimesh>=4.0.10',
        'libigl>=2.5.1',
    ],
)
