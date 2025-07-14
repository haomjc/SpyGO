# setup.py

from setuptools import setup, find_packages

setup(
    name='easyplot',
    version='0.2.0',
    packages=find_packages(),
    install_requires=[
        'matplotlib',
        'pyvista'
    ],
    description='Easy plotting functions with MATLAB-like interface, wrapping matplotlib and pyvista, with advanced grouping transformations.',
    author='Your Name',
    author_email='your.email@example.com',
    url='https://github.com/yourusername/easyplot',  # update with your URL if available
    classifiers=[
        'Programming Language :: Python :: 3',
        'Operating System :: OS Independent',
    ],
)
