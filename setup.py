from setuptools import find_packages, setup

setup(
    name="torchpf",
    version="0.1.0",
    description="torchpf: A Tool for Pytorch Model Analyzer.",
    author="Swall0w, PC Jiang",
    url="https://github.com/2985578957/torchpf.git",

    install_requires = [
        'torch',
        'torchvision',
        'numpy',
        'pandas',
    ],
    tests_requires = [
        'pydocstyle',
    ],
    license='MIT',
    packages=find_packages(exclude=('tests'))
    )
