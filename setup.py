import os
import platform
import setuptools


def remove_requirement(requirements, name):
    return [x for x in requirements if name != x.split(' ')[0]]

os = platform.system()

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as req_file:
    requirements = req_file.read().splitlines()

dependency_links = []

# Linux specific requirements
if os == 'Linux':
    requirements = remove_requirement(requirements,'torch')
    requirements.append('torch == 1.1.0')

# OSX specific requirements
if os == 'Darwin':
    requirements = requirements

# Windows specific requirements
if os == 'Windows':
    print('HERE !')
    requirements = remove_requirement(requirements,'torch')
    requirements = remove_requirement(requirements,'torchvision')
    requirements.append('torch == 1.1.0')
    requirements.append('torchvision == 0.3')

    dependency_links.append('https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-win_amd64.whl#egg=torch-1.1.0')
    dependency_links.append('https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp37-cp37m-win_amd64.whl#egg=torchvision-0.3')


setuptools.setup(
    name="lightwood",
    version='0.7.1',
    author="MindsDB Inc",
    author_email="jorge@mindsdb.com",
    description="Lightwood's goal is to make it very simple for developers to use the power of artificial neural networks in their projects. ",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/mindsdb/lightwood",
    packages=setuptools.find_packages(),
    install_requires=requirements,
    dependency_links=dependency_links,
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ),
    python_requires=">=3.6"
)
