import os
import platform
import setuptools


about = {}
with open("lightwood/__about__.py") as fp:
    exec(fp.read(), about)


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
    requirements = remove_requirement(requirements,'torch')
    requirements = remove_requirement(requirements,'torchvision')
    requirements.append('torch == 1.1.0.0')
    requirements.append('torchvision == 0.3.0.0')

    #dependency_links.append('https://download.pytorch.org/whl/cpu/torch-1.1.0-cp37-cp37m-win_amd64.whl#egg=torch-1.1.0.0')
    #dependency_links.append('https://download.pytorch.org/whl/cpu/torchvision-0.3.0-cp37-cp37m-win_amd64.whl#egg=torchvision-0.3.0.0')
    dependency_links.append('https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl#egg=torch-1.1.0.0')
    dependency_links.append('https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl#egg=torchvision-0.3.0.0')

setuptools.setup(
    name=about['__title__'],
    version=about['__version__'],
    url=about['__github__'],
    download_url=about['__pypi__'],
    license=about['__license__'],
    author=about['__author__'],
    author_email=about['__email__'],
    description=about['__description__'],
    long_description=long_description,
    long_description_content_type="text/markdown",
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
