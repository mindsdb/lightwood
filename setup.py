import os
import sys
import setuptools

print('Installing lightwood dynamically !')

about = {}
with open("lightwood/__about__.py") as fp:
    exec(fp.read(), about)


def remove_requirements(requirements, name, replace=None):
    new_requirements = []
    for requirement in requirements:
        if requirement.split(' ')[0] != name:
            new_requirements.append(requirement)
        elif replace is not None:
            new_requirements.append(replace)
    return new_requirements

sys_platform = sys.platform

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

dependency_links = []

# Linux specific requirements
if sys_platform == 'linux' or sys_platform.startswith('linux'):
    requirements = remove_requirements(requirements,'torch',replace='torch == 1.1.0')
    

# OSX specific requirements
elif sys_platform == 'darwin':
    requirements = remove_requirements(requirements,'torch',replace='torch == 1.1.0.post2')

# Windows specific requirements
elif sys_platform in ['win32','cygwin'] :

    # Bellow should work for python3.7 + cudnn 10... though, surprisingly, it seems to also work for no cudnn
    requirements = remove_requirements(requirements,'torch',replace='torch @ https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl')
    requirements = remove_requirements(requirements,'torchvision',replace='torchvision @ https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl')

    requirements.append('cwrap')

    # This doens't work as well as the `@` version
    #dependency_links.append('https://download.pytorch.org/whl/cu100/torch-1.1.0-cp37-cp37m-win_amd64.whl#egg=torch-1.1.0')
    #dependency_links.append('https://download.pytorch.org/whl/cu100/torchvision-0.3.0-cp37-cp37m-win_amd64.whl#egg=torchvision-0.3.0')

# For stuff like freebsd
else:
    print('\n\n====================\n\nError, platform {sys_platform} not recognized, proceeding to install anyway, but lightwood might not work properly !\n\n====================\n\n')

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
    package_data={'project': ['requirements.txt']},
    install_requires=requirements,
    dependency_links=dependency_links,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.6"
)
