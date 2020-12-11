import os
import sys
import setuptools
import subprocess


def remove_requirements(requirements, name, replace=''):
    new_requirements = []
    for requirement in requirements:
        if requirement.split(' ')[0] != name:
            new_requirements.append(requirement)
        elif replace is not None:
            new_requirements.append(replace)
    return new_requirements

sys_platform = sys.platform

about = {}
with open("lightwood/__about__.py") as fp:
    exec(fp.read(), about)

with open("README.md", "r") as fh:
    long_description = fh.read()

with open('requirements.txt') as req_file:
    requirements = [req.strip() for req in req_file.read().splitlines()]

with open('optional_requirements.txt') as req_file:
    optional_requirements = [req.strip() for req in req_file.read().splitlines()]

# Windows specific requirements
if sys_platform in ['win32','cygwin','windows']:
    requirements = remove_requirements(requirements,'torch')
    requirements = remove_requirements(requirements,'torchvision')
    requirements = remove_requirements(requirements,'torchvision')

    requirements = remove_requirements(requirements,'dask')
    requirements = remove_requirements(requirements,'pydub')

    print('Trying to install pytorch and torchvision!')
    code = 1
    try:
        code = subprocess.call(['pip', 'install', 'torch===1.7.0+cpu', 'torchvision===0.8.1+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
        if code != 0:
            raise Exception('Torch and torchvsion instalation failed !')
    except:
        try:
            code = subprocess.call(['pip3', 'install', 'torch===1.7.0+cpu', 'torchvision===0.8.1+cpu', '-f', 'https://download.pytorch.org/whl/torch_stable.html'])
            if code != 0:
                raise Exception('Torch and torchvision installation failed !')
        except:
            print('Failed to install pytorch, please install pytorch and torchvision manually by following the simple instructions over at: https://pytorch.org/get-started/locally/')
    if code == 0:
        print('Successfully installed pytorch and torchvision CPU version! (If you need the GPU version, please install it manually, checkout the mindsdb docs and the pytorch docs if you need help)')


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
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    extras_require = {
        'full': optional_requirements
    },
    python_requires=">=3.6"
)
