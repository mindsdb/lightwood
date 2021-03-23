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
    # These have to be installed manually or via the installers in windows
    requirements = remove_requirements(requirements,'torch')
    requirements = remove_requirements(requirements,'torchvision')


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
