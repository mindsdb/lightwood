import sys
import setuptools
import os


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

extra_requirements = {}
for fn in os.listdir('.'):
    if fn.startswith('requirements_') and fn.endswith('.txt'):
        extra_name = fn.replace('requirements_', '').replace('.txt', '')
        with open(fn) as fp:
            extra = [req.strip() for req in fp.read().splitlines()]
        extra_requirements[extra_name] = extra
full_requirements = []
for v in extra_requirements.values():
    full_requirements += v
extra_requirements['all_extras'] = list(set(full_requirements))

# Windows specific requirements
if sys_platform in ['win32', 'cygwin', 'windows']:
    # These have to be installed manually or via the installers in windows
    requirements = remove_requirements(requirements, 'torch')

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
    packages=setuptools.find_packages(exclude=["tests", "tests.*"]),
    package_data={'project': ['requirements.txt']},
    install_requires=requirements,
    extras_require=extra_requirements,
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.7"
)
