# ------------------------------------------------------------------------------------------------------
#  Copyright (c) Leo Hanisch. All rights reserved.
#  Licensed under the BSD 3-Clause License. See LICENSE.txt in the project root for license information.
# ------------------------------------------------------------------------------------------------------

from os import path

from setuptools import find_packages, setup

# pylint: disable=exec-used,undefined-variable

with open(path.join(path.abspath(path.dirname(__file__)), './README.md'), 'r', encoding='utf8') as rf:
    LONG_DESCRIPTION = rf.read()

# with open(path.join(path.abspath(path.dirname(__file__)), 'vague-requirements-scripts/_version.py'), 'r', encoding='utf8') as f:
#     exec(f.read())
setup(
    name='vaguerequirementslib',  # PEP8: Packages should also have short, all-lowercase names, the use of underscores is discouraged
    version='0.0.1',
    packages=find_packages('scripts'),
    package_dir={"": "scripts"},
    # Include files specified in MANIFEST.in
    # include_package_data=True,
    description='Some helper for the vague requirements thesis.',
    long_description=LONG_DESCRIPTION,
    long_description_content_type='text/markdown',
    url='https://github.com/HaaLeo/vague-requirements-scripts',
    author='Leo Hanisch',
    license='BSD 3-Clause License',
    install_requires=[
    ],
    project_urls={
        'Issue Tracker': 'https://github.com/HaaLeo/vague-requirements-scripts/issues',
        # 'Changelog': 'https://github.com/HaaLeo/vague-requirements-scripts/blob/master/CHANGELOG.md#changelog'
    },
    python_requires='>=3.6',
    keywords=[
        'vague',
        'requirements'
    ],
    classifiers=[
        'Development Status :: 5 - Production/Stable',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'License :: OSI Approved :: BSD License',
        'Operating System :: OS Independent',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'Topic :: Education',
        'Topic :: Scientific/Engineering :: Artificial Intelligence'
    ]
)
