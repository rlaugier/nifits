import sys
from setuptools import setup

setup(name="nifits",
    version="0.0.10",
    description="A framework to handle the NIFITS Nullin Interferometry data standard",
    url="--",
    author="Romain Laugier",
    author_email="romain.laugier@kuleuven.be",
    license="BSD-3-Clause",
    classifiers=[
      'Development Status :: 2 - Pre-alpha',
      'Intended Audience :: Professional Astronomers',
      'Topic :: High Angular Resolution Astronomy :: Interferometry :: High-contrast',
      'Programming Language :: Python :: 3.8'
    ],
    packages=["nifits", "nifits.io", "nifits.backend", "nifits.extra", "nitest"],
    install_requires=[
            'numpy', 'scipy', 'matplotlib', 'astropy',
            'sympy', 'einops'
    ],
    zip_safe=False)
