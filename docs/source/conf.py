# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Path setup --------------------------------------------------------------

# If extensions (or modules to document with autodoc) are in another directory,
# add these directories to sys.path here. If the directory is relative to the
# documentation root, use os.path.abspath to make it absolute, like shown here.
#
import os
import sys
# sys.path.insert(0, os.path.abspath('../nifits/*'))
sys.path.insert(0, os.path.abspath('../../nifits/'))


# -- Project information -----------------------------------------------------

project = 'NIFITS'
copyright = '2024, R. Laugier'
author = 'R. Laugier'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
from nifits import __version__
release = __version__
extensions = [
    "autoapi.extension",
    'sphinx.ext.autodoc',
    "sphinx.ext.mathjax",
    "sphinx.ext.viewcode",
    # "sphinx.ext.imgmath",
    "myst_parser",
#    "numpydoc",
    "sphinx.ext.napoleon", # conda install conda-forge::sphinxcontrib-napoleon
    # "sphinx.ext.autosummary",
    "sphinx_autodoc_typehints", # conda install conda-forge::sphinx-autodoc-typyhints
    "sphinx_design"
#    "sphinx.ext.doctest",
#    "sphinx.ext.inheritance_diagram"
]

source_suffix = [".rst", ".md"]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

autoapi_dirs = ["../../nifits"]
# autoapi_dirs = ["../../nitest"]


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'furo'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']


# -- Extension configuration -------------------------------------------------