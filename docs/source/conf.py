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
# sys.path.insert(0, os.path.abspath('../../hotpp'))

import shutil
if os.path.exists("examples"):
    shutil.rmtree("examples")
shutil.copytree('../../examples', 'examples')
# -- Project information -----------------------------------------------------

project = 'HotPP'
copyright = '2022, Gegejun'
author = 'Gegejun'

# The full version, including alpha/beta/rc tags
release = '0.1'


# -- General configuration ---------------------------------------------------

# Add any Sphinx extension module names here, as strings. They can be
# extensions coming with Sphinx (named 'sphinx.ext.*') or your custom
# ones.
extensions = [
    'sphinx.ext.autodoc',
    'sphinx.ext.napoleon',
    'sphinx.ext.doctest',
    'sphinx.ext.intersphinx',  # 链接其他文档
    'sphinx.ext.todo',
    'sphinx.ext.coverage',
    'sphinx.ext.mathjax',
    'recommonmark',
    'sphinx_markdown_tables',
    'nbsphinx',
    # 'nbsphinx-link',
    # 'sphinx_gallery.gen_gallery',
    # 'sphinx-mathjax-offline',
]

# Add any paths that contain templates here, relative to this directory.
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------

# The theme to use for HTML and HTML Help pages.  See the documentation for
# a list of builtin themes.
#
html_theme = 'sphinx_rtd_theme'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']

source_suffix = {
    '.rst': 'restructuredtext',
    '.md': 'markdown',
}

nbsphinx_allow_errors = True
#sphinx_gallery_conf = {
#     'examples_dirs': '../../examples',   # path to your example scripts
#     'gallery_dirs': 'auto_examples',  # path to where to save gallery generated output
#     #'filename_pattern': '/plot_',
#     #'ignore_pattern': r'__init__\.py',
#}

# Chinese mathjax cdn
# mathjax_path = "//cdn.bootcdn.net/ajax/libs/mathjax/3.2.0/es5/a11y/assistive-mml.js"
# local path
mathjax_path = "mathjax/tex-chtml.js"
