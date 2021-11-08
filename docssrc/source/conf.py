# Configuration file for the Sphinx documentation builder.
#
# This file only contains a selection of the most common options. For a full
# list see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# 2021.09.07
# Strongly inspired from
# https://github.com/MDAnalysis/mdanalysis/blob/master/package/doc/sphinx/source/conf.py

import sys
import os
import datetime
import sphinx_rtd_theme

# ----------------- #
# Project information
# ----------------- #
sys.path.append(os.path.abspath('../../lightwood'))

# ----------------- #
# Project information
# ----------------- #
project = 'lightwood'
copyright = '2021, MindsDB'
authors = "MindsDB"
# author = 'Natasha Seelam (natasha@mindsdb.com)'
now = datetime.datetime.now()
copyright = u'2017-{}, '.format(now.year) + authors

# Version of the package
packageversion = __import__('lightwood').__version__

version = packageversion
release = packageversion

# ----------------- #
# Master document
# ----------------- #
master_doc = "index"

# ----------------- #
# General Config
# ----------------- #

# Enable sphinx extensions
extensions = [
    'sphinx.ext.autodoc',
    'autoapi.extension',
    'sphinx.ext.autosectionlabel',
    'sphinx_autodoc_typehints',
    'myst_parser',
    'sphinx_rtd_theme',
    'sphinx.ext.viewcode',
    'sphinx.ext.napoleon',
    'nbsphinx'
]

# Enable markdown usage
source_suffix = {
    '.rst': 'restructuredtext',
    '.txt': 'markdown',
    '.md': 'markdown',
}

source_parsers = {'.md': 'recommonmark.parser.CommonMarkParser'}

# Templates
templates_path = ['_templates']

# List of patterns, relative to source directory, that match files and
# directories to ignore when looking for source files.
# This pattern also affects html_static_path and html_extra_path.
exclude_patterns = []

# ----------------- #
# Formatting and theme
# ----------------- #

# Default colors
colors = {
    'shamrock': '#00b06d',
    'dark_blue': '#2c263f',
    'aqua': '#4dd9ca',
    'wheat': '#fedc8c',
    'watermelon': '#f25c63',
    'blueberry': '#6751ad',
    'white': '#ffffff',
    'slate': '#5d6970',
}

# HTML details
html_theme = 'sphinx_rtd_theme'


html_theme_options = {
    'canonical_url': '',
    'logo_only': True,
    'display_version': True,
    'prev_next_buttons_location': 'bottom',
    'style_external_links': False,
    'style_nav_header_background': 'white',
    # Toc options
    'collapse_navigation': True,
    'sticky_navigation': True,
    'navigation_depth': 4,
    'includehidden': True,
    'titles_only': False,
}

# Pygments syntax highlight themes
pygments_style = 'sphinx'

# to include decorated objects like __init__
autoclass_content = 'both'

# Add any paths that contain custom static files (such as style sheets) here,
# relative to this directory. They are copied after the builtin static files,
# so a file named "default.css" will overwrite the builtin "default.css".
html_static_path = ['_static']
html_css_files = ['custom.css']


# Brand logo
html_logo = "_static/logos/mindsdblogo.png"

html_sidebars = {
    '**': [
        'about.html',
        'navigation.html',
        'relations.html',
        'searchbox.html',
    ]
}

# ----------------- #
# Autodoc capability
# ----------------- #
autoapi_template_dir = '_autoapi_templates'
autoapi_root = 'docs'
autoapi_generate_api_docs = False

autoapi_dirs = ['../../lightwood']

# autodoc_member_order = 'bysource' # Keep order of the members accordingly
