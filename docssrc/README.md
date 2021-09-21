## Compiling the docs
`pip3 install sphinx sphinx_rtd_theme autoapi[extension] sphinx-autoapi sphinx_autodoc_typehints myst_parser sphinx_rtd_theme`
`cd docs`
`sphinx-build -b html source _build`
`cd _build`
`python3 -m http.server`


## Creating the docs
First, make a new directory (should exist) named `docs`.

Within `docs`, run sphinx-quickstart (https://www.sphinx-doc.org/en/master/usage/quickstart.html).

I opted to separate source/build directories. This allows, in the long run, simplicity between code + build.
