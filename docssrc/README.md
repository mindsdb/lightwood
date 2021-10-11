## Compiling the docs
`pip3 install 'Sphinx==4.1.2' 'sphinx-autoapi==1.8.4' 'sphinx-autodoc-typehints==1.12.0' 'sphinx-code-include==1.1.1' 'sphinx-rtd-theme==0.5.2' 'sphinxcontrib-applehelp==1.0.2' 'sphinxcontrib-devhelp==1.0.2' 'sphinxcontrib-htmlhelp==2.0.0' 'sphinxcontrib-jsmath==1.0.1' 'sphinxcontrib-napoleon==0.7' 'sphinxcontrib-qthelp==1.0.3' 'sphinxcontrib-serializinghtml==1.1.5' autoapi nbsphinx myst_parser`
`cd docssrc`
`make github`
`cd ../docs && python3 -m http.server`
Should now be available at: 0.0.0.0:8000


## Creating the docs
*They are already created, you shouldn't have to do this unless you are restarting from scratch*

First, make a new directory (should exist) named `docs`.

Within `docssrc`, run sphinx-quickstart (https://www.sphinx-doc.org/en/master/usage/quickstart.html).

I opted to separate source/build directories. This allows, in the long run, simplicity between code + build.
