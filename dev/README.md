# Contributor guide

## Setting up vscode environment

* Install and enable setting sync using github account (if you use multiple machines)
* Install pylance (for types) and make sure to disable pyright
* Go to `Python > Lint: Enabled` and disable everything *but* flake8
* Set `python.linting.flake8Path` to the full path to flake8 (which flake8)
* Set `Python › Formatting: Provider` to autopep8
* Add `--global-config=<path_to>/lightwood/.flake8` and `--experimental` to `Python › Formatting: Autopep8 Args`
* Install live share and live share whiteboard