# Contributing to Lightwood

We love to receive contributions from the community and hear your opinions! We want to make contributing to Lightwood as easily as it can be.

# How can you help us?

* Report a bug
* Improve documentation
* Discuss the code implementation
* Submit a bug fix
* Propose new features
* Test Lightwood
* Solve an issue

# Code contributions
In general, we follow the "fork-and-pull" Git workflow.

1. Fork the Lightwood repository
2. Checkout the `staging` branch, this is the development version that gets released weekly
4. Make changes and commit them
5. Make sure that the CI tests pass
6. Submit a Pull request from your repo to the `staging` branch of mindsdb/lightwood so that we can review your changes

> You will need to sign a CLI agreement for the code since lightwood is under a GPL license
> Be sure to merge the latest from `staging` before making a pull request!
> You can run the test suite locally by running `flake8 .` to check style and `python -m unittest discover tests` to run the automated tests. This doesn't guarantee it will pass remotely since we run on multiple envs, but should work in most cases.

# Feature and Bug reports
We use GitHub issues to track bugs and features. Report them by opening a [new issue](https://github.com/mindsdb/lightwood/issues/new/choose) and fill out all of the required inputs.

# Code review process
The Pull Request reviews are done on a regular basis. 

If your change has a chance to affecting performance we will run our private benchmark suite to validate it.

Please, make sure you respond to our feedback and questions.

# Community
If you have additional questions or you want to chat with MindsDB core team, you can join our community slack.

# Setting up a dev environment

- Clone lightwood
- `cd lightwood && pip install requirements.txt`
- Add it to your python path (e.g. by adding `export PYTHONPATH='/where/you/cloned/lightwood:$PYTHONPATH` as a newline at the end of your `~/.bashrc` file)
- Check that the unittest are passing by going into the directory where you cloned lightwood and running: `python -m unittest discover tests` 

> If `python` default to python2.x on your environment use `python3` and `pip3` instead

## Setting up a vscode environment

Currently, the prefred environment for working with lightwood is vscode, it's a very popular python IDE. Any IDE should however work, while we don't have guides for those please use the following as a template.

* Install and enable setting sync using github account (if you use multiple machines)
* Install pylance (for types) and make sure to disable pyright
* Go to `Python > Lint: Enabled` and disable everything *but* flake8
* Set `python.linting.flake8Path` to the full path to flake8 (which flake8)
* Set `Python › Formatting: Provider` to autopep8
* Add `--global-config=<path_to>/lightwood/.flake8` and `--experimental` to `Python › Formatting: Autopep8 Args`
* Install live share and live share whiteboard