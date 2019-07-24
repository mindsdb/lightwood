rm -rf build
rm -rf dist
rm -rf lightwood.egg-info
# python3 setup.py --help-commands

echo "mode (prod/dev)?"

read mode

if [ "$mode" = "prod" ]; then

    # @TODO: Not sure if these 2 are actually required
    python3 setup.py develop --uninstall
    python3 setup.py clean

    python3 setup.py sdist

    echo "Do you want to publish this version (yes/no)?"

    read publish

    if [ "$publish" = "yes" ]; then
        echo "Publishing lightwood to Pypi"
        python3 -m twine upload dist/*
	cd docs
	mkdocs gh-deploy
    fi


fi

if [ "$mode" = "dev" ]; then
    pip3 uninstall lightwood
    python3 setup.py develop --uninstall
    python3 setup.py develop
fi
