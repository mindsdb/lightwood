## Compiling the docs
- Make sure you are in `docssrc`, then follow the instructions under `run` in our [documentation building github actions job](https://github.com/mindsdb/lightwood/blob/main/.github/workflows/doc_build.yml#L21)
- Then go into the newly build docs and start a server to see them: `cd ../docs && python3 -m http.server`
- Should now be available at: 0.0.0.0:8000 | Alternatively, you can just open the `index.html` with a browser and that should work too

## Ref

for how autosummary works: https://stackoverflow.com/questions/2701998/sphinx-autodoc-is-not-automatic-enough

## Manual steps

currently notebooks have to be built manually using: `find . -iname '*.ipynb' -exec jupyter nbconvert --to notebook --inplace --execute {} \;`