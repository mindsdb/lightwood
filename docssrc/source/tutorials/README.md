## How to make a tutorial notebook?

We run tutorial notebooks as tests, this is great, it gives us more test-cases and makes sure they can never be outdated. 

In order to preserve our (and the reader's) sanity these need to behave in such a way that they can:
1. Be ran by the CI tools
2. Be executed locally by a user

Thus, the following standards are put in place for new and existing tutorials:

1. Source data from a link and load it in a dataframe using `pd.read_csv('{link}')`. This is waived if your tutorial works with weird data
2. Custom code *must* be written as part of the notebook, not as a separate file. If you need to export it to a file (e.g. in order to load it as a lightwood module), use `%%writefile my_file.py` at the top of the jupyter codeblock, this will write the code into a file
3. Notebooks *mustn't* save any files outside of custom lightwood modules (which I take as necessary evil). These can too easily get into git and become confusing. This is waived if for some reason your tutorial really requires saving files, otherwise just comment out those lines.
4. Notebooks editing json-ai *must* do so inside the code (i.e. generate it, then accessed the required key and edit it), rather than loading the edited json-ai from a file. If you wish to display both versions and/or the diff between them just `print` it.

If your totorial is anything more than a single `.ipynb` notebook and some accompanying .png or .jpg files expect it to be rejected.