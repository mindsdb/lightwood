## How to make a tutorial notebook?

We use some of our tutorial notebooks as unit-tests to ensure that our pipeline is up-to-date, and to keep our examples relevant. 

In order to preserve our (and the reader's) sanity these need to behave so that they can:
1. Run via the CI tools
2. Execute locally by a user

To make things easier, the Lightwood team has proposed a general set of rules for tutorials:

1. If you are using an external dataset, please ensure there is a URL that links to it (i.e.: load it in a dataframe using `pd.read_csv('{link}')`). Exceptions can be made for custom data types if the download dataset link is provided. We try to avoid hosting large datasets via Github, but please contact us if you believe it should be in our benchmarking suite.
2. Show any **custom code within the notebook**. If you need to export it to a file (e.g. in order to load it as a lightwood module), use `%%writefile my_file.py` at the top of the jupyter codeblock, this will write the code into a file.
3. Please do not save any extra files within the notebook (`.json` files may be ok); if your tutorial really requires saving extra files, please contact us and we can help.
4. Please edit json-ai within the notebook as opposed to externally (i.e. generate a default, then make changes based on the key you need). You can show the difference between default and custom json-ai via a `print` statement.
5. The notebook must not have any code metadata, otherwise github actions will fail to run them (in the json representation, grep for `kernel` and you will find the global `metadata` key, set that to `{}`)


If your tutorial is anything more than a single `.ipynb` notebook and some accompanying .png or .jpg files, it may be rejected automatically. We would be more than happy to work with you to help adapt them to fit our automated notebooks. 
