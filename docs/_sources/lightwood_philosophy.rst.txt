:mod:`Lightwood Philosophy`
================================

Lightwood abstracts the ML pipeline into 3 core steps:

1. Pre-processing and data cleaning
2. Feature engineering
3. Model building and training

.. image:: _static/logos/lightwood.png
    :align: center
    :alt: Lightwood "under-the-hood"

i) Pre-processing and cleaning
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
For each column in your dataset, Lightwood will identify the suspected data type (numeric, categorical, etc.) via a brief statistical analysis. From this, it will generate a JSON-AI syntax. 

If the user keeps default behavior, Lightwood will perform a brief pre-processing approach to clean each column according to its identified data type. From there, it will split the data into train/dev/test splits.

The `cleaner` and `splitter` objects respectively refer to the pre-processing and the data splitting functions.

ii) Feature Engineering
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
Data can be converted into features via "encoders". Encoders represent the rules for transforming pre-processed data into a numerical representations that a model can be used. 

Encoders can be **rule-based** or **learned**. A rule-based encoder transforms data per a specific set of instructions (ex: normalized numerical data) whereas a learned encoder produces a representation of the data after training (ex: a "\[CLS\]" token in a language model).

Encoders are assigned to each column of data based on the data type; users can override this assignment either at the column-based level or at the data-type based level. Encoders inherit from the `BaseEncoder` class. 

iii) Model Building and Training
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
We call a predictive model that intakes *encoded* feature data and outputs a prediction for the target of interest a `mixer` model. Users can either use Lightwood's default mixers or create their own approaches inherited from the `BaseMixer` class.

We predominantly use PyTorch based approaches, but can support other models.