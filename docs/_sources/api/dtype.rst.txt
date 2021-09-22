Data Types (dtypes)
--------------------
Lightwood supports several data types used in standard machine learning pipelines. The ``dtype`` class is used to label columns of information as the right input format. The type inference procedure affects what feature engineering methodology is used on a labeled column.

Currently, the supported way to encourage new data types is to include a custom tag in this file and to import a custom cleaning approach. Users may inherit the basic functionality of the cleaner and include their own flag specific to their data type. For steps on how to do this, please see the tutorials.

.. autoclass:: api.dtype.dtype
   :members: