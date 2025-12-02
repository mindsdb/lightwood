from typing import Iterable, Any

import numpy as np
from type_infer.helpers import is_nan_numeric


def convert_numpy_to_python(obj: Any) -> Any:
    """
    Recursively converts numpy types to Python native types.
    This is necessary for compatibility with numpy 2.0, where the string representation
    of numpy scalars changed (e.g., np.float64(1.5) instead of 1.5).

    :param obj: Any object that may contain numpy types
    :return: The same object structure with numpy types converted to Python native types
    """
    if isinstance(obj, dict):
        return {convert_numpy_to_python(k): convert_numpy_to_python(v) for k, v in obj.items()}
    elif isinstance(obj, (list, tuple)):
        converted = [convert_numpy_to_python(item) for item in obj]
        return type(obj)(converted)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, (np.bool_, bool)):
        return bool(obj)
    else:
        return obj


def is_none(value):
    """
    Pandas has no way to guarantee "stability" for the type of a column, it choses to arbitrarily change it based on the values.
    Pandas also change the values in the columns based on the types.
    Lightwood relies on having ``None`` values for a cells that represent "missing" or "corrupt".
    
    When we assign ``None`` to a cell in a dataframe this might get turned to `nan` or other values, this function checks if a cell is ``None`` or any other values a pd.DataFrame might convert ``None`` to.

    It also checks some extra values (like ``''``) that pandas never converts ``None`` to (hopefully). But lightwood would still consider those values "None values", and this will allow for more generic use later.
    """ # noqa
    # TODO: consider removing this helper entirely, arguably should move into dataprep_ml

    # start dispatch with types that are expensive to e.g. cast into strings
    if type(value) in (np.ndarray,) and value.size == 0:
        return True

    if type(value) != str and isinstance(value, Iterable) and len(value) == 0:
        return True
    elif type(value) != str and isinstance(value, Iterable):
        return False

    if value is None:
        return True

    if is_nan_numeric(value):
        return True

    if str(value) == '':
        return True

    if str(value) in ('None', 'nan', 'NaN', 'np.nan', 'NAN', 'NONE'):
        return True

    return False

