
import pandas as pd

from lightwood.api.types import TimeseriesSettings
from type_infer.dtype import dtype
from lightwood.encoder.helpers import MinMaxNormalizer, CatNormalizer


def generate_target_group_normalizers(
        data: pd.DataFrame,
        target: str,
        dtype_dict: dict,
        tss: TimeseriesSettings
):
    """
    Helper function called from data_source. It generates and fits all needed normalizers for a target variable based on its grouped entities.
    
    :return: modified data with dictionary with normalizers for said target variable based on some grouped-by columns
    """  # noqa
    normalizers = {}
    target_dtype = dtype_dict[target]

    # categorical normalizers
    if target_dtype in (dtype.categorical, dtype.binary, dtype.cat_tsarray):
        normalizers['__default'] = CatNormalizer()
        normalizers['__default'].prepare(data)

    # numerical normalizers, here we spawn one per each group combination
    else:
        grouped = data.groupby(by=tss.group_by) if tss.group_by else data.groupby(lambda x: '__default')
        for name, subset in grouped:
            if subset.shape[0] > 0:
                target_data = subset[target].values
                if target_dtype == dtype.num_tsarray:
                    target_data = target_data.reshape(-1, 1).astype(float)

                normalizers[name] = MinMaxNormalizer(combination=name)
                normalizers[name].prepare(target_data)

        if not normalizers.get('__default'):
            # ...plus a default one, used at inference time and fitted with all training data
            normalizers['__default'] = MinMaxNormalizer()
            normalizers['__default'].prepare(data[target].values)

    return normalizers
