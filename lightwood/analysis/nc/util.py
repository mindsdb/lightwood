from typing import Union, Optional

import torch
import numpy as np
import pandas as pd
from torch.nn.functional import softmax

from type_infer.dtype import dtype


def t_softmax(x, t=1.0, axis=1):
    """ Softmax with temperature scaling """
    return softmax(torch.Tensor(x) / t, dim=axis).numpy()


def clean_df(df, namespace, label_encoders):
    """ Returns cleaned DF for nonconformist calibration """
    # @TODO: reevaluate whether this can be streamlined
    enc = label_encoders
    ns = namespace
    target = ns.target

    if ns.is_classification:
        y = df.pop(target).values
        if enc and isinstance(enc.categories_[0][0], str):
            cats = enc.categories_[0].tolist()
            # the last element is "__mdb_unknown_cat"
            y = np.array([cats.index(i) if i in cats else len(cats) - 1 for i in y])
        y = y.clip(-pow(2, 63), pow(2, 63)).astype(int)
    elif ns.is_multi_ts:
        target_cols = [ns.target] + [f'{ns.target}_timestep_{i}' for i in range(1, ns.tss.horizon)]
        y = np.transpose(np.array([df.pop(col) for col in target_cols]))
    else:
        y = df.pop(target).values
        y = y.astype(float)

    return df, y


def set_conf_range(
        X: pd.DataFrame,
        icp,
        target_type: dtype,
        analysis_info: dict,
        positive_domain: bool = False,
        std_tol: int = 1,
        group: str = '__default',
        significance: Optional[float] = None
):
    """
    Automatically sets confidence level for numerical and categorical tasks.

    :param X: Validation data.
    :param icp: Inductive conformal predictor that sets the confidence level. Either IcpClassifier or IcpRegressor.
    :param target_type: dtype of the target column.
    :param analysis_info:
    :param positive_domain: Flag that indicates whether target is expected to be a positive number.
    :param std_tol: Tolerance for automatic confidence level selection; bigger tolerance means higher confidence, in general.
    :param group: For tasks with multiple different target groups (where each may have a different std_dev), indicates what group is being considered.
    :param significance: Desired confidence level. Can be preset (0 < x <= 0.99)

    :return: set confidence plus predictions regions (for numerical tasks) or pvalues (for categorical tasks).
    """  # noqa
    # numerical
    if target_type in (dtype.integer, dtype.float, dtype.quantity):

        # ICP gets all possible bounds (shape: (B, 2, 99))
        all_ranges = icp.predict(X.values)

        # iterate over confidence levels until spread >= a multiplier of the dataset stddev
        if significance is not None:
            conf = int(100 * (1 - significance))
            return significance, all_ranges[:, :, conf]
        else:
            for tol in [std_tol, std_tol + 1, std_tol + 2]:
                for significance in range(99):
                    ranges = all_ranges[:, :, significance]
                    spread = np.mean(ranges[:, 1] - ranges[:, 0])
                    tolerance = analysis_info['df_target_stddev'][group] * tol

                    if spread <= tolerance:
                        confidence = (99 - significance) / 100
                        if positive_domain:
                            ranges[ranges < 0] = 0
                        return confidence, ranges
            else:
                ranges = all_ranges[:, :, 0]
                if positive_domain:
                    ranges[ranges < 0] = 0
                return 0.9901, ranges

    # time series
    elif target_type in (dtype.num_array, dtype.num_tsarray):
        pass

    # categorical
    elif target_type in (dtype.binary, dtype.categorical, dtype.cat_array, dtype.cat_tsarray):
        pvals = icp.predict(X.values)  # p-values at which each class is included in the predicted set
        conf = get_categorical_conf(pvals)
        return conf, pvals

    # default
    return 0.005, np.zeros((X.shape[0], 2))


def get_numeric_conf_range(
        all_confs: np.ndarray,
        df_target_stddev: dict = {},
        positive_domain: bool = False,
        std_tol: int = 1,
        group: Optional[str] = '__default',
        fixed_conf: float = None
):
    """
    Gets prediction bounds for numerical targets, based on ICP estimation and width tolerance.
    
    :param all_confs: All possible bounds depending on confidence level.
    :param df_target_stddev: Observed train standard deviation for each group target.
    :param positive_domain: Flag that indicates whether target is expected to be a positive number.
    :param std_tol: Tolerance for automatic confidence level selection; bigger tolerance means higher confidence, in general.
    :param group: For tasks with multiple different target groups (where each may have a different std_dev), indicates what group is being considered.
    :param fixed_conf: Pre-determined confidence for the ICP, 0-1 bounded. Can be specified to bypass automatic confidence/bound detection, or to adjust the threshold sensitivity in anomaly detection tasks.
    
    :return: array with confidence for each data instance, along with lower and upper bounds for each prediction.
    """  # noqa

    if fixed_conf is None:
        significances = []
        conf_ranges = []
        std_dev = df_target_stddev[group]
        tolerance = std_dev * std_tol

        for sample_idx in range(all_confs.shape[0]):
            sample = all_confs[sample_idx, :, :]
            for idx in range(sample.shape[1]):
                significance = (99 - idx) / 100
                diff = sample[1, idx] - sample[0, idx]
                if diff <= tolerance:
                    conf_range = list(sample[:, idx])
                    significances.append(significance)
                    conf_ranges.append(conf_range)
                    break
            else:
                significances.append(0.9991)  # default: confident that value falls inside big bounds
                bounds = sample[:, 0]
                sigma = (bounds[1] - bounds[0]) / 4
                conf_range = [bounds[0] - sigma, bounds[1] + sigma]
                conf_ranges.append(conf_range)

        conf_ranges = np.array(conf_ranges)
    else:
        # fixed error rate
        conf = max(0.01, min(1.0, fixed_conf))
        error_rate = 1 - conf
        conf_idx = int(100 * error_rate) - 1
        conf_ranges = all_confs[:, :, conf_idx]
        significances = [conf for _ in range(conf_ranges.shape[0])]

    if positive_domain:
        conf_ranges[conf_ranges < 0] = 0
    return np.array(significances), conf_ranges


def get_ts_conf_range(
        all_confs: np.ndarray,
        df_target_stddev: dict = {},
        positive_domain: bool = False,
        std_tol: int = 1,
        group: Optional[str] = '__default',
        fixed_conf: float = None
):
    all_significances = []
    all_conf_ranges = []
    for timestep in range(all_confs.shape[1]):
        sigs, confs = get_numeric_conf_range(
            all_confs[:, timestep, :, :],
            df_target_stddev,
            positive_domain,
            std_tol,
            group,
            fixed_conf
        )
        all_significances.append(sigs)
        all_conf_ranges.append(confs)

    return np.vstack(all_significances).T, np.stack(all_conf_ranges).swapaxes(0, 1)


def get_categorical_conf(raw_confs: np.ndarray):
    """
    ICP confidence estimation for categorical targets from raw p-values:
        1.0 minus 2nd highest p-value yields confidence for predicted label.
    :param all_confs: p-value for each class per data point
    :return: confidence for each data point
    """
    if len(raw_confs.shape) == 1:
        raw_confs = np.expand_dims(raw_confs, axis=0)
    if raw_confs.shape[-1] == 1:
        # single-class edge case (only happens if predictor sees just one known label at calibration)
        confs = np.clip(raw_confs[:, 0], 0.0001, 0.9999)
    else:
        second_p = np.sort(raw_confs, axis=1)[:, -2]
        confs = np.clip(np.subtract(1, second_p), 0.0001, 0.9999)
    return confs


def get_anomalies(insights: pd.DataFrame, observed_series: Union[pd.Series, list], cooldown: int = 1) -> np.ndarray:
    """
    Simple procedure for unsupervised anomaly detection in time series forecasting. 
    Uses ICP analysis block output so that any true value falling outside of the lower and upper bounds is tagged as anomalous.
      
    :param insights: dataframe with row insights used during the `.explain()` phase of all analysis blocks.  
    :param observed_series: true values from the predicted time series. If empty, no anomalies are flagged.
    :param cooldown: minimum amount of observations (assuming regular sampling frequency) that need to pass between two consecutive anomalies.
    
    :return: np.ndarray of boolean flags, indicating anomalous behavior for each predicted value.
    """  # noqa
    anomalies = []
    counter = 0

    # cast to float (safe, we only call this method if series is numerical)
    try:
        observed_series = [float(value) for value in observed_series]
    except (TypeError, ValueError):
        return [None for _ in observed_series]

    lower_bounds = insights['lower'].tolist()
    upper_bounds = insights['upper'].tolist()

    for (l, u), t in zip(zip(lower_bounds, upper_bounds), observed_series):
        if t is not None:
            if isinstance(l, list):
                anomaly = not (l[0] <= t <= u[0])
            else:
                anomaly = not (l <= t <= u)

            if anomaly and (counter == 0 or counter >= cooldown):
                anomalies.append(anomaly)  # new anomaly event triggers, reset counter
                counter = 1
            elif anomaly and counter < cooldown:
                anomalies.append(False)  # overwrite as not anomalous if still in cooldown
                counter += 1
            else:
                anomalies.append(anomaly)
                counter = 0
        else:
            anomalies.append(None)
            counter += 1

    return np.array(anomalies)
