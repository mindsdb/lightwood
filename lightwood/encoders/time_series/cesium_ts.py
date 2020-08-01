import warnings
import math

import numpy as np
import torch
from cesium import featurize
from lightwood.encoders.encoder_base import BaseEncoder

DEFAULT_FEATURES_TO_USE = [
    "all_times_nhist_numpeaks",
    "all_times_nhist_peak1_bin",
    "all_times_nhist_peak2_bin",
    "all_times_nhist_peak3_bin",
    "all_times_nhist_peak4_bin",
    "all_times_nhist_peak_1_to_2",
    "all_times_nhist_peak_1_to_3",
    "all_times_nhist_peak_1_to_4",
    "all_times_nhist_peak_2_to_3",
    "all_times_nhist_peak_2_to_4",
    "all_times_nhist_peak_3_to_4",
    "all_times_nhist_peak_val",
    "avg_double_to_single_step",
    "avg_err",
    "avgt",
    "cad_probs_1",
    "cad_probs_10",
    "cad_probs_20",
    "cad_probs_30",
    "cad_probs_40",
    "cad_probs_50",
    "cad_probs_100",
    "cad_probs_500",
    "cad_probs_1000",
    "cad_probs_5000",
    "cad_probs_10000",
    "cad_probs_50000",
    "cad_probs_100000",
    "cad_probs_500000",
    "cad_probs_1000000",
    "cad_probs_5000000",
    "cad_probs_10000000",
    "cads_avg",
    "cads_med",
    "cads_std",
    "mean",
    "med_double_to_single_step",
    "med_err",
    "n_epochs",
    "std_double_to_single_step",
    "std_err",
    "total_time",
    "amplitude",
    "flux_percentile_ratio_mid20",
    "flux_percentile_ratio_mid35",
    "flux_percentile_ratio_mid50",
    "flux_percentile_ratio_mid65",
    "flux_percentile_ratio_mid80",
    "max_slope",
    "maximum",
    "median",
    "median_absolute_deviation",
    "minimum",
    "percent_amplitude",
    "percent_beyond_1_std",
    "percent_close_to_median",
    "percent_difference_flux_percentile",
    "period_fast",
    "qso_log_chi2_qsonu",
    "qso_log_chi2nuNULL_chi2nu",
    "skew",
    "std",
    "stetson_j",
    "stetson_k",
    "weighted_average",
    "fold2P_slope_10percentile",
    "fold2P_slope_90percentile",
    "freq1_amplitude1",
    "freq1_amplitude2",
    "freq1_amplitude3",
    "freq1_amplitude4",
    "freq1_freq",
    "freq1_lambda",
    "freq1_rel_phase2",
    "freq1_rel_phase3",
    "freq1_rel_phase4",
    "freq1_signif",
    "freq2_amplitude1",
    "freq2_amplitude2",
    "freq2_amplitude3",
    "freq2_amplitude4",
    "freq2_freq",
    "freq2_rel_phase2",
    "freq2_rel_phase3",
    "freq2_rel_phase4",
    "freq3_amplitude1",
    "freq3_amplitude2",
    "freq3_amplitude3",
    "freq3_amplitude4",
    "freq3_freq",
    "freq3_rel_phase2",
    "freq3_rel_phase3",
    "freq3_rel_phase4",
    "freq_amplitude_ratio_21",
    "freq_amplitude_ratio_31",
    "freq_frequency_ratio_21",
    "freq_frequency_ratio_31",
    "freq_model_max_delta_mags",
    "freq_model_min_delta_mags",
    "freq_model_phi1_phi2",
    "freq_n_alias",
    "freq_signif_ratio_21",
    "freq_signif_ratio_31",
    "freq_varrat",
    "freq_y_offset",
    "linear_trend",
    "medperc90_2p_p",
    "p2p_scatter_2praw",
    "p2p_scatter_over_mad",
    "p2p_scatter_pfold_over_mad",
    "p2p_ssqr_diff_over_var",
    "scatter_res_raw"
]

FEATURES_WITH_DEFAULT_NONE = [
    "cad_probs_1",
    "cad_probs_10",
    "cad_probs_20",
    "cad_probs_30",
    "cad_probs_40",
    "cad_probs_50",
    "cad_probs_100",
    "cad_probs_500",
    "cad_probs_1000",
    "cads_std",
    "std_err",
    "freq_n_alias"
]


class CesiumTsEncoder(BaseEncoder):

    def __init__(self, features=DEFAULT_FEATURES_TO_USE, is_target=False):
        super().__init__(is_target)
        self._features = features

    def prepare_encoder(self, priming_data):
        pass

    def encode(self, values_data, times=None):
        """
        Encode a column data into time series

        :param values_data: a list of timeseries data eg: ['91.0 92.0 93.0 94.0', '92.0 93.0 94.0 95.0' ...]
        :param times: (optional) a list of lists such that, len(times[i])=len(values_data[i]) for
                      all i in range(len(times))
        :return: a torch.floatTensor
        """
        features_to_use = self._features
        ret = []
        for i, values in enumerate(values_data):
            if type(values) == type([]):
                values = list(map(float,values))
            else:
                values = list(map(lambda x: float(x), values.split()))
            if times is None:
                times_row = np.array([float(i) for i in range(1, len(values) + 1)])
            else:
                times_row = np.array(list(map(lambda x: float(x), times[i].split())))  # np.array(times[i])
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                row = featurize.featurize_time_series(times=times_row,
                                                      values=np.array(values),
                                                      errors=None,
                                                      features_to_use=features_to_use)

            vector_row = []
            for col in features_to_use:
                val = list(row[col][0])[0]
                val1 = 0
                if (val in ['nan', None, 'NaN', False]) \
                        or math.isnan(val) or math.isinf(val):
                    val = 0
                    val1 = 1

                if col in FEATURES_WITH_DEFAULT_NONE:
                    vector_row.extend([val, val1])  # val1 is 1 if its null
                else:
                    vector_row.append(val)
            ret.append(vector_row)
        ret_tensor = self._pytorch_wrapper(ret)
        return ret_tensor

    def decode(self, encoded_values_tensor):
        raise Exception('This encoder is not bi-directional')

# only run the test if this file is called from debugger
if __name__ == "__main__":
    import math

    data = [" ".join(str(math.sin(i / 100)) for i in range(1, 10)) for j in range(20)]

    ret = CesiumTsEncoder(features=[
        "amplitude",
        "percent_beyond_1_std",
        "maximum",
        "max_slope",
        "median",
        "median_absolute_deviation",
        "percent_close_to_median",
        "minimum",
        "skew",
        "std",
        "weighted_average"
    ]).encode(data)

    print(ret)
