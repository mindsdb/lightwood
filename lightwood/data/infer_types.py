from collections import Counter
import random
import dateutil
from scipy.stats import norm
import pandas as pd
import numpy as np
import imghdr
import sndhdr
import multiprocessing as mp
from mindsdb_datasources import DataSource
from lightwood.api import TypeInformation
from lightwood.api import dtype
from lightwood.helpers.parallelism import get_nr_procs
from lightwood.helpers.text import get_identifier_description_mp, cast_string_to_python_type, get_language_dist, analyze_sentences
from lightwood.helpers.log import log


def get_binary_type(element: object) -> str:
    try:
        is_img = imghdr.what(element)
        if is_img is not None:
            return dtype.image

        # @TODO: currently we don differentiate between audio and video
        is_audio = sndhdr.what(element)
        if is_audio is not None:
            return dtype.audio
    except Exception:
        # Not a file or file doesn't exist
        return None


def get_numeric_type(element: object) -> str:
    """ Returns the subtype inferred from a number string, or False if its not a number"""
    pytype = cast_string_to_python_type(str(element))
    if isinstance(pytype, float):
        return dtype.float
    elif isinstance(pytype, int):
        return dtype.integer
    else:
        return None


def type_check_sequence(element: object) -> str:
    dtype_guess = None

    for sep_char in [',', '\t', '|', ' ']:
        all_nr = True
        if '[' in element:
            ele_arr = element.rstrip(']').lstrip('[').split(sep_char)
        else:
            ele_arr = element.rstrip(')').lstrip('(').split(sep_char)

        for ele in ele_arr:
            if not get_numeric_type(ele):
                all_nr = False
                break

        if len(ele_arr) > 1 and all_nr:
            dtype_guess = dtype.array

    return dtype_guess


def type_check_date(element: object) -> str:
    try:
        dt = dateutil.parser.parse(element)

        # Not accurate 100% for a single datetime str,
        # but should work in aggregate
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and len(element) <= 16:
            return dtype.date
        else:
            return dtype.datetime

    except ValueError:
        return None


def count_data_types_in_column(data):
    dtype_counts = Counter()

    type_checkers = [get_numeric_type,
                     type_check_sequence,
                     get_binary_type,
                     type_check_date]

    for element in data:
        for type_checker in type_checkers:
            dtype_guess = type_checker(element)
            if dtype_guess is not None:
                dtype_counts[dtype_guess] += 1
            else:
                dtype_counts[dtype.invalid] += 1

    return dtype_counts


def get_column_data_type(arg_tup):
    """
    Provided the column data, define its data type and data subtype.

    :param data: an iterable containing a sample of the data frame
    :param full_data: an iterable containing the whole column of a data frame

    :return: type and type distribution, we can later use type_distribution to determine data quality
    NOTE: type distribution is the count that this column has for belonging cells to each DATA_TYPE
    """
    data, full_data, col_name, pct_invalid = arg_tup
    additional_info = {'other_potential_dtypes': []}

    warn = []
    info = []
    if len(data) == 0:
        warn.append(f'Column {col_name} has no data in it. ')
        warn.append(f'Please remove {col_name} from the training file or fill in some of the values !')
        return None, None, additional_info, warn, info

    dtype_counts = count_data_types_in_column(data)

    known_dtype_dist = {k: v for k, v in dtype_counts.items()}
    
    max_known_dtype, max_known_dtype_count = max(
        known_dtype_dist.items(),
        key=lambda kv: kv[1]
    )

    actual_pct_invalid = 100 * (len(data) - max_known_dtype_count) / len(data)
    if max_known_dtype is None or max_known_dtype == dtype.invalid or actual_pct_invalid > pct_invalid:
        curr_dtype = None
    else:
        curr_dtype = max_known_dtype

    nr_vals = len(full_data)
    nr_distinct_vals = len(set(full_data))

    # Check for Tags subtype
    if curr_dtype != dtype.array:
        lengths = []
        unique_tokens = set()

        can_be_tags = False
        if all(isinstance(x, str) for x in data):
            can_be_tags = True
            delimiter = ','
            for item in data:
                item_tags = [t.strip() for t in item.split(delimiter)]
                lengths.append(len(item_tags))
                unique_tokens = unique_tokens.union(set(item_tags))

        # If more than 30% of the samples contain more than 1 category and there's more than 6 of them and they are shared between the various cells
        if can_be_tags and np.mean(lengths) > 1.3 and len(unique_tokens) >= 6 and len(unique_tokens) / np.mean(lengths) < (len(data) / 4):
            curr_dtype = dtype.tags

    # Categorical based on unique values
    if curr_dtype != dtype.date and curr_dtype != dtype.datetime and curr_dtype != dtype.tags:
        if nr_distinct_vals < (nr_vals / 20) or nr_distinct_vals < 6:
            if (curr_dtype != dtype.integer and curr_dtype != dtype.float) or (nr_distinct_vals < 20):
                if curr_dtype is not None:
                    additional_info['other_potential_dtypes'].append(curr_dtype)
                curr_dtype = dtype.categorical

    # If curr_data_type is still None, then it's text or category
    if curr_dtype is None:
        lang_dist = get_language_dist(data)

        # Normalize lang probabilities
        for lang in lang_dist:
            lang_dist[lang] /= len(data)

        # If most cells are unknown language then it's categorical
        if lang_dist['Unknown'] > 0.5:
            curr_dtype = dtype.categorical
        else:
            nr_words, word_dist, nr_words_dist = analyze_sentences(data)

            if 1 in nr_words_dist and nr_words_dist[1] == nr_words:
                curr_dtype = dtype.categorical
            else:
                if len(word_dist) > 500 and nr_words / len(data) > 5:
                    curr_dtype = dtype.rich_text
                else:
                    curr_dtype = curr_dtype.short_text

                dtype_counts = {curr_dtype: len(data)}

                return curr_dtype, dtype_counts, additional_info, warn, info

    if curr_dtype in [dtype.categorical, dtype.rich_text, dtype.short_text]:
        dtype_counts = {curr_dtype: len(data)}

    return curr_dtype, dict(dtype_counts), additional_info, warn, info


def calculate_sample_size(
    population_size,
    margin_error=.05,
    confidence_level=.99,
    sigma=1 / 2
):
    """
    Calculate the minimal sample size to use to achieve a certain
    margin of error and confidence level for a sample estimate
    of the population mean.
    Inputs
    -------
    population_size: integer
        Total size of the population that the sample is to be drawn from.
    margin_error: number
        Maximum expected difference between the true population parameter,
        such as the mean, and the sample estimate.
    confidence_level: number in the interval (0, 1)
        If we were to draw a large number of equal-size samples
        from the population, the true population parameter
        should lie within this percentage
        of the intervals (sample_parameter - e, sample_parameter + e)
        where e is the margin_error.
    sigma: number
        The standard deviation of the population.  For the case
        of estimating a parameter in the interval [0, 1], sigma=1/2
        should be sufficient.
    """
    alpha = 1 - (confidence_level)
    # dictionary of confidence levels and corresponding z-scores
    # computed via norm.ppf(1 - (alpha/2)), where norm is
    # a normal distribution object in scipy.stats.
    # Here, ppf is the percentile point function.
    zdict = {
        .90: 1.645,
        .91: 1.695,
        .99: 2.576,
        .97: 2.17,
        .94: 1.881,
        .93: 1.812,
        .95: 1.96,
        .98: 2.326,
        .96: 2.054,
        .92: 1.751
    }
    if confidence_level in zdict:
        z = zdict[confidence_level]
    else:
        # Inf fix
        if alpha == 0.0:
            alpha += 0.001
        z = norm.ppf(1 - (alpha / 2))
    N = population_size
    M = margin_error
    numerator = z**2 * sigma**2 * (N / (N - 1))
    denom = M**2 + ((z**2 * sigma**2) / (N - 1))
    return numerator / denom


def sample_data(df: pd.DataFrame):
    population_size = len(df)
    if population_size <= 50:
        sample_size = population_size
    else:
        sample_size = int(round(calculate_sample_size(population_size, 0.01, 1 - 0.005)))

    population_size = len(df)
    input_data_sample_indexes = random.sample(range(population_size), sample_size)
    return df.iloc[input_data_sample_indexes]


def infer_types(data: DataSource, pct_invalid: float) -> TypeInformation:
    type_information = TypeInformation()
    data = data.df

    sample_df = sample_data(data)
    sample_size = len(sample_df)
    population_size = len(data)
    log.info(f'Analyzing a sample of {sample_size}')
    log.info(f'from a total population of {population_size}, this is equivalent to {round(sample_size*100/population_size, 1)}% of your data.')

    nr_procs = get_nr_procs(data)
    if nr_procs > 1:
        log.info(f'Using {nr_procs} processes to deduct types.')
        pool = mp.Pool(processes=nr_procs)
        # Make type `object` so that dataframe cells can be python lists
        answer_arr = pool.map(get_column_data_type, [
            (sample_df[x].dropna(), data[x], x, pct_invalid) for x in sample_df.columns.values
        ])
        pool.close()
        pool.join()
    else:
        answer_arr = []
        for x in sample_df.columns.values:
            answer_arr.append(get_column_data_type([sample_df[x].dropna(), data[x], x, pct_invalid]))

    for i, col_name in enumerate(sample_df.columns.values):
        (data_dtype, data_dtype_dist, additional_info, warn, info) = answer_arr[i]

        for msg in warn:
            log.warning(msg)
        for msg in info:
            log.info(msg)

        if data_dtype is None:
            data_dtype = dtype.invalid

        type_information.dtypes[col_name] = data_dtype
        type_information.additional_info[col_name] = {
            'dtype_dist': data_dtype_dist
        }

    if nr_procs > 1:
        pool = mp.Pool(processes=nr_procs)
        answer_arr = pool.map(get_identifier_description_mp, [
            (data[x], x, type_information.dtypes[x])
            for x in sample_df.columns.values
        ])
        pool.close()
        pool.join()
    else:
        answer_arr = []
        for x in sample_df.columns.values:
            answer = get_identifier_description_mp([data[x], x, type_information.dtypes[x]])
            answer_arr.append(answer)

    for i, col_name in enumerate(sample_df.columns.values):
        # work with the full data
        type_information.identifiers[col_name] = answer_arr[i]

        # @TODO Column removal logic was here, if the column was an identifier, move it elsewhere

    return type_information
