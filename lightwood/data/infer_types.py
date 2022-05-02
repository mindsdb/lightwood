from collections import Counter
import random
from typing import List
import dateutil
from scipy.stats import norm
import pandas as pd
import numpy as np
import imghdr
import sndhdr
import multiprocessing as mp
from lightwood.api.types import TypeInformation
from lightwood.api.dtype import dtype
from lightwood.helpers.parallelism import get_nr_procs
from lightwood.helpers.text import (get_identifier_description_mp, cast_string_to_python_type, get_language_dist,
                                    analyze_sentences)
from lightwood.helpers.log import log
import re
from lightwood.helpers.numeric import is_nan_numeric
from lightwood.helpers.seed import seed


# @TODO: hardcode for distance, time, subunits of currency (e.g. cents) and other common units
# @TODO: The json ml will contain the pattern we want to extract out of our quantity column, for the user modify (unit+multiplier) # noqa
# @TODO: Add tests with plenty of examples
def get_quantity_col_info(col_data: List[object]) -> str:
    char_const = None
    nr_map = set()
    for val in col_data:
        val = str(val)
        char_part = re.sub("[0-9.,]", '', val)
        numeric_bit = re.sub("[^0-9.,]", '', val).replace(',', '.')

        if len(char_part) == 0:
            char_part = None

        if len(re.sub("[^0-9]", '', numeric_bit)) == 0 or numeric_bit.count('.') > 1:
            numeric_bit = None
        else:
            numeric_bit = float(numeric_bit)

        if numeric_bit is None:
            return False, None
        else:
            nr_map.add(numeric_bit)

        if char_const is None:
            char_const = char_part

        if char_part is None or char_part != char_const:
            return False, None

    if len(nr_map) > 20 and len(nr_map) > len(col_data) / 200:
        return True, {char_const: {
            'multiplier': 1
        }}
    else:
        return False, None


def get_binary_type(element: object) -> str:
    try:
        is_img = imghdr.what(element)
        if is_img is not None:
            return dtype.image

        # @TODO: currently we don differentiate between audio and video
        is_audio = sndhdr.what(element)
        # apparently `sndhdr` is really bad..
        for audio_ext in ['.wav', '.mp3']:
            if element.endswith(audio_ext):
                is_audio = True
        if is_audio is not None:
            return dtype.audio
    except Exception:
        # Not a file or file doesn't exist
        return None


def get_numeric_type(element: object) -> str:
    """ Returns the subtype inferred from a number string, or False if its not a number"""
    string_as_nr = cast_string_to_python_type(str(element))

    try:
        if string_as_nr == int(string_as_nr):
            string_as_nr = int(string_as_nr)
    except Exception:
        pass

    if isinstance(string_as_nr, float):
        return dtype.float
    elif isinstance(string_as_nr, int):
        return dtype.integer
    else:
        try:
            if is_nan_numeric(element):
                return dtype.integer
            else:
                return None
        except Exception:
            return None


def type_check_sequence(element: object) -> str:
    dtype_guess = None

    if isinstance(element, List):
        all_nr = all([get_numeric_type(ele) for ele in element])
        if all_nr:
            dtype_guess = dtype.num_array
        else:
            dtype_guess = dtype.cat_array
    else:
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
                dtype_guess = dtype.num_array

    return dtype_guess


def type_check_date(element: object) -> str:
    try:
        dt = dateutil.parser.parse(str(element))

        # Not accurate 100% for a single datetime str,
        # but should work in aggregate
        if dt.hour == 0 and dt.minute == 0 and dt.second == 0 and len(str(element)) <= 16:
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
            try:
                dtype_guess = type_checker(element)
            except Exception:
                dtype_guess = None
            if dtype_guess is not None:
                dtype_counts[dtype_guess] += 1
                break
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
    log.info(f'Infering type for: {col_name}')
    additional_info = {'other_potential_dtypes': []}

    warn = []
    info = []
    if len(data) == 0:
        warn.append(f'Column {col_name} has no data in it. ')
        warn.append(f'Please remove {col_name} from the training file or fill in some of the values !')
        return None, None, additional_info, warn, info

    dtype_counts = count_data_types_in_column(data)

    known_dtype_dist = {k: v for k, v in dtype_counts.items()}
    if dtype.float in known_dtype_dist and dtype.integer in known_dtype_dist:
        known_dtype_dist[dtype.float] += known_dtype_dist[dtype.integer]
        del known_dtype_dist[dtype.integer]

    if dtype.datetime in known_dtype_dist and dtype.date in known_dtype_dist:
        known_dtype_dist[dtype.datetime] += known_dtype_dist[dtype.date]
        del known_dtype_dist[dtype.date]

    max_known_dtype, max_known_dtype_count = max(
        known_dtype_dist.items(),
        key=lambda kv: kv[1]
    )

    actual_pct_invalid = 100 * (len(data) - max_known_dtype_count) / len(data)
    if max_known_dtype is None or max_known_dtype == dtype.invalid:
        curr_dtype = None
    elif actual_pct_invalid > pct_invalid:
        if max_known_dtype in (dtype.integer, dtype.float) and actual_pct_invalid <= 5 * pct_invalid:
            curr_dtype = max_known_dtype
        else:
            curr_dtype = None
    else:
        curr_dtype = max_known_dtype

    nr_vals = len(full_data)
    nr_distinct_vals = len(set([str(x) for x in full_data]))

    # Is it a quantity?
    if curr_dtype not in (dtype.datetime, dtype.date):
        is_quantity, quantitiy_info = get_quantity_col_info(full_data)
        if is_quantity:
            additional_info['quantitiy_info'] = quantitiy_info
            curr_dtype = dtype.quantity
            known_dtype_dist = {
                dtype.quantity: nr_vals
            }

    # Check for Tags subtype
    if curr_dtype not in (dtype.quantity, dtype.num_array):
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

        # If more than 30% of the samples contain more than 1 category and there's more than 6 and less than 30 of them and they are shared between the various cells # noqa
        if (can_be_tags and np.mean(lengths) > 1.3 and
                6 <= len(unique_tokens) <= 30 and
                len(unique_tokens) / np.mean(lengths) < (len(data) / 4)):
            curr_dtype = dtype.tags

    # Categorical based on unique values
    if curr_dtype not in (dtype.date, dtype.datetime, dtype.tags, dtype.cat_array):
        if curr_dtype in (dtype.integer, dtype.float):
            is_categorical = nr_distinct_vals < 10
        else:
            is_categorical = nr_distinct_vals < min(max((nr_vals / 100), 10), 3000)

        if is_categorical:
            if curr_dtype is not None:
                additional_info['other_potential_dtypes'].append(curr_dtype)
            curr_dtype = dtype.categorical

    # If curr_data_type is still None, then it's text or category
    if curr_dtype is None:
        log.info(f'Doing text detection for column: {col_name}')
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
                    curr_dtype = dtype.short_text

                return curr_dtype, {curr_dtype: len(data)}, additional_info, warn, info

    if curr_dtype in [dtype.categorical, dtype.rich_text, dtype.short_text, dtype.cat_array]:
        known_dtype_dist = {curr_dtype: len(data)}

    if nr_distinct_vals < 3 and curr_dtype == dtype.categorical:
        curr_dtype = dtype.binary
        known_dtype_dist[dtype.binary] = known_dtype_dist[dtype.categorical]
        del known_dtype_dist[dtype.categorical]

    log.info(f'Column {col_name} has data type {curr_dtype}')
    return curr_dtype, known_dtype_dist, additional_info, warn, info


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


def infer_types(data: pd.DataFrame, pct_invalid: float, seed_nr: int = 420) -> TypeInformation:
    seed(seed_nr)
    type_information = TypeInformation()
    sample_df = sample_data(data)
    sample_size = len(sample_df)
    population_size = len(data)
    log.info(f'Analyzing a sample of {sample_size}')
    log.info(
        f'from a total population of {population_size}, this is equivalent to {round(sample_size*100/population_size, 1)}% of your data.') # noqa

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
        if answer_arr[i] is not None:
            log.warning(f'Column {col_name} is an identifier of type "{answer_arr[i]}"')
            type_information.identifiers[col_name] = answer_arr[i]

        # @TODO Column removal logic was here, if the column was an identifier, move it elsewhere

    return type_information
