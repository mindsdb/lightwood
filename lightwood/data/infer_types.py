import random
import dateutil
import string
import numpy as np
import imghdr
import sndhdr
from copy import deepcopy
from collections import Counter, defaultdict
import multiprocessing as mp
from functools import partial
import datetime
from mindsdb_datasources import DataSource
from lightwood.api import TypeInformation
from lightwood.api import dtype


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
            additional_info['separator'] = sep_char
            dtype_guess = dtype.array

    return dtype_guess

def type_check_date(element: object) -> str:
    dtype_guess = None
    try:
        dt = dateutil.parser.parse(element, **lmd.get('dateutil_parser_kwargs_per_column', {}).get(col_name, {}))

        # Not accurate 100% for a single datetime str,
        # but should work in aggregate
        if dt.hour == 0 and dt.minute == 0 and \
            dt.second == 0 and len(element) <= 16:
            dtype_guess = dtype.date
        else:
            dtype_guess = dtype.datetime


def count_data_types_in_column(data, lmd, col_name):
    dtype_counts = Counter()
    additional_info = {}

        except ValueError:
            pass
        return dtype_guess

    type_checkers = [get_numeric_type,
                     type_check_sequence,
                     get_binary_type,
                     type_check_date]
    for element in data:
        for type_checker in type_checkers:
            type_guess, subtype_guess = type_checker(element)
            if type_guess is not None:
                break
        else:
            type_guess, subtype_guess = 'Unknown', 'Unknown'

        type_counts[type_guess] += 1
        subtype_counts[subtype_guess] += 1

    return type_counts, subtype_counts, additional_info


def get_column_data_type(arg_tup, lmd):
    """
    Provided the column data, define its data type and data subtype.

    :param data: an iterable containing a sample of the data frame
    :param full_data: an iterable containing the whole column of a data frame

    :return: type and type distribution, we can later use type_distribution to determine data quality
    NOTE: type distribution is the count that this column has for belonging cells to each DATA_TYPE
    """
    data, full_data, col_name = arg_tup
    additional_info = {'other_potential_subtypes': [], 'other_potential_types': []}

    warn = []
    info = []
    if len(data) == 0:
        warn.append(f'Column {col_name} has no data in it. ')
        warn.append(f'Please remove {col_name} from the training file or fill in some of the values !')
        return None, None, None, None, additional_info, warn, info

    type_dist, subtype_dist = {}, {}

    # User-provided dtype
    if col_name in lmd['data_subtypes']:
        curr_data_type = lmd['data_types'][col_name]
        curr_data_subtype = lmd['data_subtypes'][col_name]
        type_dist[curr_data_type] = len(data)
        subtype_dist[curr_data_subtype] = len(data)
        info.append(f'Manually setting the types for column {col_name} to {curr_data_type}->{curr_data_subtype}')
        return curr_data_type, curr_data_subtype, type_dist, subtype_dist, additional_info, warn, info

    # Forced categorical dtype
    if col_name in lmd['force_categorical_encoding']:
        curr_data_type = DATA_TYPES.CATEGORICAL
        curr_data_subtype = DATA_SUBTYPES.MULTIPLE
        type_dist[DATA_TYPES.CATEGORICAL] = len(data)
        subtype_dist[DATA_SUBTYPES.MULTIPLE] = len(data)
        return curr_data_type, curr_data_subtype, type_dist, subtype_dist, additional_info, warn, info

    type_dist, subtype_dist, new_additional_info = count_data_types_in_column(data, lmd, col_name)

    if new_additional_info:
        additional_info.update(new_additional_info)

    # @TODO consider removing or flagging rows where data type is unknown in the future, might just be corrupt data...
    known_type_dist = {k: v for k, v in type_dist.items() if k != 'Unknown'}

    if known_type_dist:
        max_known_dtype, max_known_dtype_count = max(
            known_type_dist.items(),
            key=lambda kv: kv[1]
        )
    else:
        max_known_dtype, max_known_dtype_count = None, None

    nr_vals = len(full_data)
    nr_distinct_vals = len(set(full_data))

    # Data is mostly not unknown, go with type counting results
    if max_known_dtype and max_known_dtype_count > type_dist['Unknown']:
        curr_data_type = max_known_dtype

        possible_subtype_counts = [(k, v) for k, v in subtype_dist.items()
                                if k in DATA_TYPES_SUBTYPES[curr_data_type]]
        curr_data_subtype, _ = max(
            possible_subtype_counts,
            key=lambda pair: pair[1]
        )
    else:
        curr_data_type, curr_data_subtype = None, None

    # Check for Tags subtype
    if curr_data_subtype != DATA_SUBTYPES.ARRAY:
        lengths = []
        unique_tokens = set()

        can_be_tags = False
        if all(isinstance(x, str) for x in data):
            can_be_tags = True
            delimiter = lmd.get('tags_delimiter', ',')
            for item in data:
                item_tags = [t.strip() for t in item.split(delimiter)]
                lengths.append(len(item_tags))
                unique_tokens = unique_tokens.union(set(item_tags))

        # If more than 30% of the samples contain more than 1 category and there's more than 6 of them and they are shared between the various cells
        if can_be_tags and np.mean(lengths) > 1.3 and len(unique_tokens) >= 6 and len(unique_tokens)/np.mean(lengths) < (len(data)/4):
            curr_data_type = DATA_TYPES.CATEGORICAL
            curr_data_subtype = DATA_SUBTYPES.TAGS

    # Categorical based on unique values
    if curr_data_type != DATA_TYPES.DATE and curr_data_subtype != DATA_SUBTYPES.TAGS:
        if nr_distinct_vals < (nr_vals / 20) or nr_distinct_vals < 6:
            if (curr_data_type != DATA_TYPES.NUMERIC) or (nr_distinct_vals < 20):
                if curr_data_type is not None:
                    additional_info['other_potential_types'].append(curr_data_type)
                    additional_info['other_potential_subtypes'].append(curr_data_subtype)
                curr_data_type = DATA_TYPES.CATEGORICAL

    # If curr_data_type is still None, then it's text or category
    if curr_data_type is None:
        lang_dist = get_language_dist(data)

        # Normalize lang probabilities
        for lang in lang_dist:
            lang_dist[lang] /= len(data)

        # If most cells are unknown language then it's categorical
        if lang_dist['Unknown'] > 0.5:
            curr_data_type = DATA_TYPES.CATEGORICAL
        else:
            nr_words, word_dist, nr_words_dist = analyze_sentences(data)

            if 1 in nr_words_dist and nr_words_dist[1] == nr_words:
                curr_data_type = DATA_TYPES.CATEGORICAL
            else:
                curr_data_type = DATA_TYPES.TEXT

                if len(word_dist) > 500 and nr_words / len(data) > 5:
                    curr_data_subtype = DATA_SUBTYPES.RICH
                else:
                    curr_data_subtype = DATA_SUBTYPES.SHORT

                type_dist = {curr_data_type: len(data)}
                subtype_dist = {curr_data_subtype: len(data)}

                return curr_data_type, curr_data_subtype, type_dist, subtype_dist, additional_info, warn, info

    if curr_data_type == DATA_TYPES.CATEGORICAL and curr_data_subtype != DATA_SUBTYPES.TAGS:
        if nr_distinct_vals > 2:
            curr_data_subtype = DATA_SUBTYPES.MULTIPLE
        else:
            curr_data_subtype = DATA_SUBTYPES.SINGLE

    if curr_data_type in [DATA_TYPES.CATEGORICAL, DATA_TYPES.TEXT]:
        type_dist = {curr_data_type: len(data)}
        subtype_dist = {curr_data_subtype: len(data)}

    return curr_data_type, curr_data_subtype, type_dist, subtype_dist, additional_info, warn, info

def infer_types(data: DataSource) -> TypeInformation:
    stats_v2 = TypeInformation()

    sample_settings = self.transaction.lmd['sample_settings']
    if sample_settings['sample_for_analysis']:
        sample_margin_of_error = sample_settings['sample_margin_of_error']
        sample_confidence_level = sample_settings['sample_confidence_level']
        sample_percentage = sample_settings['sample_percentage']
        sample_function = self.transaction.hmd['sample_function']

        sample_df = data.sample_df(sample_function,
                                         sample_margin_of_error,
                                         sample_confidence_level,
                                         sample_percentage)

        sample_size = len(sample_df)
        population_size = len(data.data_frame)
        self.transaction.log.info(f'Analyzing a sample of {sample_size} '
                                  f'from a total population of {population_size},'
                                  f' this is equivalent to {round(sample_size*100/population_size, 1)}% of your data.')
    else:
        sample_df = data.data_frame

    nr_procs = get_nr_procs(self.transaction.lmd.get('max_processes', None),
                            self.transaction.lmd.get('max_per_proc_usage', None),
                            sample_df)
    if nr_procs > 1 and False:
        self.transaction.log.info(f'Using {nr_procs} processes to deduct types.')
        pool = mp.Pool(processes=nr_procs)
        # Make type `object` so that dataframe cells can be python lists
        answer_arr = pool.map(partial(get_column_data_type, lmd=self.transaction.lmd), [
            (sample_df[x].dropna(), data.data_frame[x], x) for x in sample_df.columns.values
        ])
        pool.close()
        pool.join()
    else:
        answer_arr = []
        for x in sample_df.columns.values:
            answer_arr.append(get_column_data_type([sample_df[x].dropna(), data.data_frame[x], x], lmd=self.transaction.lmd))

    for i, col_name in enumerate(sample_df.columns.values):
        (data_type, data_subtype, data_type_dist, data_subtype_dist, additional_info, warn, info) = answer_arr[i]

        for msg in warn:
            self.log.warning(msg)
        for msg in info:
            self.log.info(msg)

        typing = {
            'data_type': data_type,
            'data_subtype': data_subtype,
            'data_type_dist': data_type_dist,
            'data_subtype_dist': data_subtype_dist,
            'description': """A data type, in programming, is a classification that specifies which type of value a variable has and what type of mathematical, relational or logical operations can be applied to it without causing an error. A string, for example, is a data type that is used to classify text and an integer is a data type used to classify whole numbers."""
        }

        stats_v2[col_name]['typing'] = typing

        stats_v2[col_name]['additional_info'] = additional_info

    if nr_procs > 1:
        pool = mp.Pool(processes=nr_procs)
        answer_arr = pool.map(get_identifier_description_mp, [
            (data.data_frame[x],
                x,
                stats_v2[x]['typing']['data_type'],
                stats_v2[x]['typing']['data_subtype'],
                stats_v2[x]['additional_info']) for x in sample_df.columns.values
        ])
        pool.close()
        pool.join()
    else:
        answer_arr = []
        for x in sample_df.columns.values:
            answer = get_identifier_description_mp([data.data_frame[x], x, stats_v2[x]['typing']['data_type'], stats_v2[x]['typing']['data_subtype'], stats_v2[x]['additional_info']])
            answer_arr.append(answer)

    for i, col_name in enumerate(sample_df.columns.values):
        # work with the full data
        stats_v2[col_name]['identifier'] = answer_arr[i]

        if stats_v2[col_name]['identifier'] is not None:
            if col_name not in self.transaction.lmd['force_column_usage']:
                if col_name not in self.transaction.lmd['predict_columns']:
                    if (self.transaction.lmd.get('tss', None) and
                            self.transaction.lmd['tss']['is_timeseries'] and
                            (col_name in self.transaction.lmd['tss']['order_by'] or
                            col_name in (self.transaction.lmd['tss']['group_by'] or []))):
                        pass
                    else:
                        self.transaction.lmd['columns_to_ignore'].append(col_name)
                        self.transaction.data.data_frame.drop(columns=[col_name], inplace=True)

        stats_v2[col_name]['broken'] = None
        if data_type is None or data_subtype is None:
            if col_name in self.transaction.lmd['force_column_usage'] or col_name in self.transaction.lmd['predict_columns']:
                err_msg = f'Failed to deduce type for critical column: {col_name}'
                log.error(err_msg)
                raise Exception(err_msg)

            self.transaction.lmd['columns_to_ignore'].append(col_name)
            stats_v2[col_name]['broken'] = {
                'failed_at': 'Type detection'
                ,'reason': 'Unable to detect type for unknown reasons.'
            }

            if len(col_data) < 1:
                stats_v2[col_name]['broken']['reason'] = 'Unable to detect type due to too many empty, null, None or nan values.'


        if data_subtype_dist:
            self.log.info(f'Data distribution for column "{col_name}" '
                          f'of type "{data_type}" '
                          f'and subtype "{data_subtype}"')
            try:
                self.log.infoChart(data_subtype_dist,
                                   type='list',
                                   uid=f'Data Type Distribution for column "{col_name}"')
            except Exception:
                # Functionality is specific to mindsdb logger
                pass

    self.transaction.lmd['useable_input_columns'] = []
    for col_name in self.transaction.lmd['columns']:
        if col_name not in self.transaction.lmd['columns_to_ignore'] and col_name not in self.transaction.lmd['predict_columns'] and stats_v2[col_name]['broken'] is None and stats_v2[col_name]['identifier'] is None:
                self.transaction.lmd['useable_input_columns'].append(col_name)
