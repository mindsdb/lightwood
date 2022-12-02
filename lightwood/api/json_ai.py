# TODO: _add_implicit_values unit test ensures NO changes for a fully specified file.
from copy import deepcopy
from type_infer.base import TypeInformation
from dataprep_ml import StatisticalAnalysis

from lightwood.helpers.templating import call, inline_dict, align
from lightwood.helpers.templating import _consolidate_analysis_blocks
from type_infer.dtype import dtype
from lightwood.api.types import (
    JsonAI,
    ProblemDefinition,
)
import inspect
from lightwood.helpers.log import log
from lightwood.__about__ import __version__ as lightwood_version


# For custom modules, we create a module loader with necessary imports below
IMPORT_EXTERNAL_DIRS = """
for import_dir in [os.path.join(os.path.expanduser('~/lightwood_modules'), lightwood_version.replace('.', '_')), os.path.join('/etc/lightwood_modules', lightwood_version.replace('.', '_'))]:
    if os.path.exists(import_dir) and os.access(import_dir, os.R_OK):
        for file_name in list(os.walk(import_dir))[0][2]:
            if file_name[-3:] != '.py':
                continue
            mod_name = file_name[:-3]
            loader = importlib.machinery.SourceFileLoader(mod_name,
                                                          os.path.join(import_dir, file_name))
            module = ModuleType(loader.name)
            loader.exec_module(module)
            sys.modules[mod_name] = module
            exec(f'import {mod_name}')
""" # noqa

IMPORTS = """
import lightwood
from lightwood import __version__ as lightwood_version
from lightwood.analysis import *
from lightwood.api import *
from lightwood.data import *
from lightwood.encoder import *
from lightwood.ensemble import *
from lightwood.helpers.device import *
from lightwood.helpers.general import *
from lightwood.helpers.ts import *
from lightwood.helpers.log import *
from lightwood.helpers.numeric import *
from lightwood.helpers.parallelism import *
from lightwood.helpers.seed import *
from lightwood.helpers.text import *
from lightwood.helpers.torch import *
from lightwood.mixer import *

from dataprep_ml.insights import statistical_analysis
from dataprep_ml.cleaners import cleaner
from dataprep_ml.splitters import splitter
from dataprep_ml.imputers import *

import pandas as pd
from typing import Dict, List, Union
import os
from types import ModuleType
import importlib.machinery
import sys
import time
"""


def lookup_encoder(
    col_dtype: str,
    col_name: str,
    is_target: bool,
    problem_defintion: ProblemDefinition,
    is_target_predicting_encoder: bool,
    statistical_analysis: StatisticalAnalysis,
):
    """
    Assign a default encoder for a given column based on its data type, and whether it is a target. Encoders intake raw (but cleaned) data and return an feature representation. This function assigns, per data type, what the featurizer should be. This function runs on each column within the dataset available for model building to assign how it should be featurized.

    Users may override to create a custom encoder to enable their own featurization process. However, in order to generate template JSON-AI, this code runs automatically. Users may edit the generated syntax and use custom approaches while model building.

    For each encoder, "args" may be passed. These args depend an encoder requires during its preparation call.

    :param col_dtype: A data-type of a column specified
    :param col_name: The name of the column
    :param is_target: Whether the column is the target for prediction. If true, only certain possible feature representations are allowed, particularly for complex data types.
    :param problem_definition: The ``ProblemDefinition`` criteria; this populates specifics on how models and encoders may be trained.
    :param is_target_predicting_encoder:
    """  # noqa

    tss = problem_defintion.timeseries_settings
    encoder_lookup = {
        dtype.integer: "NumericEncoder",
        dtype.float: "NumericEncoder",
        dtype.binary: "BinaryEncoder",
        dtype.categorical: "CategoricalAutoEncoder"
        if statistical_analysis is None
        or len(statistical_analysis.histograms[col_name]) > 100
        else "OneHotEncoder",
        dtype.tags: "MultiHotEncoder",
        dtype.date: "DatetimeEncoder",
        dtype.datetime: "DatetimeEncoder",
        dtype.image: "Img2VecEncoder",
        dtype.rich_text: "PretrainedLangEncoder",
        dtype.short_text: "CategoricalAutoEncoder",
        dtype.quantity: "NumericEncoder",
        dtype.audio: "MFCCEncoder",
        dtype.num_array: "NumArrayEncoder",
        dtype.cat_array: "CatArrayEncoder",
        dtype.num_tsarray: "TimeSeriesEncoder",
        dtype.cat_tsarray: "TimeSeriesEncoder",
    }

    # If column is a target, only specific feature representations are allowed that enable supervised tasks
    target_encoder_lookup_override = {
        dtype.rich_text: "VocabularyEncoder",
        dtype.categorical: "OneHotEncoder",
    }

    # Assign a default encoder to each column.
    encoder_dict = {"module": encoder_lookup[col_dtype], "args": {}}

    # If the column is a target, ensure that the feature representation can enable supervised tasks
    if is_target:
        encoder_dict["args"] = {"is_target": "True"}

        if col_dtype in target_encoder_lookup_override:
            encoder_dict["module"] = target_encoder_lookup_override[col_dtype]

        if col_dtype in (dtype.categorical, dtype.binary):
            if problem_defintion.unbias_target:
                encoder_dict["args"][
                    "target_weights"
                ] = "$statistical_analysis.target_weights"
            if problem_defintion.target_weights is not None:
                encoder_dict["args"][
                    "target_weights"
                ] = problem_defintion.target_weights

        if col_dtype in (dtype.integer, dtype.float, dtype.num_array, dtype.num_tsarray):
            encoder_dict["args"][
                "positive_domain"
            ] = "$statistical_analysis.positive_domain"

    # Time-series representations require more advanced flags
    if tss.is_timeseries:
        gby = tss.group_by if tss.group_by is not None else []
        if col_name == tss.order_by:
            encoder_dict["module"] = "ArrayEncoder"
            encoder_dict["args"]["original_type"] = f'"{tss.target_type}"'
            encoder_dict["args"]["window"] = f"{tss.window}"

        if is_target:
            if col_dtype in [dtype.integer]:
                encoder_dict["args"]["grouped_by"] = f"{gby}"
                encoder_dict["module"] = "TsNumericEncoder"
            if col_dtype in [dtype.float]:
                encoder_dict["args"]["grouped_by"] = f"{gby}"
                encoder_dict["module"] = "TsNumericEncoder"
            if tss.horizon > 1:
                encoder_dict["args"]["grouped_by"] = f"{gby}"
                encoder_dict["args"]["timesteps"] = f"{tss.horizon}"
                if col_dtype in [dtype.num_tsarray]:
                    encoder_dict["module"] = "TsArrayNumericEncoder"
                elif col_dtype in [dtype.cat_tsarray]:
                    encoder_dict["module"] = "TsCatArrayEncoder"

        if "__mdb_ts_previous" in col_name or col_name in tss.historical_columns:
            encoder_dict["module"] = "TimeSeriesEncoder"
            encoder_dict["args"]["original_type"] = f'"{tss.target_type}"'
            encoder_dict["args"]["window"] = f"{tss.window}"

    # Set arguments for the encoder
    if encoder_dict["module"] == "PretrainedLangEncoder" and not is_target:
        encoder_dict["args"]["output_type"] = "$dtype_dict[$target]"

    if eval(encoder_dict["module"]).is_trainable_encoder:
        encoder_dict["args"]["stop_after"] = "$problem_definition.seconds_per_encoder"

    if is_target_predicting_encoder:
        encoder_dict["args"]["embed_mode"] = "False"
    return encoder_dict


def generate_json_ai(
    type_information: TypeInformation,
    statistical_analysis: StatisticalAnalysis,
    problem_definition: ProblemDefinition,
) -> JsonAI:
    """
    Given ``type_infer.TypeInformation``, ``dataprep_ml.StatisticalAnalysis``, and the ``ProblemDefinition``, generate a JSON config file with the necessary elements of the ML pipeline populated.

    :param TypeInformation: Specifies what data types each column within the dataset are. Generated by `mindsdb/type_infer`.
    :param statistical_analysis:
    :param problem_definition: Specifies details of the model training/building procedure, as defined by ``ProblemDefinition``

    :returns: JSON-AI object with fully populated details of the ML pipeline
    """  # noqaexec
    exec(IMPORTS, globals())
    exec(IMPORT_EXTERNAL_DIRS, globals())
    target = problem_definition.target
    input_cols = []
    dependency_dict = {}
    tss = problem_definition.timeseries_settings

    dtype_dict_override = problem_definition.dtype_dict
    dtype_dict = type_information.dtypes

    for k in type_information.identifiers:
        if not (tss.is_timeseries and tss.group_by and k in tss.group_by) and k != target:
            del dtype_dict[k]

    for k, v in dtype_dict_override.items():
        dtype_dict[k] = v

    for col_name, col_dtype in dtype_dict.items():
        if (
                (col_name not in type_information.identifiers
                 and col_dtype not in (dtype.invalid, dtype.empty)
                 and col_name != target)
                or
                (tss.group_by is not None and col_name in tss.group_by)
        ):
            if col_name != problem_definition.target:
                input_cols.append(col_name)

    is_target_predicting_encoder = False
    is_ts = problem_definition.timeseries_settings.is_timeseries
    imputers = []

    # Single text column classification
    if (
        len(input_cols) == 1
        and type_information.dtypes[input_cols[0]] in (dtype.rich_text)
        and type_information.dtypes[target] in (dtype.categorical, dtype.binary)
    ):
        is_target_predicting_encoder = True

    submodels = []
    if is_target_predicting_encoder:
        submodels.extend(
            [
                {
                    "module": "Unit",
                    "args": {
                        "target_encoder": "$encoders[self.target]",
                        "stop_after": "$problem_definition.seconds_per_mixer",
                    },
                }
            ]
        )
    else:
        if not tss.is_timeseries:
            submodels.extend(
                [
                    {
                        "module": "Neural",
                        "args": {
                            "fit_on_dev": True,
                            "stop_after": "$problem_definition.seconds_per_mixer",
                            "search_hyperparameters": True,
                        },
                    }
                ]
            )
        else:
            submodels.extend(
                [
                    {
                        "module": "NeuralTs",
                        "args": {
                            "fit_on_dev": True,
                            "stop_after": "$problem_definition.seconds_per_mixer",
                            "search_hyperparameters": True,
                        },
                    }
                ]
            )

        if (not tss.is_timeseries or tss.horizon == 1) and dtype_dict[target] not in (dtype.num_array, dtype.cat_array):
            submodels.extend(
                [
                    {
                        "module": "LightGBM",
                        "args": {
                            "stop_after": "$problem_definition.seconds_per_mixer",
                            "fit_on_dev": True,
                        },
                    },
                    {
                        "module": "Regression",
                        "args": {
                            "stop_after": "$problem_definition.seconds_per_mixer",
                        },
                    },
                    {
                        "module": "RandomForest",
                        "args": {
                            "stop_after": "$problem_definition.seconds_per_mixer",
                            "fit_on_dev": True,
                        },
                    },
                ]
            )
        elif tss.is_timeseries and tss.horizon > 1:
            submodels.extend(
                [
                    {
                        "module": "LightGBMArray",
                        "args": {
                            "fit_on_dev": True,
                            "stop_after": "$problem_definition.seconds_per_mixer",
                            "ts_analysis": "$ts_analysis",
                            "tss": "$problem_definition.timeseries_settings",
                        },
                    }
                ]
            )

            if tss.use_previous_target and dtype_dict[target] in (dtype.integer, dtype.float, dtype.quantity):
                submodels.extend(
                    [
                        {
                            "module": "SkTime",
                            "args": {
                                "stop_after": "$problem_definition.seconds_per_mixer",
                                "horizon": "$problem_definition.timeseries_settings.horizon",
                            },
                        },
                        {
                            "module": "ETSMixer",
                            "args": {
                                "stop_after": "$problem_definition.seconds_per_mixer",
                                "horizon": "$problem_definition.timeseries_settings.horizon",
                            },
                        },
                        {
                            "module": "ARIMAMixer",
                            "args": {
                                "stop_after": "$problem_definition.seconds_per_mixer",
                                "horizon": "$problem_definition.timeseries_settings.horizon",
                            },
                        }
                    ]
                )

    model = {
        "module": "BestOf",
        "args": {
            "submodels": submodels,
        }
    }

    num_ts_dtypes = (dtype.integer, dtype.float, dtype.quantity)
    if tss.is_timeseries and tss.horizon > 1:
        if dtype_dict[target] in num_ts_dtypes:
            dtype_dict[target] = dtype.num_tsarray
            problem_definition.anomaly_detection = True
        else:
            dtype_dict[target] = dtype.cat_tsarray
    elif tss.is_timeseries and dtype_dict[target] in num_ts_dtypes:
        problem_definition.anomaly_detection = True

    encoders = {
        target: lookup_encoder(
            dtype_dict[target],
            target,
            True,
            problem_definition,
            False,
            statistical_analysis,
        )
    }

    for col in input_cols:
        encoders[col] = lookup_encoder(
            dtype_dict[col],
            col,
            False,
            problem_definition,
            is_target_predicting_encoder,
            statistical_analysis,
        )

    # Decide on the accuracy functions to use
    output_dtype = dtype_dict[target]
    if output_dtype in [
        dtype.integer,
        dtype.float,
        dtype.date,
        dtype.datetime,
        dtype.quantity,
    ]:
        accuracy_functions = ["r2_score"]
    elif output_dtype in [dtype.categorical, dtype.tags, dtype.binary]:
        accuracy_functions = ["balanced_accuracy_score"]
    elif output_dtype in (dtype.num_tsarray, ):
        accuracy_functions = ["complementary_smape_array_accuracy"]
    elif output_dtype in (dtype.num_array, ):
        accuracy_functions = ["evaluate_num_array_accuracy"]
    elif output_dtype in (dtype.cat_array, dtype.cat_tsarray):
        accuracy_functions = ["evaluate_cat_array_accuracy"]
    else:
        raise Exception(
            f"Please specify a custom accuracy function for output type {output_dtype}"
        )

    if is_ts:
        if output_dtype in [dtype.integer, dtype.float, dtype.quantity]:
            # forces this acc fn for t+1 time series forecasters
            accuracy_functions = ["complementary_smape_array_accuracy"]

        if output_dtype in (dtype.integer, dtype.float, dtype.quantity, dtype.num_tsarray):
            imputers.append({"module": "NumericalImputer",
                             "args": {
                                 "value": "'zero'",
                                 "target": f"'{target}'"}}
                            )
        elif output_dtype in [dtype.categorical, dtype.tags, dtype.binary, dtype.cat_tsarray]:
            imputers.append({"module": "CategoricalImputer",
                             "args": {
                                 "value": "'mode'",
                                 "target": f"'{target}'"}}
                            )

    if problem_definition.time_aim is None:
        # 5 days
        problem_definition.time_aim = 3 * 24 * 3600

    # Encoders are assigned 1/3 of the time unless a user overrides this (equal time per encoder)
    if problem_definition.seconds_per_encoder is None:
        nr_trainable_encoders = len(
            [
                x
                for x in encoders.values()
                if eval(x["module"]).is_trainable_encoder
            ]
        )
        if nr_trainable_encoders > 0:
            problem_definition.seconds_per_encoder = 0.33 * problem_definition.time_aim / nr_trainable_encoders

    # Mixers are assigned 1/3 of the time aim (or 2/3 if there are no trainable encoders )\
    # unless a user overrides this (equal time per mixer)
    if problem_definition.seconds_per_mixer is None:
        if problem_definition.seconds_per_encoder is None:
            problem_definition.seconds_per_mixer = 0.66 * problem_definition.time_aim / len(model['args']['submodels'])
        else:
            problem_definition.seconds_per_mixer = 0.33 * problem_definition.time_aim / len(model['args']['submodels'])

    return JsonAI(
        cleaner=None,
        splitter=None,
        analyzer=None,
        explainer=None,
        encoders=encoders,
        imputers=imputers,
        dtype_dict=dtype_dict,
        dependency_dict=dependency_dict,
        model=model,
        problem_definition=problem_definition,
        identifiers=type_information.identifiers,
        timeseries_transformer=None,
        timeseries_analyzer=None,
        accuracy_functions=accuracy_functions,
    )


def _merge_implicit_values(field: dict, implicit_value: dict) -> dict:
    """
    Helper function for `_populate_implicit_field`.
    Takes a user-defined field along with its implicit value, and merges them together.

    :param field: JsonAI field with user-defined parameters.
    :param implicit_value: implicit values for the field.
    :return: original field with implicit values merged into it.
    """
    exec(IMPORTS, globals())
    exec(IMPORT_EXTERNAL_DIRS, globals())
    module = eval(field["module"])

    if inspect.isclass(module):
        args = list(inspect.signature(module.__init__).parameters.keys())[1:]
    else:
        args = module.__code__.co_varnames

    for arg in args:
        if "args" not in field:
            field["args"] = implicit_value["args"]
        else:
            if arg not in field["args"]:
                if arg in implicit_value["args"]:
                    field["args"][arg] = implicit_value["args"][arg]

    return field


def _populate_implicit_field(
    json_ai: JsonAI, field_name: str, implicit_value: dict, is_timeseries: bool
) -> None:
    """
    Populate the implicit field of the JsonAI, either by filling it in entirely if missing, or by introspecting the class or function and assigning default values to the args in it's signature that are in the implicit default but haven't been populated by the user

    :params: json_ai: ``JsonAI`` object that describes the ML pipeline that may not have every detail fully specified.
    :params: field_name: Name of the field the implicit field in ``JsonAI``
    :params: implicit_value: The dictionary containing implicit values for the module and arg in the field
    :params: is_timeseries: Whether or not this is a timeseries problem

    :returns: nothing, this method mutates the respective field of the ``JsonAI`` object it receives
    """  # noqa
    # These imports might be slow, in which case the only <easy> solution is to line this code
    field = json_ai.__getattribute__(field_name)
    if field is None:
        # This if is to only populated timeseries-specific implicit fields for implicit problems
        if is_timeseries or field_name not in (
            "timeseries_analyzer",
            "timeseries_transformer",
        ):
            field = implicit_value

    # If the user specified one or more subfields in a field that's a list
    # Populate them with implicit arguments form the implicit values from that subfield
    elif isinstance(field, list) and isinstance(implicit_value, list):
        for i in range(len(field)):
            sub_field_implicit = [
                x for x in implicit_value if x["module"] == field[i]["module"]
            ]
            if len(sub_field_implicit) == 1:
                field[i] = _merge_implicit_values(field[i], sub_field_implicit[0])
        for sub_field_implicit in implicit_value:
            if (
                len([x for x in field if x["module"] == sub_field_implicit["module"]])
                == 0
            ):
                field.append(sub_field_implicit)
    # If the user specified the field, add implicit arguments which we didn't specify
    else:
        field = _merge_implicit_values(field, implicit_value)
    json_ai.__setattr__(field_name, field)


def _add_implicit_values(json_ai: JsonAI) -> JsonAI:
    """
    To enable brevity in writing, auto-generate the "unspecified/missing" details required in the ML pipeline.

    :params: json_ai: ``JsonAI`` object that describes the ML pipeline that may not have every detail fully specified.

    :returns: ``JSONAI`` object with all necessary parameters that were previously left unmentioned filled in.
    """
    problem_definition = json_ai.problem_definition
    tss = problem_definition.timeseries_settings
    is_ts = tss.is_timeseries

    # Add implicit ensemble arguments
    json_ai.model["args"]["target"] = json_ai.model["args"].get("target", "$target")
    json_ai.model["args"]["data"] = json_ai.model["args"].get("data", "encoded_test_data")
    json_ai.model["args"]["mixers"] = json_ai.model["args"].get("mixers", "$mixers")
    json_ai.model["args"]["fit"] = json_ai.model["args"].get("fit", True)
    json_ai.model["args"]["args"] = json_ai.model["args"].get("args", "$pred_args")  # TODO correct?

    # @TODO: change this to per-parameter basis and signature inspection
    if json_ai.model["module"] in ("BestOf", "ModeEnsemble", "WeightedMeanEnsemble"):
        json_ai.model["args"]["accuracy_functions"] = json_ai.model["args"].get("accuracy_functions",
                                                                                "$accuracy_functions")

    if json_ai.model["module"] in ("BestOf", "TsStackedEnsemble", "WeightedMeanEnsemble"):
        tsa_val = "self.ts_analysis" if is_ts else None
        json_ai.model["args"]["ts_analysis"] = json_ai.model["args"].get("ts_analysis", tsa_val)

    if json_ai.model["module"] in ("MeanEnsemble", "ModeEnsemble", "StackedEnsemble", "TsStackedEnsemble",
                                   "WeightedMeanEnsemble"):
        json_ai.model["args"]["dtype_dict"] = json_ai.model["args"].get("dtype_dict", "$dtype_dict")

    # Add implicit mixer arguments
    mixers = json_ai.model['args']['submodels']
    for i in range(len(mixers)):
        if not mixers[i].get("args", False):
            mixers[i]["args"] = {}

        if mixers[i]["module"] == "Unit":
            continue

        # common
        mixers[i]["args"]["target"] = mixers[i]["args"].get("target", "$target")
        mixers[i]["args"]["dtype_dict"] = mixers[i]["args"].get("dtype_dict", "$dtype_dict")
        mixers[i]["args"]["stop_after"] = mixers[i]["args"].get("stop_after", "$problem_definition.seconds_per_mixer")

        # specific
        if mixers[i]["module"] in ("Neural", "NeuralTs"):
            mixers[i]["args"]["target_encoder"] = mixers[i]["args"].get(
                "target_encoder", "$encoders[self.target]"
            )
            mixers[i]["args"]["net"] = mixers[i]["args"].get(
                "net",
                '"DefaultNet"'
                if not tss.is_timeseries or not tss.use_previous_target
                else '"ArNet"',
            )
            if mixers[i]["module"] == "NeuralTs":
                mixers[i]["args"]["timeseries_settings"] = mixers[i]["args"].get(
                    "timeseries_settings", "$problem_definition.timeseries_settings"
                )
                mixers[i]["args"]["ts_analysis"] = mixers[i]["args"].get("ts_analysis", "$ts_analysis")

        elif mixers[i]["module"] == "LightGBM":
            mixers[i]["args"]["input_cols"] = mixers[i]["args"].get(
                "input_cols", "$input_cols"
            )
            mixers[i]["args"]["target_encoder"] = mixers[i]["args"].get(
                "target_encoder", "$encoders[self.target]"
            )
            mixers[i]["args"]["use_optuna"] = True

        elif mixers[i]["module"] == "Regression":
            mixers[i]["args"]["target_encoder"] = mixers[i]["args"].get(
                "target_encoder", "$encoders[self.target]"
            )

        elif mixers[i]["module"] == "RandomForest":
            mixers[i]["args"]["target_encoder"] = mixers[i]["args"].get(
                "target_encoder", "$encoders[self.target]"
            )
            mixers[i]["args"]["use_optuna"] = True

        elif mixers[i]["module"] == "LightGBMArray":
            mixers[i]["args"]["input_cols"] = mixers[i]["args"].get(
                "input_cols", "$input_cols"
            )
            mixers[i]["args"]["target_encoder"] = mixers[i]["args"].get(
                "target_encoder", "$encoders[self.target]"
            )
            mixers[i]["args"]["tss"] = mixers[i]["args"].get("tss", "$problem_definition.timeseries_settings")
            mixers[i]["args"]["ts_analysis"] = mixers[i]["args"].get("ts_analysis", "$ts_analysis")
            mixers[i]["args"]["fit_on_dev"] = mixers[i]["args"].get("fit_on_dev", "True")
            mixers[i]["args"]["use_stl"] = mixers[i]["args"].get("use_stl", "False")

        elif mixers[i]["module"] in ("NHitsMixer", "GluonTSMixer"):
            mixers[i]["args"]["horizon"] = "$problem_definition.timeseries_settings.horizon"
            mixers[i]["args"]["window"] = "$problem_definition.timeseries_settings.window"
            mixers[i]["args"]["ts_analysis"] = mixers[i]["args"].get(
                "ts_analysis", "$ts_analysis"
            )
            problem_definition.fit_on_all = False  # takes too long otherwise

        elif mixers[i]["module"] in ("SkTime", "ProphetMixer", "ETSMixer", "ARIMAMixer"):
            mixers[i]["args"]["ts_analysis"] = mixers[i]["args"].get(
                "ts_analysis", "$ts_analysis"
            )
            if "horizon" not in mixers[i]["args"]:
                mixers[i]["args"]["horizon"] = "$problem_definition.timeseries_settings.horizon"

            # enforce fit_on_all if this mixer is specified
            problem_definition.fit_on_all = True

    for name in json_ai.encoders:
        if name not in json_ai.dependency_dict:
            json_ai.dependency_dict[name] = []

    # Add "hidden" fields
    hidden_fields = {
        "cleaner": {
            "module": "cleaner",
            "args": {
                "pct_invalid": "$problem_definition.pct_invalid",
                "identifiers": "$identifiers",
                "data": "data",
                "dtype_dict": "$dtype_dict",
                "target": "$target",
                "mode": "$mode",
                "imputers": "$imputers",
                "timeseries_settings": "$problem_definition.timeseries_settings.to_dict()",
                "anomaly_detection": "$problem_definition.anomaly_detection",
            },
        },
        "splitter": {
            "module": "splitter",
            "args": {
                "tss": "$problem_definition.timeseries_settings.to_dict()",
                "data": "data",
                "seed": "$problem_definition.seed_nr",
                "target": "$target",
                "dtype_dict": "$dtype_dict",
                "pct_train": 0.8,
                "pct_dev": 0.1,
                "pct_test": 0.1,
            },
        },
        "analyzer": {
            "module": "model_analyzer",
            "args": {
                "stats_info": "$statistical_analysis",
                "tss": "$problem_definition.timeseries_settings",
                "accuracy_functions": "$accuracy_functions",
                "predictor": "$ensemble",
                "data": "encoded_test_data",
                "train_data": "encoded_train_data",
                "target": "$target",
                "dtype_dict": "$dtype_dict",
                "analysis_blocks": "$analysis_blocks",
                "ts_analysis": "$ts_analysis" if is_ts else None,
            },
        },
        "explainer": {
            "module": "explain",
            "args": {
                "problem_definition": "$problem_definition",
                "stat_analysis": "$statistical_analysis",
                "data": "data",
                "encoded_data": "encoded_data",
                "predictions": "df",
                "runtime_analysis": "$runtime_analyzer",
                "ts_analysis": "$ts_analysis" if is_ts else None,
                "target_name": "$target",
                "target_dtype": "$dtype_dict[self.target]",
                "explainer_blocks": "$analysis_blocks",
                "pred_args": "$pred_args",
            },
        },
        "analysis_blocks": [
            {
                "module": "ICP",
                "args": {
                    "fixed_significance": None,
                    "confidence_normalizer": False,
                },
            },
            {
                "module": "AccStats",
                "args": {"deps": ["ICP"]},
            },
            {
                "module": "ConfStats",
                "args": {"deps": ["ICP"]},
            },
            {
                "module": "PermutationFeatureImportance",
                "args": {"deps": ["AccStats"]},
            },
        ] if problem_definition.use_default_analysis else [],
        "timeseries_transformer": {
            "module": "transform_timeseries",
            "args": {
                "timeseries_settings": "$problem_definition.timeseries_settings",
                "data": "data",
                "dtype_dict": "$dtype_dict",
                "target": "$target",
                "mode": "$mode",
                "ts_analysis": "$ts_analysis"
            },
        },
        "timeseries_analyzer": {
            "module": "timeseries_analyzer",
            "args": {
                "timeseries_settings": "$problem_definition.timeseries_settings",
                "data": "data",
                "dtype_dict": "$dtype_dict",
                "target": "$target",
            },
        },
    }

    for field_name, implicit_value in hidden_fields.items():
        _populate_implicit_field(json_ai, field_name, implicit_value, tss.is_timeseries)

    # further consolidation
    to_inspect = ['analysis_blocks']
    consolidation_methods = {
        'analysis_blocks': _consolidate_analysis_blocks
    }
    for k in to_inspect:
        method = consolidation_methods[k]
        setattr(json_ai, k, method(json_ai, k))

    return json_ai


def code_from_json_ai(json_ai: JsonAI) -> str:
    """
    Generates a custom ``PredictorInterface`` given the specifications from ``JsonAI`` object.

    :param json_ai: ``JsonAI`` object with fully specified parameters

    :returns: Automated syntax of the ``PredictorInterface`` object.
    """
    json_ai = deepcopy(json_ai)
    # ----------------- #
    # Fill in any missing values
    json_ai = _add_implicit_values(json_ai)

    # ----------------- #

    # Instantiate data types
    dtype_dict = {}

    for k in json_ai.dtype_dict:
        if json_ai.dtype_dict[k] not in (dtype.invalid, dtype.empty):
            dtype_dict[k] = json_ai.dtype_dict[k]

    # Populate imputers
    imputer_dict = {}
    if json_ai.imputers:
        for imputer in json_ai.imputers:
            imputer_dict[imputer['args']['target'].replace('\'', '').replace('\"', '')] = call(imputer)
    json_ai.imputers = imputer_dict
    imputers = inline_dict(json_ai.imputers)

    # Populate encoders
    encoder_dict = {}
    for col_name, encoder in json_ai.encoders.items():
        encoder_dict[col_name] = call(encoder)

    # Populate time-series specific details
    # TODO: consider moving this to a `JsonAI override` phase
    tss = json_ai.problem_definition.timeseries_settings
    if tss.is_timeseries:
        if tss.use_previous_target:
            col_name = f"__mdb_ts_previous_{json_ai.problem_definition.target}"
            target_type = json_ai.dtype_dict[json_ai.problem_definition.target]
            json_ai.problem_definition.timeseries_settings.target_type = target_type
            encoder_dict[col_name] = call(
                lookup_encoder(
                    target_type,
                    col_name,
                    False,
                    json_ai.problem_definition,
                    False,
                    None,
                )
            )

            dtype_dict[col_name] = target_type
            # @TODO: Is populating the json_ai at this stage even necessary?
            json_ai.encoders[col_name] = encoder_dict[col_name]
            json_ai.dtype_dict[col_name] = target_type
            json_ai.dependency_dict[col_name] = []

    # ----------------- #

    input_cols = [x.replace("'", "\\'").replace('"', '\\"') for x in json_ai.encoders
                  if x != json_ai.problem_definition.target]
    input_cols = ",".join([f"""'{name}'""" for name in input_cols])

    # ----------------- #
    # Time-series specific code blocks
    # ----------------- #

    ts_transform_code = ""
    ts_analyze_code = None
    ts_encoder_code = ""
    if json_ai.timeseries_transformer is not None:
        ts_transform_code = f"""
log.info('Transforming timeseries data')
data = {call(json_ai.timeseries_transformer)}
"""
        ts_analyze_code = f"""
self.ts_analysis = {call(json_ai.timeseries_analyzer)}
"""
    # @TODO: set these kwargs/properties in the json ai construction (if possible)
    if json_ai.timeseries_analyzer is not None:
        ts_encoder_code = """
if encoder.is_timeseries_encoder:
    kwargs['ts_analysis'] = self.ts_analysis
"""

    if json_ai.problem_definition.timeseries_settings.is_timeseries:
        ts_target_code = """
if encoder.is_target:
    encoder.normalizers = self.ts_analysis['target_normalizers']
    encoder.group_combinations = self.ts_analysis['group_combinations']
"""
    else:
        ts_target_code = ""

    # ----------------- #
    # Statistical Analysis Body
    # ----------------- #

    analyze_data_body = f"""
self.statistical_analysis = statistical_analysis(data,
                                                 self.dtype_dict,
                                                 self.problem_definition.to_dict(),
                                                 {json_ai.identifiers})

# Instantiate post-training evaluation
self.analysis_blocks = [{', '.join([call(block) for block in json_ai.analysis_blocks])}]
    """

    analyze_data_body = align(analyze_data_body, 2)

    # ----------------- #
    # Pre-processing Body
    # ----------------- #

    clean_body = f"""
log.info('Cleaning the data')
self.imputers = {imputers}
data = {call(json_ai.cleaner)}

# Time-series blocks
{ts_transform_code}
"""

    clean_body += '\nreturn data'

    clean_body = align(clean_body, 2)

    # ----------------- #
    # Train-Test Splitter Body
    # ----------------- #

    split_body = f"""
log.info("Splitting the data into train/test")
train_test_data = {call(json_ai.splitter)}

return train_test_data
    """

    split_body = align(split_body, 2)

    # ----------------- #
    # Prepare features Body
    # ----------------- #

    prepare_body = """
self.mode = 'train'

if self.statistical_analysis is None:
    raise Exception("Please run analyze_data first")
"""
    if ts_analyze_code is not None:
        prepare_body += f"""
if self.mode != 'predict':
    {align(ts_analyze_code, 1)}
"""

    prepare_body += f"""
# Column to encoder mapping
self.encoders = {inline_dict(encoder_dict)}

# Prepare the training + dev data
concatenated_train_dev = pd.concat([data['train'], data['dev']])

encoder_prepping_dict = {{}}

# Prepare encoders that do not require learned strategies
for col_name, encoder in self.encoders.items():
    if col_name != self.target and not encoder.is_trainable_encoder:
        encoder_prepping_dict[col_name] = [encoder, concatenated_train_dev[col_name], 'prepare']

# Setup parallelization
parallel_prepped_encoders = mut_method_call(encoder_prepping_dict)
for col_name, encoder in parallel_prepped_encoders.items():
    self.encoders[col_name] = encoder

# Prepare the target
if self.target not in parallel_prepped_encoders:
    if self.encoders[self.target].is_trainable_encoder:
        self.encoders[self.target].prepare(data['train'][self.target], data['dev'][self.target])
    else:
        self.encoders[self.target].prepare(pd.concat([data['train'], data['dev']])[self.target])

# Prepare any non-target encoders that are learned
for col_name, encoder in self.encoders.items():
    if col_name != self.target and encoder.is_trainable_encoder:
        priming_data = pd.concat([data['train'], data['dev']])
        kwargs = {{}}
        if self.dependencies[col_name]:
            kwargs['dependency_data'] = {{}}
            for col in self.dependencies[col_name]:
                kwargs['dependency_data'][col] = {{
                    'original_type': self.dtype_dict[col],
                    'data': priming_data[col]
                }}
            {align(ts_encoder_code, 3)}

        # If an encoder representation requires the target, provide priming data
        if hasattr(encoder, 'uses_target'):
            kwargs['encoded_target_values'] = self.encoders[self.target].encode(priming_data[self.target])

        encoder.prepare(data['train'][col_name], data['dev'][col_name], **kwargs)

    {align(ts_target_code, 1)}
"""
    prepare_body = align(prepare_body, 2)

    # ----------------- #
    # Featurize Data Body
    # ----------------- #

    feature_body = f"""
log.info('Featurizing the data')

feature_data = {{ key: EncodedDs(self.encoders, data, self.target) for key, data in split_data.items() if key != "stratified_on"}}

return feature_data

"""  # noqa

    feature_body = align(feature_body, 2)

    # ----------------- #
    # Fit Mixer Body
    # ----------------- #

    fit_body = f"""
self.mode = 'train'

# --------------- #
# Extract data
# --------------- #
# Extract the featurized data into train/dev/test
encoded_train_data = enc_data['train']
encoded_dev_data = enc_data['dev']
encoded_test_data = enc_data['test']
filtered_df = filter_ds(encoded_test_data, self.problem_definition.timeseries_settings)
encoded_test_data = EncodedDs(encoded_test_data.encoders, filtered_df, encoded_test_data.target)

log.info('Training the mixers')

# --------------- #
# Fit Models
# --------------- #
# Assign list of mixers
self.mixers = [{', '.join([call(x) for x in json_ai.model["args"]["submodels"]])}]

# Train mixers
trained_mixers = []
for mixer in self.mixers:
    try:
        self.fit_mixer(mixer, encoded_train_data, encoded_dev_data)
        trained_mixers.append(mixer)
    except Exception as e:
        log.warning(f'Exception: {{e}} when training mixer: {{mixer}}')
        if {json_ai.problem_definition.strict_mode} and mixer.stable:
            raise e

# Update mixers to trained versions
self.mixers = trained_mixers

# --------------- #
# Create Ensembles
# --------------- #
log.info('Ensembling the mixer')
# Create an ensemble of mixers to identify best performing model
self.pred_args = PredictionArguments()
# Dirty hack
self.ensemble = {call(json_ai.model)}
self.supports_proba = self.ensemble.supports_proba
"""
    fit_body = align(fit_body, 2)

    # ----------------- #
    # Analyze Ensemble Body
    # ----------------- #

    analyze_ensemble = f"""

# --------------- #
# Extract data
# --------------- #
# Extract the featurized data into train/dev/test
encoded_train_data = enc_data['train']
encoded_dev_data = enc_data['dev']
encoded_test_data = enc_data['test']

# --------------- #
# Analyze Ensembles
# --------------- #
log.info('Analyzing the ensemble of mixers')
self.model_analysis, self.runtime_analyzer = {call(json_ai.analyzer)}
"""
    analyze_ensemble = align(analyze_ensemble, 2)

    # ----------------- #
    # Adjust Ensemble Body
    # ----------------- #

    adjust_body = f"""
self.mode = 'train'

# --------------- #
# Prepare data
# --------------- #
if dev_data is None:
    data = train_data
    split = splitter(
        data=data,
        pct_train=0.8,
        pct_dev=0.2,
        pct_test=0,
        tss=self.problem_definition.timeseries_settings.to_dict(),
        seed=self.problem_definition.seed_nr,
        target=self.target,
        dtype_dict=self.dtype_dict)
    train_data = split['train']
    dev_data = split['dev']

if adjust_args is None or not adjust_args.get('learn_call'):
    train_data = self.preprocess(train_data)
    dev_data = self.preprocess(dev_data)

dev_data = EncodedDs(self.encoders, dev_data, self.target)
train_data = EncodedDs(self.encoders, train_data, self.target)

# --------------- #
# Update/Adjust Mixers
# --------------- #
log.info('Updating the mixers')

for mixer in self.mixers:
    mixer.partial_fit(train_data, dev_data, adjust_args)
"""  # noqa

    adjust_body = align(adjust_body, 2)

    # ----------------- #
    # Learn Body
    # ----------------- #

    learn_body = """
self.mode = 'train'
n_phases = 8 if self.problem_definition.fit_on_all else 7

# Perform stats analysis
log.info(f'[Learn phase 1/{n_phases}] - Statistical analysis')
self.analyze_data(data)

# Pre-process the data
log.info(f'[Learn phase 2/{n_phases}] - Data preprocessing')
data = self.preprocess(data)

# Create train/test (dev) split
log.info(f'[Learn phase 3/{n_phases}] - Data splitting')
train_dev_test = self.split(data)

# Prepare encoders
log.info(f'[Learn phase 4/{n_phases}] - Preparing encoders')
self.prepare(train_dev_test)

# Create feature vectors from data
log.info(f'[Learn phase 5/{n_phases}] - Feature generation')
enc_train_test = self.featurize(train_dev_test)

# Prepare mixers
log.info(f'[Learn phase 6/{n_phases}] - Mixer training')
self.fit(enc_train_test)

# Analyze the ensemble
log.info(f'[Learn phase 7/{n_phases}] - Ensemble analysis')
self.analyze_ensemble(enc_train_test)

# ------------------------ #
# Enable model partial fit AFTER it is trained and evaluated for performance with the appropriate train/dev/test splits.
# This assumes the predictor could continuously evolve, hence including reserved testing data may improve predictions.
# SET `json_ai.problem_definition.fit_on_all=False` TO TURN THIS BLOCK OFF.

# Update the mixers with partial fit
if self.problem_definition.fit_on_all:

    log.info(f'[Learn phase 8/{n_phases}] - Adjustment on validation requested')
    self.adjust(enc_train_test["test"].data_frame, ConcatedEncodedDs([enc_train_test["train"],
                                                                      enc_train_test["dev"]]).data_frame,
                                                                      adjust_args={'learn_call': True})

"""
    learn_body = align(learn_body, 2)
    # ----------------- #
    # Predict Body
    # ----------------- #

    predict_body = f"""
self.mode = 'predict'
n_phases = 3 if self.pred_args.all_mixers else 4

if len(data) == 0:
    raise Exception("Empty input, aborting prediction. Please try again with some input data.")

log.info(f'[Predict phase 1/{{n_phases}}] - Data preprocessing')
if self.problem_definition.ignore_features:
    log.info(f'Dropping features: {{self.problem_definition.ignore_features}}')
    data = data.drop(columns=self.problem_definition.ignore_features, errors='ignore')
for col in self.input_cols:
    if col not in data.columns:
        data[col] = [None] * len(data)

# Pre-process the data
data = self.preprocess(data)

# Featurize the data
log.info(f'[Predict phase 2/{{n_phases}}] - Feature generation')
encoded_ds = self.featurize({{"predict_data": data}})["predict_data"]
encoded_data = encoded_ds.get_encoded_data(include_target=False)

log.info(f'[Predict phase 3/{{n_phases}}] - Calling ensemble')
self.pred_args = PredictionArguments.from_dict(args)
df = self.ensemble(encoded_ds, args=self.pred_args)

if self.pred_args.all_mixers:
    return df
else:
    log.info(f'[Predict phase 4/{{n_phases}}] - Analyzing output')
    insights, global_insights = {call(json_ai.explainer)}
    return insights
"""

    predict_body = align(predict_body, 2)

    predictor_code = f"""
{IMPORTS}
{IMPORT_EXTERNAL_DIRS}

class Predictor(PredictorInterface):
    target: str
    mixers: List[BaseMixer]
    encoders: Dict[str, BaseEncoder]
    ensemble: BaseEnsemble
    mode: str

    def __init__(self):
        seed({json_ai.problem_definition.seed_nr})
        self.target = '{json_ai.problem_definition.target}'
        self.mode = 'inactive'
        self.problem_definition = ProblemDefinition.from_dict({json_ai.problem_definition.to_dict()})
        self.accuracy_functions = {json_ai.accuracy_functions}
        self.identifiers = {json_ai.identifiers}
        self.dtype_dict = {inline_dict(dtype_dict)}
        self.lightwood_version = '{lightwood_version}'

        # Any feature-column dependencies
        self.dependencies = {inline_dict(json_ai.dependency_dict)}

        self.input_cols = [{input_cols}]

        # Initial stats analysis
        self.statistical_analysis = None
        self.ts_analysis = None
        self.runtime_log = dict()

    @timed
    def analyze_data(self, data: pd.DataFrame) -> None:
        # Perform a statistical analysis on the unprocessed data
{analyze_data_body}

    @timed
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Preprocess and clean data
{clean_body}

    @timed
    def split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Split the data into training/testing splits
{split_body}

    @timed
    def prepare(self, data: Dict[str, pd.DataFrame]) -> None:
        # Prepare encoders to featurize data
{prepare_body}

    @timed
    def featurize(self, split_data: Dict[str, pd.DataFrame]):
        # Featurize data into numerical representations for models
{feature_body}

    @timed
    def fit(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        # Fit predictors to estimate target
{fit_body}

    @timed
    def fit_mixer(self, mixer, encoded_train_data, encoded_dev_data) -> None:
        mixer.fit(encoded_train_data, encoded_dev_data)

    @timed
    def analyze_ensemble(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        # Evaluate quality of fit for the ensemble of mixers
{analyze_ensemble}

    @timed
    def learn(self, data: pd.DataFrame) -> None:
        if self.problem_definition.ignore_features:
            log.info(f'Dropping features: {{self.problem_definition.ignore_features}}')
            data = data.drop(columns=self.problem_definition.ignore_features, errors='ignore')
{learn_body}

    @timed
    def adjust(self, train_data: Union[EncodedDs, ConcatedEncodedDs, pd.DataFrame],
        dev_data: Optional[Union[EncodedDs, ConcatedEncodedDs, pd.DataFrame]] = None,
        adjust_args: Optional[dict] = None) -> None:
        # Update mixers with new information
{adjust_body}

    @timed
    def predict(self, data: pd.DataFrame, args: Dict = {{}}) -> pd.DataFrame:
{predict_body}
"""

    try:
        import black
    except Exception:
        black = None

    if black is not None:
        log.info('Unable to import black formatter, predictor code might be a bit ugly.')
        predictor_code = black.format_str(predictor_code, mode=black.FileMode())

    return predictor_code


def validate_json_ai(json_ai: JsonAI) -> bool:
    """
    Checks the validity of a ``JsonAI`` object

    :param json_ai: A ``JsonAI`` object

    :returns: Whether the JsonAI is valid, i.e. doesn't contain prohibited values, unknown values and can be turned into code.
    """ # noqa
    from lightwood.api.high_level import predictor_from_code, code_from_json_ai

    try:
        predictor_from_code(code_from_json_ai(json_ai))
        return True
    except Exception:
        return False
