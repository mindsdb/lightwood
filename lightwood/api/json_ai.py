# TODO: _add_implicit_values unit test ensures NO changes for a fully specified file.
from typing import Dict
from lightwood.helpers.templating import call, inline_dict, align
from lightwood.api import dtype
import numpy as np
from lightwood.api.types import (
    JsonAI,
    TypeInformation,
    StatisticalAnalysis,
    Feature,
    Output,
    ProblemDefinition,
)
import inspect
from lightwood.helpers.log import log


# For custom modules, we create a module loader with necessary imports below
IMPORT_EXTERNAL_DIRS = """
for import_dir in [os.path.expanduser('~/lightwood_modules'), '/etc/lightwood_modules']:
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
"""

IMPORTS = """
import lightwood
from lightwood.analysis import *
from lightwood.api import *
from lightwood.data import *
from lightwood.encoder import *
from lightwood.ensemble import *
from lightwood.helpers.device import *
from lightwood.helpers.general import *
from lightwood.helpers.log import *
from lightwood.helpers.numeric import *
from lightwood.helpers.parallelism import *
from lightwood.helpers.seed import *
from lightwood.helpers.text import *
from lightwood.helpers.torch import *
from lightwood.mixer import *
import pandas as pd
from typing import Dict, List
import os
from types import ModuleType
import importlib.machinery
import sys
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
        dtype.integer: "Integer.NumericEncoder",
        dtype.float: "Float.NumericEncoder",
        dtype.binary: "Binary.BinaryEncoder",
        dtype.categorical: "Categorical.CategoricalAutoEncoder"
        if statistical_analysis is None
        or len(statistical_analysis.histograms[col_name]) > 100
        else "Categorical.OneHotEncoder",
        dtype.tags: "Tags.MultiHotEncoder",
        dtype.date: "Date.DatetimeEncoder",
        dtype.datetime: "Datetime.DatetimeEncoder",
        dtype.image: "Image.Img2VecEncoder",
        dtype.rich_text: "Rich_Text.PretrainedLangEncoder",
        dtype.short_text: "Short_Text.CategoricalAutoEncoder",
        dtype.array: "Array.ArrayEncoder",
        dtype.tsarray: "TimeSeries.TimeSeriesEncoder",
        dtype.quantity: "Quantity.NumericEncoder",
        dtype.audio: "Audio.MFCCEncoder"
    }

    # If column is a target, only specific feature representations are allowed that enable supervised tasks
    target_encoder_lookup_override = {
        dtype.rich_text: "Rich_Text.VocabularyEncoder",
        dtype.categorical: "Categorical.OneHotEncoder",
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
                    "target_class_distribution"
                ] = "$statistical_analysis.target_class_distribution"

        if col_dtype in (dtype.integer, dtype.float, dtype.array, dtype.tsarray):
            encoder_dict["args"][
                "positive_domain"
            ] = "$statistical_analysis.positive_domain"

    # Time-series representations require more advanced flags
    if tss.is_timeseries:
        gby = tss.group_by if tss.group_by is not None else []
        if col_name in tss.order_by + tss.historical_columns:
            encoder_dict["module"] = col_dtype.capitalize() + ".TimeSeriesEncoder"
            encoder_dict["args"]["original_type"] = f'"{col_dtype}"'
            encoder_dict["args"]["target"] = "self.target"
            encoder_dict["args"]["grouped_by"] = f"{gby}"

        if is_target:
            if col_dtype in [dtype.integer]:
                encoder_dict["args"]["grouped_by"] = f"{gby}"
                encoder_dict["module"] = "Integer.TsNumericEncoder"
            if col_dtype in [dtype.float]:
                encoder_dict["args"]["grouped_by"] = f"{gby}"
                encoder_dict["module"] = "Float.TsNumericEncoder"
            if tss.nr_predictions > 1:
                encoder_dict["args"]["grouped_by"] = f"{gby}"
                encoder_dict["args"]["timesteps"] = f"{tss.nr_predictions}"
                encoder_dict["module"] = "TimeSeries.TsArrayNumericEncoder"
        if "__mdb_ts_previous" in col_name:
            encoder_dict["module"] = "Array.ArrayEncoder"
            encoder_dict["args"]["original_type"] = f'"{tss.target_type}"'
            encoder_dict["args"]["window"] = f"{tss.window}"

    # Set arguments for the encoder
    if encoder_dict["module"] == "Rich_Text.PretrainedLangEncoder" and not is_target:
        encoder_dict["args"]["output_type"] = "$dtype_dict[$target]"

    if eval(encoder_dict["module"].split(".")[1]).is_trainable_encoder:
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
    Given ``TypeInformation``, ``StatisticalAnalysis``, and the ``ProblemDefinition``, generate a JSON config file with the necessary elements of the ML pipeline populated.

    :param TypeInformation: Specifies what data types each column within the dataset are
    :param statistical_analysis:
    :param problem_definition: Specifies details of the model training/building procedure, as defined by ``ProblemDefinition``

    :returns: JSON-AI object with fully populated details of the ML pipeline
    """  # noqaexec
    exec(IMPORTS, globals())
    exec(IMPORT_EXTERNAL_DIRS, globals())
    target = problem_definition.target
    input_cols = []
    for col_name, col_dtype in type_information.dtypes.items():
        if (
            col_name not in type_information.identifiers
            and col_dtype not in (dtype.invalid, dtype.empty)
            and col_name != target
        ):
            input_cols.append(col_name)

    tss = problem_definition.timeseries_settings
    is_target_predicting_encoder = False
    is_ts = problem_definition.timeseries_settings.is_timeseries
    # Single text column classification
    if (
        len(input_cols) == 1
        and type_information.dtypes[input_cols[0]] in (dtype.rich_text)
        and type_information.dtypes[target] in (dtype.categorical, dtype.binary)
    ):
        is_target_predicting_encoder = True

    if is_target_predicting_encoder:
        mixers = [
            {
                "module": "Unit",
                "args": {
                    "target_encoder": "$encoders[self.target]",
                    "stop_after": "$problem_definition.seconds_per_mixer",
                },
            }
        ]
    else:
        mixers = [
            {
                "module": "Neural",
                "args": {
                    "fit_on_dev": True,
                    "stop_after": "$problem_definition.seconds_per_mixer",
                    "search_hyperparameters": True,
                },
            }
        ]

        if not tss.is_timeseries or tss.nr_predictions == 1:
            mixers.extend(
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
                ]
            )
        elif tss.nr_predictions > 1:
            mixers.extend(
                [
                    {
                        "module": "LightGBMArray",
                        "args": {
                            "fit_on_dev": True,
                            "stop_after": "$problem_definition.seconds_per_mixer",
                            "n_ts_predictions": "$problem_definition.timeseries_settings.nr_predictions",
                        },
                    }
                ]
            )

            if tss.use_previous_target:
                mixers.extend(
                    [
                        {
                            "module": "SkTime",
                            "args": {
                                "stop_after": "$problem_definition.seconds_per_mixer",
                                "n_ts_predictions": "$problem_definition.timeseries_settings.nr_predictions",
                            },
                        }
                    ]
                )

    outputs = {
        target: Output(
            data_dtype=type_information.dtypes[target],
            encoder=None,
            mixers=mixers,
            ensemble={
                "module": "BestOf",
                "args": {
                    "args": "$pred_args",
                    "accuracy_functions": "$accuracy_functions",
                    "ts_analysis": "self.ts_analysis" if is_ts else None,
                },
            },
        )
    }

    if tss.is_timeseries and tss.nr_predictions > 1:
        list(outputs.values())[0].data_dtype = dtype.tsarray

    list(outputs.values())[0].encoder = lookup_encoder(
        type_information.dtypes[target],
        target,
        True,
        problem_definition,
        False,
        statistical_analysis,
    )

    features: Dict[str, Feature] = {}
    for col_name in input_cols:
        col_dtype = type_information.dtypes[col_name]
        dependency = []
        encoder = lookup_encoder(
            col_dtype,
            col_name,
            False,
            problem_definition,
            is_target_predicting_encoder,
            statistical_analysis,
        )

        if (
            tss.is_timeseries
            and eval(encoder["module"].split(".")[1]).is_timeseries_encoder
        ):
            if tss.group_by is not None:
                for group in tss.group_by:
                    dependency.append(group)

            if tss.use_previous_target:
                dependency.append(f"__mdb_ts_previous_{target}")

        if len(dependency) > 0:
            feature = Feature(
                encoder=encoder, dependency=dependency, data_dtype=col_dtype
            )
        else:
            feature = Feature(encoder=encoder, data_dtype=col_dtype)
        features[col_name] = feature

    # Decide on the accuracy functions to use
    output_dtype = list(outputs.values())[0].data_dtype
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
    elif output_dtype in (dtype.array, dtype.tsarray):
        accuracy_functions = ["evaluate_array_accuracy"]
    else:
        raise Exception(
            f"Please specify a custom accuracy function for output type {output_dtype}"
        )

    # special dispatch for t+1 time series forecasters
    if is_ts:
        if list(outputs.values())[0].data_dtype in [dtype.integer, dtype.float]:
            accuracy_functions = ["evaluate_array_accuracy"]

    if problem_definition.time_aim is None and (
        problem_definition.seconds_per_mixer is None
        or problem_definition.seconds_per_encoder is None
    ):
        problem_definition.time_aim = (
            1000
            + np.log(statistical_analysis.nr_rows / 10 + 1)
            * np.sum(
                [
                    4
                    if x
                    in [
                        dtype.rich_text,
                        dtype.short_text,
                        dtype.array,
                        dtype.tsarray,
                        dtype.video,
                        dtype.audio,
                        dtype.image,
                    ]
                    else 1
                    for x in type_information.dtypes.values()
                ]
            )
            * 200
        )

    if problem_definition.time_aim is not None:
        nr_trainable_encoders = len(
            [
                x
                for x in features.values()
                if eval(x.encoder["module"].split(".")[1]).is_trainable_encoder
            ]
        )
        nr_mixers = len(list(outputs.values())[0].mixers)
        encoder_time_budget_pct = max(
            3.3 / 5, 1.5 + np.log(nr_trainable_encoders + 1) / 5
        )

        if nr_trainable_encoders == 0:
            problem_definition.seconds_per_encoder = 0
        else:
            problem_definition.seconds_per_encoder = int(
                problem_definition.time_aim
                * (encoder_time_budget_pct / nr_trainable_encoders)
            )
        problem_definition.seconds_per_mixer = int(
            problem_definition.time_aim * ((1 / encoder_time_budget_pct) / nr_mixers)
        )

    return JsonAI(
        cleaner=None,
        splitter=None,
        analyzer=None,
        explainer=None,
        features=features,
        outputs=outputs,
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

    # Add implicit arguments
    # @TODO: Consider removing once we have a proper editor in studio
    mixers = json_ai.outputs[json_ai.problem_definition.target].mixers
    for i in range(len(mixers)):
        if mixers[i]["module"] == "Unit":
            pass
        elif mixers[i]["module"] == "Neural":
            mixers[i]["args"]["target_encoder"] = mixers[i]["args"].get(
                "target_encoder", "$encoders[self.target]"
            )
            mixers[i]["args"]["target"] = mixers[i]["args"].get("target", "$target")
            mixers[i]["args"]["dtype_dict"] = mixers[i]["args"].get(
                "dtype_dict", "$dtype_dict"
            )
            mixers[i]["args"]["timeseries_settings"] = mixers[i]["args"].get(
                "timeseries_settings", "$problem_definition.timeseries_settings"
            )
            mixers[i]["args"]["net"] = mixers[i]["args"].get(
                "net",
                '"DefaultNet"'
                if not tss.is_timeseries or not tss.use_previous_target
                else '"ArNet"',
            )

        elif mixers[i]["module"] == "LightGBM":
            mixers[i]["args"]["target"] = mixers[i]["args"].get("target", "$target")
            mixers[i]["args"]["dtype_dict"] = mixers[i]["args"].get(
                "dtype_dict", "$dtype_dict"
            )
            mixers[i]["args"]["input_cols"] = mixers[i]["args"].get(
                "input_cols", "$input_cols"
            )
        elif mixers[i]["module"] == "Regression":
            mixers[i]["args"]["target"] = mixers[i]["args"].get("target", "$target")
            mixers[i]["args"]["dtype_dict"] = mixers[i]["args"].get(
                "dtype_dict", "$dtype_dict"
            )
            mixers[i]["args"]["target_encoder"] = mixers[i]["args"].get(
                "target_encoder", "$encoders[self.target]"
            )
        elif mixers[i]["module"] == "LightGBMArray":
            mixers[i]["args"]["target"] = mixers[i]["args"].get("target", "$target")
            mixers[i]["args"]["dtype_dict"] = mixers[i]["args"].get(
                "dtype_dict", "$dtype_dict"
            )
            mixers[i]["args"]["input_cols"] = mixers[i]["args"].get(
                "input_cols", "$input_cols"
            )
        elif mixers[i]["module"] == "SkTime":
            mixers[i]["args"]["target"] = mixers[i]["args"].get("target", "$target")
            mixers[i]["args"]["dtype_dict"] = mixers[i]["args"].get(
                "dtype_dict", "$dtype_dict"
            )
            mixers[i]["args"]["ts_analysis"] = mixers[i]["args"].get(
                "ts_analysis", "$ts_analysis"
            )

    ensemble = json_ai.outputs[json_ai.problem_definition.target].ensemble
    ensemble["args"]["target"] = ensemble["args"].get("target", "$target")
    ensemble["args"]["data"] = ensemble["args"].get("data", "encoded_test_data")
    ensemble["args"]["mixers"] = ensemble["args"].get("mixers", "$mixers")

    for name in json_ai.features:
        if json_ai.features[name].dependency is None:
            json_ai.features[name].dependency = []
        if json_ai.features[name].data_dtype is None:
            json_ai.features[name].data_dtype = (
                json_ai.features[name].encoder["module"].split(".")[0].lower()
            )

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
                "timeseries_settings": "$problem_definition.timeseries_settings",
                "anomaly_detection": "$problem_definition.anomaly_detection",
            },
        },
        "splitter": {
            "module": "splitter",
            "args": {
                "tss": "$problem_definition.timeseries_settings",
                "data": "data",
                "seed": 1,
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
                "ts_cfg": "$problem_definition.timeseries_settings",
                "accuracy_functions": "$accuracy_functions",
                "predictor": "$ensemble",
                "data": "encoded_test_data",
                "train_data": "encoded_train_data",
                "target": "$target",
                "dtype_dict": "$dtype_dict",
                "analysis_blocks": "$analysis_blocks",
            },
        },
        "explainer": {
            "module": "explain",
            "args": {
                "timeseries_settings": "$problem_definition.timeseries_settings",
                "positive_domain": "$statistical_analysis.positive_domain",
                "anomaly_detection": "$problem_definition.anomaly_detection",
                "data": "data",
                "encoded_data": "encoded_data",
                "predictions": "df",
                "analysis": "$runtime_analyzer",
                "ts_analysis": "$ts_analysis" if tss.is_timeseries else None,
                "target_name": "$target",
                "target_dtype": "$dtype_dict[self.target]",
                "explainer_blocks": "$analysis_blocks",
                "fixed_confidence": "$pred_args.fixed_confidence",
                "anomaly_error_rate": "$pred_args.anomaly_error_rate",
                "anomaly_cooldown": "$pred_args.anomaly_cooldown",
            },
        },
        "analysis_blocks": [
            {
                "module": "ICP",
                "args": {
                    "fixed_significance": None,
                    "confidence_normalizer": False,
                    "positive_domain": "$statistical_analysis.positive_domain",
                },
            },
            {
                "module": "AccStats",
                "args": {"deps": ["ICP"]},
            },
        ],
        "timeseries_transformer": {
            "module": "transform_timeseries",
            "args": {
                "timeseries_settings": "$problem_definition.timeseries_settings",
                "data": "data",
                "dtype_dict": "$dtype_dict",
                "target": "$target",
                "mode": "$mode",
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

    if len(json_ai.features) < 60:
        hidden_fields["analysis_blocks"].append({
            "module": "GlobalFeatureImportance",
            "args": {
                "disable_column_importance": "False",
            }
        })

    for field_name, implicit_value in hidden_fields.items():
        _populate_implicit_field(json_ai, field_name, implicit_value, tss.is_timeseries)

    return json_ai


def code_from_json_ai(json_ai: JsonAI) -> str:
    """
    Generates a custom ``PredictorInterface`` given the specifications from ``JsonAI`` object.

    :param json_ai: ``JsonAI`` object with fully specified parameters

    :returns: Automated syntax of the ``PredictorInterface`` object.
    """
    # ----------------- #
    # Fill in any missing values
    json_ai = _add_implicit_values(json_ai)

    # ----------------- #
    # Instantiate encoders
    encoder_dict = {
        json_ai.problem_definition.target: call(
            list(json_ai.outputs.values())[0].encoder
        )
    }

    # Instantiate Depedencies
    dependency_dict = {}
    dtype_dict = {
        json_ai.problem_definition.target: f"""'{list(json_ai.outputs.values())[0].data_dtype}'"""
    }

    # Populate features and their data-types
    for col_name, feature in json_ai.features.items():
        encoder_dict[col_name] = call(feature.encoder)
        dependency_dict[col_name] = feature.dependency
        dtype_dict[col_name] = f"""'{feature.data_dtype}'"""

    # Populate time-series specific details
    tss = json_ai.problem_definition.timeseries_settings
    if tss.is_timeseries and tss.use_previous_target:
        col_name = f"__mdb_ts_previous_{json_ai.problem_definition.target}"
        json_ai.problem_definition.timeseries_settings.target_type = list(
            json_ai.outputs.values()
        )[0].data_dtype
        encoder_dict[col_name] = call(
            lookup_encoder(
                list(json_ai.outputs.values())[0].data_dtype,
                col_name,
                False,
                json_ai.problem_definition,
                False,
                None,
            )
        )
        dependency_dict[col_name] = []
        dtype_dict[col_name] = f"""'{list(json_ai.outputs.values())[0].data_dtype}'"""
        json_ai.features[col_name] = Feature(encoder=encoder_dict[col_name])

    # ----------------- #

    input_cols = [x.replace("'", "\\'").replace('"', '\\"') for x in json_ai.features]
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
log.info("Performing statistical analysis on data")
self.statistical_analysis = lightwood.data.statistical_analysis(data,
                                                                self.dtype_dict,
                                                                {json_ai.identifiers},
                                                                self.problem_definition)

# Instantiate post-training evaluation
self.analysis_blocks = [{', '.join([call(block) for block in json_ai.analysis_blocks])}]
    """

    analyze_data_body = align(analyze_data_body, 2)

    # ----------------- #
    # Pre-processing Body
    # ----------------- #

    clean_body = f"""
log.info('Cleaning the data')
data = {call(json_ai.cleaner)}

# Time-series blocks
{ts_transform_code}
"""
    if ts_analyze_code is not None:
        clean_body += f"""
if self.mode != 'predict':
{align(ts_analyze_code,1)}
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

    prepare_body = f"""
self.mode = 'train'

if self.statistical_analysis is None:
    raise Exception("Please run analyze_data first")

# Column to encoder mapping
self.encoders = {inline_dict(encoder_dict)}

# Prepare the training + dev data
concatenated_train_dev = pd.concat([data['train'], data['dev']])

log.info('Preparing the encoders')

encoder_prepping_dict = {{}}

# Prepare encoders that do not require learned strategies
for col_name, encoder in self.encoders.items():
    if not encoder.is_trainable_encoder:
        encoder_prepping_dict[col_name] = [encoder, concatenated_train_dev[col_name], 'prepare']
        log.info(f'Encoder prepping dict length of: {{len(encoder_prepping_dict)}}')

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
    if encoder.is_trainable_encoder:
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
            kwargs['encoded_target_values'] = parallel_prepped_encoders[self.target].encode(priming_data[self.target])

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

log.info('Training the mixers')

# --------------- #
# Fit Models
# --------------- #
# Assign list of mixers
self.mixers = [{', '.join([call(x) for x in list(json_ai.outputs.values())[0].mixers])}]

# Train mixers
trained_mixers = []
for mixer in self.mixers:
    try:
        mixer.fit(encoded_train_data, encoded_dev_data)
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
self.ensemble = {call(list(json_ai.outputs.values())[0].ensemble)}
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
# Extract data
# --------------- #
# Extract the featurized data
encoded_old_data = new_data['old']
encoded_new_data = new_data['new']

# --------------- #
# Adjust (Update) Mixers
# --------------- #
log.info('Updating the mixers')

for mixer in self.mixers:
    mixer.partial_fit(encoded_new_data, encoded_old_data)
"""  # noqa

    adjust_body = align(adjust_body, 2)

    # ----------------- #
    # Learn Body
    # ----------------- #

    learn_body = f"""
self.mode = 'train'

# Perform stats analysis
self.analyze_data(data)

# Pre-process the data
data = self.preprocess(data)

# Create train/test (dev) split
train_dev_test = self.split(data)

# Prepare encoders
self.prepare(train_dev_test)

# Create feature vectors from data
enc_train_test = self.featurize(train_dev_test)

# Prepare mixers
self.fit(enc_train_test)

# Analyze the ensemble
self.analyze_ensemble(enc_train_test)

# ------------------------ #
# Enable model partial fit AFTER it is trained and evaluated for performance with the appropriate train/dev/test splits.
# This assumes the predictor could continuously evolve, hence including reserved testing data may improve predictions.
# SET `json_ai.problem_definition.fit_on_all=False` TO TURN THIS BLOCK OFF.

# Update the mixers with partial fit
if self.problem_definition.fit_on_all:

    log.info("Adjustment on validation requested.")
    update_data = {{"new": enc_train_test["test"], "old": ConcatedEncodedDs([enc_train_test["train"], enc_train_test["dev"]])}}  # noqa

    self.adjust(update_data)

"""
    learn_body = align(learn_body, 2)
    # ----------------- #
    # Predict Body
    # ----------------- #

    predict_body = f"""
# Remove columns that user specifies to ignore
self.mode = 'predict'
log.info(f'Dropping features: {{self.problem_definition.ignore_features}}')
data = data.drop(columns=self.problem_definition.ignore_features, errors='ignore')
for col in self.input_cols:
    if col not in data.columns:
        data[col] = [None] * len(data)

# Pre-process the data
data = self.preprocess(data)

# Featurize the data
encoded_ds = self.featurize({{"predict_data": data}})["predict_data"]
encoded_data = encoded_ds.get_encoded_data(include_target=False)

self.pred_args = PredictionArguments.from_dict(args)
df = self.ensemble(encoded_ds, args=self.pred_args)

if self.pred_args.all_mixers:
    return df
else:
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

        # Any feature-column dependencies
        self.dependencies = {inline_dict(dependency_dict)}

        self.input_cols = [{input_cols}]

        # Initial stats analysis
        self.statistical_analysis = None


    def analyze_data(self, data: pd.DataFrame) -> None:
        # Perform a statistical analysis on the unprocessed data
{analyze_data_body}

    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Preprocess and clean data
{clean_body}

    def split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Split the data into training/testing splits
{split_body}

    def prepare(self, data: Dict[str, pd.DataFrame]) -> None:
        # Prepare encoders to featurize data
{prepare_body}

    def featurize(self, split_data: Dict[str, pd.DataFrame]):
        # Featurize data into numerical representations for models
{feature_body}

    def fit(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        # Fit predictors to estimate target
{fit_body}

    def analyze_ensemble(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        # Evaluate quality of fit for the ensemble of mixers
{analyze_ensemble}

    def learn(self, data: pd.DataFrame) -> None:
        log.info(f'Dropping features: {{self.problem_definition.ignore_features}}')
        data = data.drop(columns=self.problem_definition.ignore_features, errors='ignore')
{learn_body}

    def adjust(self, new_data: Dict[str, pd.DataFrame]) -> None:
        # Update mixers with new information
{adjust_body}

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
