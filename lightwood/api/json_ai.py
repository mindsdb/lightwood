# TODO: add_implicit_values unit test ensures NO changes for a fully specified file.
import inspect

from type_infer.dtype import dtype
from type_infer.base import TypeInformation
from dataprep_ml import StatisticalAnalysis

from lightwood.helpers.templating import _consolidate_analysis_blocks, _add_cls_kwarg
from lightwood.helpers.constants import IMPORTS, IMPORT_EXTERNAL_DIRS
from lightwood.api.types import (
    JsonAI,
    ProblemDefinition,
)
import lightwood.ensemble
import lightwood.encoder


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
        or len(statistical_analysis.histograms[col_name]['x']) > 16
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

            if problem_defintion.target_weights is not None:
                encoder_dict["args"][
                    "target_weights"
                ] = problem_defintion.target_weights

    # Time-series representations require more advanced flags
    if tss.is_timeseries:
        gby = tss.group_by if tss.group_by is not None else []

        if tss.order_by in gby:
            raise Exception('The `order_by` column cannot be used to `group_by` simultaneously!')

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

        # add neural model
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
                    },
                ]
            )

        # add other models
        if (not tss.is_timeseries or tss.horizon == 1) and dtype_dict[target] not in (dtype.num_array, dtype.cat_array):
            submodels.extend(
                [
                    {
                        "module": "XGBoostMixer",
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

        # special forecasting dispatch
        elif tss.is_timeseries:
            submodels.extend([
                {
                    "module": "XGBoostArrayMixer",
                    "args": {},
                },
            ])

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


def add_implicit_values(json_ai: JsonAI) -> JsonAI:
    """
    To enable brevity in writing, auto-generate the "unspecified/missing" details required in the ML pipeline.

    :params: json_ai: ``JsonAI`` object that describes the ML pipeline that may not have every detail fully specified.

    :returns: ``JSONAI`` object with all necessary parameters that were previously left unmentioned filled in.
    """
    problem_definition = json_ai.problem_definition
    tss = problem_definition.timeseries_settings
    is_ts = tss.is_timeseries
    mixers = json_ai.model['args']['submodels']

    # Add implicit ensemble arguments
    param_pairs = {
        'target': json_ai.model["args"].get("target", "$target"),
        'data': json_ai.model["args"].get("data", "encoded_test_data"),
        'mixers': json_ai.model["args"].get("mixers", "$mixers"),
        'fit': json_ai.model["args"].get("fit", True),
        'args': json_ai.model["args"].get("args", "$pred_args"),
        'accuracy_functions': json_ai.model["args"].get("accuracy_functions", "$accuracy_functions"),
        'ts_analysis': json_ai.model["args"].get("ts_analysis", "self.ts_analysis" if is_ts else None),
        'dtype_dict': json_ai.model["args"].get("dtype_dict", "$dtype_dict"),
    }
    ensemble_cls = getattr(lightwood.ensemble, json_ai.model["module"])
    filtered_params = {}
    for p_name, p_value in param_pairs.items():
        _add_cls_kwarg(ensemble_cls, filtered_params, p_name, p_value)

    json_ai.model["args"] = filtered_params
    json_ai.model["args"]['submodels'] = mixers  # add mixers back in

    # Add implicit mixer arguments
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
        if mixers[i]["module"] in ("Neural", "NeuralTs", "TabTransformerMixer"):
            mixers[i]["args"]["target_encoder"] = mixers[i]["args"].get(
                "target_encoder", "$encoders[self.target]"
            )

            if mixers[i]["module"] in ("Neural", "NeuralTs"):
                mixers[i]["args"]["net"] = mixers[i]["args"].get(
                    "net",
                    '"DefaultNet"'
                    if not tss.is_timeseries or not tss.use_previous_target
                    else '"ArNet"',
                )
                mixers[i]["args"]["search_hyperparameters"] = mixers[i]["args"].get("search_hyperparameters", True)
                mixers[i]["args"]["fit_on_dev"] = mixers[i]["args"].get("fit_on_dev", True)

            if mixers[i]["module"] == "NeuralTs":
                mixers[i]["args"]["timeseries_settings"] = mixers[i]["args"].get(
                    "timeseries_settings", "$problem_definition.timeseries_settings"
                )
                mixers[i]["args"]["ts_analysis"] = mixers[i]["args"].get("ts_analysis", "$ts_analysis")

            if mixers[i]["module"] == "TabTransformerMixer":
                mixers[i]["args"]["search_hyperparameters"] = mixers[i]["args"].get("search_hyperparameters", False)
                mixers[i]["args"]["fit_on_dev"] = mixers[i]["args"].get("fit_on_dev", False)

        elif mixers[i]["module"] in ("LightGBM", "XGBoostMixer"):
            mixers[i]["args"]["input_cols"] = mixers[i]["args"].get(
                "input_cols", "$input_cols"
            )
            mixers[i]["args"]["target_encoder"] = mixers[i]["args"].get(
                "target_encoder", "$encoders[self.target]"
            )
            mixers[i]["args"]["fit_on_dev"] = mixers[i]["args"].get(
                "fit_on_dev", True
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

        elif mixers[i]["module"] in ("LightGBMArray", "XGBoostArrayMixer"):
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

    # encoder checks
    for name in json_ai.encoders:
        if name not in json_ai.dependency_dict:
            json_ai.dependency_dict[name] = []

    # filter arguments for included encoders (custom encoders will skip the check)
    for col, enc_dict in json_ai.encoders.items():
        filtered_kwargs = {}
        if hasattr(lightwood.encoder, enc_dict['module']):
            encoder_cls = getattr(lightwood.encoder, enc_dict['module'])
            for k, v in enc_dict['args'].items():
                _add_cls_kwarg(encoder_cls, filtered_kwargs, k, v)
            json_ai.encoders[col]['args'] = filtered_kwargs

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
                "pdef": "$problem_definition",
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
                "pred_args": "$pred_args",
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
