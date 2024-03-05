import os
import sys
import time
import string
import random
import tempfile
import importlib

from copy import deepcopy
from types import ModuleType

from type_infer.dtype import dtype

from lightwood.helpers.log import log
from lightwood.api.types import JsonAI
from lightwood.api.json_ai import add_implicit_values, lookup_encoder
from lightwood.helpers.constants import IMPORTS, IMPORT_EXTERNAL_DIRS
from lightwood.helpers.templating import call, inline_dict, align
from lightwood.__about__ import __version__ as lightwood_version


def code_from_json_ai(json_ai: JsonAI) -> str:
    """
    Generates a custom ``PredictorInterface`` given the specifications from ``JsonAI`` object.

    :param json_ai: ``JsonAI`` object with fully specified parameters

    :returns: Automated syntax of the ``PredictorInterface`` object.
    """
    json_ai = deepcopy(json_ai)
    # ----------------- #
    # Fill in any missing values
    json_ai = add_implicit_values(json_ai)

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
    if len(input_cols) < 1:
        raise Exception('There are no valid input features. Please check your data before trying again.')
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

prepped_encoders = {{}}

# Prepare input encoders
parallel_encoding = parallel_encoding_check(data['train'], self.encoders)

if parallel_encoding:
    log.debug('Preparing in parallel...')
    for col_name, encoder in self.encoders.items():
        if col_name != self.target and not encoder.is_trainable_encoder:
            prepped_encoders[col_name] = (encoder, concatenated_train_dev[col_name], 'prepare')
    prepped_encoders = mut_method_call(prepped_encoders)

else:
    log.debug('Preparing sequentially...')
    for col_name, encoder in self.encoders.items():
        if col_name != self.target and not encoder.is_trainable_encoder:
            log.debug(f'Preparing encoder for {{col_name}}...')
            encoder.prepare(concatenated_train_dev[col_name])
            prepped_encoders[col_name] = encoder

# Store encoders
for col_name, encoder in prepped_encoders.items():
    self.encoders[col_name] = encoder

# Prepare the target
if self.target not in prepped_encoders:
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

tss = self.problem_definition.timeseries_settings

feature_data = dict()
for key, data in split_data.items():
    if key != 'stratified_on':

        # compute and store two splits - full and filtered (useful for time series post-train analysis)
        if key not in self.feature_cache:
            featurized_split = EncodedDs(self.encoders, data, self.target)
            filtered_subset = EncodedDs(self.encoders, filter_ts(data, tss), self.target)

            for k, s in zip((key, f'{{key}}_filtered'), (featurized_split, filtered_subset)):
                self.feature_cache[k] = s

        for k in (key, f'{{key}}_filtered'):
            feature_data[k] = self.feature_cache[k]

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
encoded_test_data = enc_data['test_filtered']

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
        if mixer.trains_once:
            self.fit_mixer(mixer,
                           ConcatedEncodedDs([encoded_train_data, encoded_dev_data]),
                           encoded_test_data)
        else:
            self.fit_mixer(mixer, encoded_train_data, encoded_dev_data)
        trained_mixers.append(mixer)
    except Exception as e:
        log.warning(f'Exception: {{e}} when training mixer: {{mixer}}')
        if {json_ai.problem_definition.strict_mode} and mixer.stable:
            raise e

# Update mixers to trained versions
if not trained_mixers:
    raise Exception('No mixers could be trained! Please verify your problem definition or JsonAI model representation.')
self.mixers = trained_mixers

# --------------- #
# Create Ensembles
# --------------- #
log.info('Ensembling the mixer')
# Create an ensemble of mixers to identify best performing model
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
if not self.problem_definition.embedding_only:
    self.fit(enc_train_test)
else:
    self.mixers = []
    self.ensemble = Embedder(self.target, mixers=list(), data=enc_train_test['train'])
    self.supports_proba = self.ensemble.supports_proba

# Analyze the ensemble
log.info(f'[Learn phase 7/{n_phases}] - Ensemble analysis')
self.analyze_ensemble(enc_train_test)

# ------------------------ #
# Enable model partial fit AFTER it is trained and evaluated for performance with the appropriate train/dev/test splits.
# This assumes the predictor could continuously evolve, hence including reserved testing data may improve predictions.
# SET `json_ai.problem_definition.fit_on_all=False` TO TURN THIS BLOCK OFF.

# Update the mixers with partial fit
if self.problem_definition.fit_on_all and all([not m.trains_once for m in self.mixers]):
    log.info(f'[Learn phase 8/{n_phases}] - Adjustment on validation requested')
    self.adjust(enc_train_test["test"].data_frame, ConcatedEncodedDs([enc_train_test["train"],
                                                                      enc_train_test["dev"]]).data_frame,
                                                                      adjust_args={'learn_call': True})

self.feature_cache = dict()  # empty feature cache to avoid large predictor objects
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

self.pred_args = PredictionArguments.from_dict(args)

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

@timed
def _timed_call(encoded_ds):
    if self.pred_args.return_embedding:
        embedder = Embedder(self.target, mixers=list(), data=encoded_ds)
        df = embedder(encoded_ds, args=self.pred_args)
    else:
        df = self.ensemble(encoded_ds, args=self.pred_args)
    return df

df = _timed_call(encoded_ds)

if not(any(
            [self.pred_args.all_mixers,
             self.pred_args.return_embedding,
             self.problem_definition.embedding_only]
        )):
    log.info(f'[Predict phase 4/{{n_phases}}] - Analyzing output')
    df, global_insights = {call(json_ai.explainer)}
    self.global_insights = {{**self.global_insights, **global_insights}}

self.feature_cache = dict()  # empty feature cache to avoid large predictor objects

return df
"""

    predict_body = align(predict_body, 2)

    # ----------------- #
    # Test Body
    # ----------------- #
    test_body = """
preds = self.predict(data, args)
preds = preds.rename(columns={'prediction': self.target})
filtered = []

# filter metrics if not supported
for metric in metrics:
    # metric should be one of: an actual function, registered in the model class, or supported by the evaluator
    if not (callable(metric) or metric in self.accuracy_functions or metric in mdb_eval_accuracy_metrics):
        if strict:
            raise Exception(f'Invalid metric: {metric}')
        else:
            log.warning(f'Invalid metric: {metric}. Skipping...')
    else:
        filtered.append(metric)

metrics = filtered
try:
    labels = self.model_analysis.histograms[self.target]['x']
except:
    if strict:
        raise Exception('Label histogram not found')
    else:
        label_map = None  # some accuracy functions will crash without this, be mindful
scores = evaluate_accuracies(
                data,
                preds[self.target],
                self.target,
                metrics,
                ts_analysis=self.ts_analysis,
                labels=labels
            )

# TODO: remove once mdb_eval returns an actual list
scores = {k: [v] for k, v in scores.items() if not isinstance(v, list)}

return pd.DataFrame.from_records(scores)  # TODO: add logic to disaggregate per-mixer
"""

    test_body = align(test_body, 2)

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
        self.pred_args = PredictionArguments()

        # Any feature-column dependencies
        self.dependencies = {inline_dict(json_ai.dependency_dict)}

        self.input_cols = [{input_cols}]

        # Initial stats analysis
        self.statistical_analysis = None
        self.ts_analysis = None
        self.runtime_log = dict()
        self.global_insights = dict()

        # Feature cache
        self.feature_cache = dict()

    @timed_predictor
    def analyze_data(self, data: pd.DataFrame) -> None:
        # Perform a statistical analysis on the unprocessed data
{analyze_data_body}

    @timed_predictor
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Preprocess and clean data
{clean_body}

    @timed_predictor
    def split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Split the data into training/testing splits
{split_body}

    @timed_predictor
    def prepare(self, data: Dict[str, pd.DataFrame]) -> None:
        # Prepare encoders to featurize data
{prepare_body}

    @timed_predictor
    def featurize(self, split_data: Dict[str, pd.DataFrame]):
        # Featurize data into numerical representations for models
{feature_body}

    @timed_predictor
    def fit(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        # Fit predictors to estimate target
{fit_body}

    @timed_predictor
    def fit_mixer(self, mixer, encoded_train_data, encoded_dev_data) -> None:
        mixer.fit(encoded_train_data, encoded_dev_data)

    @timed_predictor
    def analyze_ensemble(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        # Evaluate quality of fit for the ensemble of mixers
{analyze_ensemble}

    @timed_predictor
    def learn(self, data: pd.DataFrame) -> None:
        if self.problem_definition.ignore_features:
            log.info(f'Dropping features: {{self.problem_definition.ignore_features}}')
            data = data.drop(columns=self.problem_definition.ignore_features, errors='ignore')
{learn_body}

    @timed_predictor
    def adjust(self, train_data: Union[EncodedDs, ConcatedEncodedDs, pd.DataFrame],
        dev_data: Optional[Union[EncodedDs, ConcatedEncodedDs, pd.DataFrame]] = None,
        adjust_args: Optional[dict] = None) -> None:
        # Update mixers with new information
{adjust_body}

    @timed_predictor
    def predict(self, data: pd.DataFrame, args: Dict = {{}}) -> pd.DataFrame:
{predict_body}

    def test(
        self, data: pd.DataFrame, metrics: list, args: Dict[str, object] = {{}}, strict: bool = False
        ) -> pd.DataFrame:
{test_body}
"""

    try:
        import black
    except Exception:
        black = None

    if black is not None:
        try:
            formatted_predictor_code = black.format_str(predictor_code, mode=black.FileMode())

            if type(_predictor_from_code(formatted_predictor_code)).__name__ == 'Predictor':
                predictor_code = formatted_predictor_code
            else:
                log.info('Black formatter output is invalid, predictor code might be a bit ugly')

        except Exception:
            log.info('Black formatter failed to run, predictor code might be a bit ugly')
    else:
        log.info('Unable to import black formatter, predictor code might be a bit ugly.')

    return predictor_code


def _module_from_code(code: str, module_name: str) -> ModuleType:
    """
    Create a python module (containing the generated ``Predictor`` class) from the code. This is both a python object and an associated temporary file on your filesystem

    :param code: The ``Predictor``'s code in text form
    :param module_name: The name of the newly created module

    :returns: A python module object
    """ # noqa
    dirname = tempfile.gettempdir()
    filename = os.urandom(24).hex() + str(time.time()).replace('.', '') + '.py'
    path = os.path.join(dirname, filename)
    if 'LIGHTWOOD_DEV_SAVE_TO' in os.environ:
        path = os.environ['LIGHTWOOD_DEV_SAVE_TO']

    with open(path, 'wb') as fp:
        fp.write(code.encode('utf-8'))
        spec = importlib.util.spec_from_file_location(module_name, fp.name)
        temp_module = importlib.util.module_from_spec(spec)
        sys.modules[module_name] = temp_module
        spec.loader.exec_module(temp_module)

    return temp_module


def _predictor_from_code(code: str):
    """
    :param code: The ``Predictor``'s code in text form

    :returns: A lightwood ``Predictor`` object
    """
    module_name = ''.join(random.choices(string.ascii_uppercase + string.digits, k=12))
    module_name += str(time.time()).replace('.', '')
    predictor = _module_from_code(code, module_name).Predictor()
    return predictor
