

import lightwood
from lightwood import __version__ as lightwood_version
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
from typing import Dict, List, Union
import os
from types import ModuleType
import importlib.machinery
import sys
import time


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


class Predictor(PredictorInterface):
    target: str
    mixers: List[BaseMixer]
    encoders: Dict[str, BaseEncoder]
    ensemble: BaseEnsemble
    mode: str

    def __init__(self):
        seed(420)
        self.target = 'target'
        self.mode = 'inactive'
        self.problem_definition = ProblemDefinition.from_dict({'target': 'target', 'pct_invalid': 2, 'unbias_target': True, 'seconds_per_mixer': 28512.0, 'seconds_per_encoder': 85536.0, 'expected_additional_time': 15.752806663513184, 'time_aim': 259200, 'target_weights': None, 'positive_domain': False, 'timeseries_settings': {'is_timeseries': False, 'order_by': None, 'window': None, 'group_by': None, 'use_previous_target': True, 'horizon': None, 'historical_columns': None, 'target_type': '', 'allow_incomplete_history': True, 'eval_cold_start': True, 'interval_periods': []}, 'anomaly_detection': False, 'use_default_analysis': True, 'ignore_features': ['url_legal', 'license', 'standard_error'], 'fit_on_all': True, 'strict_mode': True, 'seed_nr': 420})
        self.accuracy_functions = ['r2_score']
        self.identifiers = {'id': 'Hash-like identifier'}
        self.dtype_dict = {
'excerpt': 'rich_text',
'target': 'float'
}

        # Any feature-column dependencies
        self.dependencies = {
'target': [],
'excerpt': []
}

        self.input_cols = ['excerpt']

        # Initial stats analysis
        self.statistical_analysis = None
        self.runtime_log = dict()

    @timed
    def analyze_data(self, data: pd.DataFrame) -> None:
        # Perform a statistical analysis on the unprocessed data

        log.info("Performing statistical analysis on data")
        self.statistical_analysis = lightwood.data.statistical_analysis(data,
                                                                        self.dtype_dict,
                                                                        {'id': 'Hash-like identifier'},
                                                                        self.problem_definition)
        
        # Instantiate post-training evaluation
        self.analysis_blocks = [ICP(fixed_significance=None,confidence_normalizer=False,positive_domain=self.statistical_analysis.positive_domain), AccStats(deps=['ICP']), ConfStats(deps=['ICP'])]
            

    @timed
    def preprocess(self, data: pd.DataFrame) -> pd.DataFrame:
        # Preprocess and clean data

        log.info('Cleaning the data')
        data = MyCustomCleaner.cleaner(data=data,identifiers=self.identifiers,dtype_dict=self.dtype_dict,target=self.target,mode=self.mode,timeseries_settings=self.problem_definition.timeseries_settings,anomaly_detection=self.problem_definition.anomaly_detection)
        
        # Time-series blocks
        
        
        return data

    @timed
    def split(self, data: pd.DataFrame) -> Dict[str, pd.DataFrame]:
        # Split the data into training/testing splits

        log.info("Splitting the data into train/test")
        train_test_data = splitter(data=data,seed=1,pct_train=0.8,pct_dev=0.1,pct_test=0.1,tss=self.problem_definition.timeseries_settings,target=self.target,dtype_dict=self.dtype_dict)
        
        return train_test_data
            

    @timed
    def prepare(self, data: Dict[str, pd.DataFrame]) -> None:
        # Prepare encoders to featurize data

        self.mode = 'train'
        
        if self.statistical_analysis is None:
            raise Exception("Please run analyze_data first")
        
        # Column to encoder mapping
        self.encoders = {
        'target': NumericEncoder(is_target=True,positive_domain=self.statistical_analysis.positive_domain),
        'excerpt': PretrainedLangEncoder(output_type=self.dtype_dict[self.target],stop_after=self.problem_definition.seconds_per_encoder)
        }
        
        # Prepare the training + dev data
        concatenated_train_dev = pd.concat([data['train'], data['dev']])
        
        log.info('Preparing the encoders')
        
        encoder_prepping_dict = {}
        
        # Prepare encoders that do not require learned strategies
        for col_name, encoder in self.encoders.items():
            if not encoder.is_trainable_encoder:
                encoder_prepping_dict[col_name] = [encoder, concatenated_train_dev[col_name], 'prepare']
                log.info(f'Encoder prepping dict length of: {len(encoder_prepping_dict)}')
        
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
                kwargs = {}
                if self.dependencies[col_name]:
                    kwargs['dependency_data'] = {}
                    for col in self.dependencies[col_name]:
                        kwargs['dependency_data'][col] = {
                            'original_type': self.dtype_dict[col],
                            'data': priming_data[col]
                        }
                    
        
                # If an encoder representation requires the target, provide priming data
                if hasattr(encoder, 'uses_target'):
                    kwargs['encoded_target_values'] = parallel_prepped_encoders[self.target].encode(priming_data[self.target])
        
                encoder.prepare(data['train'][col_name], data['dev'][col_name], **kwargs)
        
            
        

    @timed
    def featurize(self, split_data: Dict[str, pd.DataFrame]):
        # Featurize data into numerical representations for models

        log.info('Featurizing the data')
        
        feature_data = { key: EncodedDs(self.encoders, data, self.target) for key, data in split_data.items() if key != "stratified_on"}
        
        return feature_data
        
        

    @timed
    def fit(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        # Fit predictors to estimate target

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
        self.mixers = [Neural(fit_on_dev=True,search_hyperparameters=True,net="DefaultNet",stop_after=self.problem_definition.seconds_per_mixer,target_encoder=self.encoders[self.target],target=self.target,dtype_dict=self.dtype_dict,timeseries_settings=self.problem_definition.timeseries_settings), LightGBM(fit_on_dev=True,use_optuna=True,stop_after=self.problem_definition.seconds_per_mixer,target=self.target,dtype_dict=self.dtype_dict,input_cols=self.input_cols,target_encoder=self.encoders[self.target]), Regression(stop_after=self.problem_definition.seconds_per_mixer,target=self.target,dtype_dict=self.dtype_dict,target_encoder=self.encoders[self.target])]
        
        # Train mixers
        trained_mixers = []
        for mixer in self.mixers:
            try:
                mixer.fit(encoded_train_data, encoded_dev_data)
                trained_mixers.append(mixer)
            except Exception as e:
                log.warning(f'Exception: {e} when training mixer: {mixer}')
                if True and mixer.stable:
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
        self.ensemble = BestOf(ts_analysis=None,data=encoded_test_data,args=self.pred_args,accuracy_functions=self.accuracy_functions,target=self.target,mixers=self.mixers)
        self.supports_proba = self.ensemble.supports_proba
        

    @timed
    def analyze_ensemble(self, enc_data: Dict[str, pd.DataFrame]) -> None:
        # Evaluate quality of fit for the ensemble of mixers

        
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
        self.model_analysis, self.runtime_analyzer = model_analyzer(data=encoded_test_data,train_data=encoded_train_data,stats_info=self.statistical_analysis,tss=self.problem_definition.timeseries_settings,accuracy_functions=self.accuracy_functions,predictor=self.ensemble,target=self.target,dtype_dict=self.dtype_dict,analysis_blocks=self.analysis_blocks)
        

    @timed
    def learn(self, data: pd.DataFrame) -> None:
        log.info(f'Dropping features: {self.problem_definition.ignore_features}')
        data = data.drop(columns=self.problem_definition.ignore_features, errors='ignore')

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
            self.adjust(enc_train_test["test"], ConcatedEncodedDs([enc_train_test["train"], enc_train_test["dev"]]))
        
        

    @timed
    def adjust(self, new_data: Union[EncodedDs, ConcatedEncodedDs, pd.DataFrame],
        old_data: Optional[Union[EncodedDs, ConcatedEncodedDs, pd.DataFrame]] = None) -> None:
        # Update mixers with new information

        self.mode = 'train'
        
        # --------------- #
        # Prepare data
        # --------------- #
        if old_data is None:
            old_data = pd.DataFrame()
        
        if isinstance(old_data, pd.DataFrame):
            old_data = EncodedDs(self.encoders, old_data, self.target)
        
        if isinstance(new_data, pd.DataFrame):
            new_data = EncodedDs(self.encoders, new_data, self.target)
        
        # --------------- #
        # Update/Adjust Mixers
        # --------------- #
        log.info('Updating the mixers')
        
        for mixer in self.mixers:
            mixer.partial_fit(new_data, old_data)
        

    @timed
    def predict(self, data: pd.DataFrame, args: Dict = {}) -> pd.DataFrame:

        self.mode = 'predict'
        
        if len(data) == 0:
            raise Exception("Empty input, aborting prediction. Please try again with some input data.")
        
        # Remove columns that user specifies to ignore
        log.info(f'Dropping features: {self.problem_definition.ignore_features}')
        data = data.drop(columns=self.problem_definition.ignore_features, errors='ignore')
        for col in self.input_cols:
            if col not in data.columns:
                data[col] = [None] * len(data)
        
        # Pre-process the data
        data = self.preprocess(data)
        
        # Featurize the data
        encoded_ds = self.featurize({"predict_data": data})["predict_data"]
        encoded_data = encoded_ds.get_encoded_data(include_target=False)
        
        self.pred_args = PredictionArguments.from_dict(args)
        df = self.ensemble(encoded_ds, args=self.pred_args)
        
        if self.pred_args.all_mixers:
            return df
        else:
            insights, global_insights = explain(data=data,encoded_data=encoded_data,predictions=df,ts_analysis=None,timeseries_settings=self.problem_definition.timeseries_settings,positive_domain=self.statistical_analysis.positive_domain,anomaly_detection=self.problem_definition.anomaly_detection,analysis=self.runtime_analyzer,target_name=self.target,target_dtype=self.dtype_dict[self.target],explainer_blocks=self.analysis_blocks,pred_args=self.pred_args)
            return insights
        
