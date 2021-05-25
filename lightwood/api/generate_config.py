from lightwood.api import LightwoodConfig, TypeInformation, StatisticalAnalysis


def chose_encoder(self, dtype, is_target):
    config = {}

    config['input_features'] = []
    config['output_features'] = []

    for col_name in self.transaction.lmd['columns']:
        if col_name in self.transaction.lmd['columns_to_ignore'] or col_name not in self.transaction.lmd['stats_v2']:
            continue

        col_stats = self.transaction.lmd['stats_v2'][col_name]
        data_subtype = col_stats['typing']['data_subtype']
        data_type = col_stats['typing']['data_type']

        other_keys = {'encoder_attrs': {}}
        if data_type == DATA_TYPES.NUMERIC:
            lightwood_data_type = ColumnDataTypes.NUMERIC
            if col_name in self.transaction.lmd['predict_columns'] and col_stats.get('positive_domain', False):
                other_keys['encoder_attrs']['positive_domain'] = True

        elif data_type == DATA_TYPES.CATEGORICAL:
            predict_proba = self.transaction.lmd['output_class_distribution']
            if col_name in self.transaction.lmd['predict_columns'] and predict_proba:
                other_keys['encoder_attrs']['predict_proba'] = predict_proba

            if data_subtype == DATA_SUBTYPES.TAGS:
                lightwood_data_type = ColumnDataTypes.MULTIPLE_CATEGORICAL
                if predict_proba:
                    self.transaction.log.warning(f'Class distribution not supported for tags prediction\n')
            else:
                lightwood_data_type = ColumnDataTypes.CATEGORICAL

        elif data_subtype in (DATA_SUBTYPES.TIMESTAMP, DATA_SUBTYPES.DATE):
            lightwood_data_type = ColumnDataTypes.DATETIME

        elif data_subtype == DATA_SUBTYPES.IMAGE:
            lightwood_data_type = ColumnDataTypes.IMAGE
            other_keys['encoder_attrs']['aim'] = 'balance'

        elif data_subtype == DATA_SUBTYPES.AUDIO:
            lightwood_data_type = ColumnDataTypes.AUDIO

        elif data_subtype == DATA_SUBTYPES.RICH:
            lightwood_data_type = ColumnDataTypes.TEXT

        elif data_subtype == DATA_SUBTYPES.SHORT:
            lightwood_data_type = ColumnDataTypes.SHORT_TEXT

        elif data_subtype == DATA_SUBTYPES.ARRAY:
            lightwood_data_type = ColumnDataTypes.TIME_SERIES

        else:
            err = f'The lightwood model backend is unable to handle data of type {data_type} and subtype {data_subtype} !'
            self.transaction.log.error(err)
            raise Exception(err)

        if self.transaction.lmd['tss']['is_timeseries'] and col_name in self.transaction.lmd['tss']['order_by']:
            lightwood_data_type = ColumnDataTypes.TIME_SERIES

        grouped_by = self.transaction.lmd['tss'].get('group_by', [])
        col_config = {
            'name': col_name,
            'type': lightwood_data_type,
            'grouped_by': col_name in grouped_by if grouped_by else False
        }

        if data_subtype == DATA_SUBTYPES.SHORT:
            col_config['encoder_class'] = lightwood.encoders.text.short.ShortTextEncoder

        if col_name in self.transaction.lmd['weight_map']:
            col_config['weights'] = self.transaction.lmd['weight_map'][col_name]

        if col_name in secondary_type_dict:
            col_config['secondary_type'] = secondary_type_dict[col_name]

        col_config.update(other_keys)

        if col_name in self.transaction.lmd['predict_columns']:
            if not ((data_type == DATA_TYPES.CATEGORICAL and data_subtype != DATA_SUBTYPES.TAGS) or data_type == DATA_TYPES.NUMERIC):
                self.nn_mixer_only = True

            if self.transaction.lmd['tss']['is_timeseries']:
                col_config['additional_info'] = {
                    'nr_predictions': self.transaction.lmd['tss']['nr_predictions'],
                    'time_series_target': True,
                }
            config['output_features'].append(col_config)

            if self.transaction.lmd['tss']['is_timeseries'] and self.transaction.lmd['tss']['use_previous_target']:
                p_col_config = copy.deepcopy(col_config)
                p_col_config['name'] = f"__mdb_ts_previous_{p_col_config['name']}"
                p_col_config['original_type'] = col_config['type']
                p_col_config['type'] = ColumnDataTypes.TIME_SERIES

                if 'secondary_type' in col_config:
                    p_col_config['secondary_type'] = col_config['secondary_type']

                config['input_features'].append(p_col_config)

            if self.nr_predictions > 1:
                self.transaction.lmd['stats_v2'][col_name]['typing']['data_subtype'] = DATA_SUBTYPES.ARRAY
                self.transaction.lmd['stats_v2'][col_name]['typing']['data_type'] = DATA_TYPES.SEQUENTIAL
                for timestep_index in range(1,self.nr_predictions):
                    additional_target_config = copy.deepcopy(col_config)
                    additional_target_config['name'] = f'{col_name}_timestep_{timestep_index}'
                    config['output_features'].append(additional_target_config)
        else:
            if col_name in self.transaction.lmd['tss']['historical_columns']:
                if 'secondary_type' in col_config:
                    col_config['secondary_type'] = col_config['secondary_type']
                col_config['original_type'] = col_config['type']
                col_config['type'] = ColumnDataTypes.TIME_SERIES

            config['input_features'].append(col_config)

    config['data_source'] = {}
    config['data_source']['cache_transformed_data'] = not self.transaction.lmd['force_disable_cache']

    config['mixer'] = {
        'class': lightwood.mixers.NnMixer,
        'kwargs': {
            'selfaware': self.transaction.lmd['use_selfaware_model']
        }
    }

    return config

def generate_config(target: str, type_information: TypeInformation, statistical_analysis: StatisticalAnalysis) -> LightwoodConfig:
    pass
