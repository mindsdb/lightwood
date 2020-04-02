import numpy as np

class ImportanceDeterminator:
    def __init__(self):
        self.alone_loss_history_map = {}
        self.dropout_loss_history_map = {}
        self.normal_loss_history_arr = []

    def register_observation(mode, loss, column=None):
        if mode == 'normal':
            self.normal_loss_history_arr.append(loss)
            return

        if mode == 'alone':
            map = self.alone_loss_history_map
        elif mode == 'dropout':
            map = self.dropout_loss_history_map

        if column not in map:
            map[column] = []

        map[column].append(loss)

    def determine_importance(columns='all', n_observations=10):
        if columns == 'all':
            columns = list(self.dropout_loss_history_map.keys())

        importance_dict = {}
        for col in columns:
            # Note, since we dropout a different column every time, we take the mean over nr_columns * nr_observations for the normal loss
            all_loss_mean = np.mean(self.normal_loss_history_arr[-n_observations*len(self.dropout_loss_history_map.keys())]:)

            dropout_loss_mean = np.mean(self.dropout_loss_history_map[col][-n_observations:])
            alone_loss_mean = np.mean(self.dropout_loss_history_map[col][-n_observations:])

            # Dropout loss has positive correlation with importance
            # Alone loss has negative correlation with importance
            importance_dict[col] = dropout_loss_mean/all_loss_mean + 1/(alone_loss_mean/all_loss_mean)

        # Normalize to get values from 0 to 1 (Maybe find a better way to do this in the future or drop entirely ?)
        for col in importance_dict[col]:
            importance_dict[col] = importance_dict[col]/(np.max(importance_dict.values()))

        return importance_dict
