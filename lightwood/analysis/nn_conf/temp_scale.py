from copy import deepcopy
from typing import Dict, Tuple
from types import SimpleNamespace

import torch
import pandas as pd
from torch import nn, optim
from sklearn.preprocessing import OrdinalEncoder

from lightwood.helpers.log import log
from lightwood.analysis.base import BaseAnalysisBlock


class TempScaler(BaseAnalysisBlock):
    """
    Original reference (MIT Licensed): https://github.com/gpleiss/temperature_scaling
    NB: Output of the neural network should be the classification logits, NOT the softmax (or log softmax)! TODO
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.ordenc = OrdinalEncoder()
        self._softmax = torch.nn.Softmax(dim=1)
        self.active = False

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))
        return logits / temperature

    def softmax(self, logits):
        return self._softmax(self.temperature_scale(logits))

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        """
        Tune and set the temperature of a neural model optimizing NLL using validation set logits.
        """
        ns = SimpleNamespace(**kwargs)

        if ns.predictor.mixers[ns.predictor.indexes_by_accuracy[0]].supports_proba:
            self.n_cls = len(ns.stats_info.train_observed_classes)
            nll_criterion = nn.CrossEntropyLoss()
            self.ordenc.fit([[val] for val in ns.stats_info.train_observed_classes])

            # collect logits and labels for the validation set
            logits_list = []
            labels_list = []
            with torch.no_grad():
                prob_cols = [col for col in ns.normal_predictions.columns
                             if '__mdb_proba' in col and
                             '__mdb_unknown_cat' not in col]
                if not prob_cols:
                    return info  # early stop if no proba info is available
                for logits, label in zip(ns.normal_predictions[prob_cols].values, ns.data[ns.target]):
                    logits_list.append(logits.tolist())
                    labels_list.append(int(self.ordenc.transform([[label]]).flatten()[0]))
                logits = torch.tensor(logits_list)
                labels = torch.tensor(labels_list).long()

            # NLL and ECE before temp scaling
            before_temperature_nll = nll_criterion(logits, labels).item()
            log.info(f'Before calibration - NLL: {round(before_temperature_nll, 3)}')

            # optimize w.r.t. NLL
            optimizer = optim.LBFGS([self.temperature], lr=0.001, max_iter=1000)

            def eval_loss():
                optimizer.zero_grad()
                loss = nll_criterion(self.temperature_scale(logits), labels)
                loss.backward()
                return loss

            optimizer.step(eval_loss)

            # NLL and ECE after temp scaling
            after_temperature_nll = nll_criterion(self.temperature_scale(logits), labels).item()
            log.info(f'Optimal temperature: {round(self.temperature.item(), 3)}')
            log.info(f'After calibration - NLL: {round(after_temperature_nll, 3)}')

            output = deepcopy(ns.normal_predictions)
            output['confidence'] = torch.max(self.softmax(logits), dim=1).values.detach().numpy()
            info['result_df'] = output
        self.active = True
        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object], **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """ Perform temperature scaling on logits """
        prob_cols = [col for col in row_insights.columns
                     if '__mdb_proba' in col and
                     '__mdb_unknown_cat' not in col]
        if self.active and prob_cols:
            logits = torch.tensor(row_insights[prob_cols].values)
            confs = self.softmax(logits)
            row_insights['confidence'] = torch.max(confs, dim=1).values.detach().numpy().reshape(-1, 1)
        else:
            row_insights['confidence'] = torch.max(
                torch.tensor(row_insights[prob_cols].values), dim=1).values.detach().numpy().reshape(-1, 1)
        return row_insights, global_insights

