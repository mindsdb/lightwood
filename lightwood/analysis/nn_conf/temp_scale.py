from typing import Dict, Tuple
from types import SimpleNamespace

import torch
import pandas as pd
from torch import nn, optim
from torch.nn import functional as F
from sklearn.preprocessing import OrdinalEncoder

from lightwood.analysis.base import BaseAnalysisBlock
from lightwood.helpers.log import log


class TempScaler(BaseAnalysisBlock):
    """
    Original reference (MIT Licensed): https://github.com/gpleiss/temperature_scaling
    NB: Output of the neural network should be the classification logits, NOT the softmax (or log softmax)!
    """
    def __init__(self):
        super().__init__()
        self.temperature = nn.Parameter(torch.ones(1))
        self.ordenc = OrdinalEncoder()
        self.softmax = torch.nn.Softmax()

    def temperature_scale(self, logits):
        temperature = self.temperature.unsqueeze(1).expand(logits.size(0), logits.size(1))  # expand & match logits size
        return logits / temperature

    def analyze(self, info: Dict[str, object], **kwargs) -> Dict[str, object]:
        """
        Tune and set the temperature of a neural model optimizing NLL using validation set logits.
        """
        ns = SimpleNamespace(**kwargs)
        self.n_cls = len(ns.stats_info.train_observed_classes)
        nll_criterion = nn.CrossEntropyLoss()
        ece_criterion = _ECELoss()
        self.ordenc.fit([[val] for val in ns.stats_info.train_observed_classes])

        # collect logits and labels for the validation set
        logits_list = []
        labels_list = []
        with torch.no_grad():
            prob_cols = [col for col in ns.normal_predictions.columns
                         if '__mdb_proba' in col and
                         '__mdb_unknown_cat' not in col]
            for logits, label in zip(ns.normal_predictions[prob_cols].values, ns.data[ns.target]):
                logits_list.append(logits.tolist())
                labels_list.append(int(self.ordenc.transform([[label]]).flatten()[0]))
            logits = torch.tensor(logits_list)
            labels = torch.tensor(labels_list).long()

        # NLL and ECE before temp scaling
        before_temperature_nll = nll_criterion(logits, labels).item()
        before_temperature_ece = ece_criterion(logits, labels).item()
        log.info('Before calibration - NLL: %.3f, ECE: %.3f' % (before_temperature_nll, before_temperature_ece))

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
        after_temperature_ece = ece_criterion(self.temperature_scale(logits), labels).item()
        log.info('Optimal temperature: %.3f' % self.temperature.item())
        log.info('After calibration - NLL: %.3f, ECE: %.3f' % (after_temperature_nll, after_temperature_ece))

        return info

    def explain(self,
                row_insights: pd.DataFrame,
                global_insights: Dict[str, object], **kwargs) -> Tuple[pd.DataFrame, Dict[str, object]]:
        """ Perform temperature scaling on logits """
        ns = SimpleNamespace(**kwargs)
        conf_cols = [col for col in row_insights.columns
                     if '__mdb_proba' in col and
                     '__mdb_unknown_cat' not in col]
        logits = torch.tensor(row_insights[conf_cols].values)
        scaled = logits / self.temperature
        confs = self.softmax(scaled)
        row_insights['confidence'] = torch.max(confs, axis=1).values.detach().numpy().reshape(-1, 1)
        return row_insights, global_insights


class _ECELoss(nn.Module):
    """
    Calculates the Expected Calibration Error of a model.
    (This isn't necessary for temperature scaling, just a cool metric).

    The input to this loss is the logits of a model, NOT the softmax scores.

    This divides the confidence outputs into equally-sized interval bins.
    In each bin, we compute the confidence gap:

    bin_gap = | avg_confidence_in_bin - accuracy_in_bin |

    We then return a weighted average of the gaps, based on the number
    of samples in each bin

    See: Naeini, Mahdi Pakdaman, Gregory F. Cooper, and Milos Hauskrecht.
    "Obtaining Well Calibrated Probabilities Using Bayesian Binning." AAAI.
    2015.
    """
    def __init__(self, n_bins=15):
        """
        n_bins (int): number of confidence interval bins
        """
        super(_ECELoss, self).__init__()
        bin_boundaries = torch.linspace(0, 1, n_bins + 1)
        self.bin_lowers = bin_boundaries[:-1]
        self.bin_uppers = bin_boundaries[1:]

    def forward(self, logits, labels):
        softmaxes = F.softmax(logits, dim=1)
        confidences, predictions = torch.max(softmaxes, 1)
        accuracies = predictions.eq(labels)

        ece = torch.zeros(1, device=logits.device)
        for bin_lower, bin_upper in zip(self.bin_lowers, self.bin_uppers):
            # Calculated |confidence - accuracy| in each bin
            in_bin = confidences.gt(bin_lower.item()) * confidences.le(bin_upper.item())
            prop_in_bin = in_bin.float().mean()
            if prop_in_bin.item() > 0:
                accuracy_in_bin = accuracies[in_bin].float().mean()
                avg_confidence_in_bin = confidences[in_bin].mean()
                ece += torch.abs(avg_confidence_in_bin - accuracy_in_bin) * prop_in_bin

        return ece
