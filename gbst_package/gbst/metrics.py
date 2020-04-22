# coding: utf-8
"""The interface for user-defined metrics."""

import numpy as np
from sklearn.metrics import roc_auc_score

def evalauc(preds, deval):
    """
    Average AUC metric as default, but written in python.
    This serves as an example of implementing new metrics.

    Parameters
    ----------
    preds: the predicted f() of deval, calculated by GBST model.
    deval: the eval dataset.

    Returns
    -------
    metric_name : str
    metric_value: float
    """
    labels = deval.get_label().astype(int)
    y_arr = np.zeros([preds.shape[0], preds.shape[1]])
    for i, label in enumerate(labels):
        y_arr[i, :label] = 1
        y_arr[i, label:] = 0
    hazards = 1./(1.+np.exp(-preds))
    mults = np.ones(hazards.shape[0])
    auc_total = []
    for timestep in range(0, hazards.shape[1]):
        mults = mults * (1 - hazards[:, timestep])
        try:
            auc = roc_auc_score(y_true=y_arr[:, timestep], y_score=mults)
            auc_total.append(auc)
        except Exception as e:
            # If all candidates are alive/default, then roc_auc_score will throw an exception.
            # Such cases are excluded from aggregation.
            pass
    return 'AUC', float(np.sum(auc_total)) / len(auc_total)

