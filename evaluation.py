import numpy as np
import pandas as pd 
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score
import collections

def log_loss_multilabel(y_true, y_prob):
    """
    In multi-label setting log loss is calculated for each class, then averaged 
    
    :param y_true: matrix of binary class labels
    :param y_prob: matrix of class probabilities 
    :returns: float
    """
    assert y_true.shape[1] == y_prob.shape[1], "number of classes for y_pred and y_true must match" 
    log_loss_per_class = [log_loss(y_true[:, i], y_prob[:, i]) for i in range(y_true.shape[1])]
    return np.mean(log_loss_per_class)


def cross_validate_multilabel(model, X, Y, **cv_kwargs):
    """cross validation for a multi-label target"""
    # scores is ndarray of shape (number of Y classes, number of cross validation folds)
    scores_per_class = np.array([cross_val_score(model, X, y, **cv_kwargs) for y in Y.T])
    # return average score across folds, for each class
    return scores_per_class.mean(axis=1)


def multilabel_results(cv_scores, class_labels=None, test_labels=None, aggregate=True):
    """
    Return DataFrame containing one or more sets of multilabel cross_validated results
    
    :param cv_scores: list of lists of cross validated scores
    :param class_labels: list of class labels for dataframe columns 
    :param aggregate: boolean, set True to create additional column containing multilabel mean score
    :param test_labels: list of test names to use as index index
    :returns: DataFrame
    """
    # if input corresponds to a single row (not a list of lists) then coerce to format expected by fromrecords
    if not isinstance(cv_scores[0], collections.Sequence):
        cv_scores = [cv_scores]
    df = pd.DataFrame.from_records(cv_scores, columns=class_labels, index=list(test_labels))
    if aggregate:
        df['all'] = df.mean(axis=1)
    return df