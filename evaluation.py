import numpy as np
from sklearn.metrics import log_loss
from sklearn.model_selection import cross_val_score

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
    # return average score across classes, for each fold
    return scores_per_class.mean(axis=0)