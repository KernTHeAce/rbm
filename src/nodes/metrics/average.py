from sklearn.metrics import precision_recall_fscore_support

from .const import MetricAverageValues as av

class MetricAverage:
    def __call__(self, y_true, y_pred, average: str = av.WEIGHTED):
        """
        Average metrics
        Parameters:
            average : {'binary', 'micro', 'macro', 'samples','weighted'}, \
                    default=None
                If ``None``, the scores for each class are returned. Otherwise, this
                determines the type of averaging performed on the data:

                ``'binary'``:
                    Only report results for the class specified by ``pos_label``.
                    This is applicable only if targets (``y_{true,pred}``) are binary.
                ``'micro'``:
                    Calculate metrics globally by counting the total true positives,
                    false negatives and false positives.
                ``'macro'``:
                    Calculate metrics for each label, and find their unweighted
                    mean.  This does not take label imbalance into account.
                ``'weighted'``:
                    Calculate metrics for each label, and find their average weighted
                    by support (the number of true instances for each label). This
                    alters 'macro' to account for label imbalance; it can result in an
                    F-score that is not between precision and recall.
                ``'samples'``:
                    Calculate metrics for each instance, and find their average (only
                    meaningful for multilabel classification where this differs from
                    :func:`accuracy_score`).
        Returns:
            precision : float (if average is not None) or array of float, shape =\
                [n_unique_labels]

            recall : float (if average is not None) or array of float, shape =\
                [n_unique_labels]

            fbeta_score : float (if average is not None) or array of float, shape =\
                [n_unique_labels]

            support : None (if average is not None) or array of int, shape =\
                [n_unique_labels]
                The number of occurrences of each label in ``y_true``.
        """
        precision, recall, f1, support = precision_recall_fscore_support(
            y_true=y_true, y_pred=y_pred, average=average
        )
        return precision, recall, f1, support
