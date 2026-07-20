def fbeta_score(tp, fp, fn, beta=1.0, tp_recall=None):
    if tp_recall is None:
        tp_recall = tp

    if tp + fp == 0:
        precision = 0
    else:
        precision = tp / (tp + fp)

    if tp_recall + fn == 0:
        recall = 0
    else:
        recall = tp_recall / (tp_recall + fn)

    if precision == 0 and recall == 0:
        return 0.0

    beta_squared = beta**2
    fbeta = (
        (1 + beta_squared)
        * (precision * recall)
        / ((beta_squared * precision) + recall)
    )

    return fbeta
