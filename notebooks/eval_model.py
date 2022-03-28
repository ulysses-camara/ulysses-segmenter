"""Compute segmenter model metrics."""
import numpy as np
import sklearn.metrics


def compute_metrics(
    eval_pred,
    invalid_label_index: int = -100,
    available_labels: tuple[int, ...] = (0, 1, 2, 3),
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute per-class and macro validation scores for segmenter models."""
    pred_logits, labels = eval_pred
    predictions = np.argmax(pred_logits, axis=-1)

    true_predictions: list[int] = [
        pp
        for (p, l) in zip(predictions, labels)
        for (pp, ll) in zip(p, l)
        if ll != invalid_label_index
    ]

    true_labels: list[int] = [ll for l in labels for ll in l if ll != invalid_label_index]

    conf_mat = sklearn.metrics.confusion_matrix(
        y_true=true_labels,
        y_pred=true_predictions,
        labels=available_labels,
    )

    per_cls_precision = conf_mat.diagonal() / (eps + conf_mat.sum(axis=0))
    per_cls_recall = conf_mat.diagonal() / (eps + conf_mat.sum(axis=1))

    macro_precision = float(np.mean(per_cls_precision))
    macro_recall = float(np.mean(per_cls_recall))
    macro_f1 = 2.0 * macro_precision * macro_recall / (eps + macro_precision + macro_recall)

    overall_accuracy = float(np.sum(conf_mat.diagonal())) / len(true_labels)

    res: dict[str, float] = {
        **{f"per_cls_precision_{cls_i}": score for cls_i, score in enumerate(per_cls_precision)},
        **{f"per_cls_recall_{cls_i}": score for cls_i, score in enumerate(per_cls_recall)},
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "overall_accuracy": overall_accuracy,
    }

    return res
