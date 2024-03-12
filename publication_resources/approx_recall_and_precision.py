"""Estimate Precision and Recall for any segmentation model or algorithm."""

import bisect
import re


def estimate_seg_perf(sentences_pred: list[str], sentences_true: list[str], remove_whitespaces: bool = False):
    """Estimate Precision and Recall for any segmentation model or algorithm."""
    if remove_whitespaces:
        # NOTE: some algorithms may produce output with additional spaces after or before punctuation symbols, like ','.
        # This optional preprocessing avoids misclassifying instances due to misplaced whitespace characters.
        reg_whitespaces = re.compile(r"\s+")
        sentences_pred = [reg_whitespaces.sub("", item) for item in sentences_pred]
        sentences_true = [reg_whitespaces.sub("", item) for item in sentences_true]

    estimated_correct = 0

    # NOTE: sorting to perform binary search.
    sorted_sentences_true = sorted(sentences_true)

    for pred in sentences_pred:
        if len(sorted_sentences_true) == 0:
            break

        # NOTE: bisect = binary search.
        i = bisect.bisect_right(sorted_sentences_true, pred) - 1

        if pred.startswith(sorted_sentences_true[i]) or sorted_sentences_true[i].startswith(pred):
            estimated_correct += 1
            sorted_sentences_true.pop(i)  # NOTE: avoid reusing true sentences.

    estimated_precision = estimated_correct / len(sentences_pred)
    estimated_recall = estimated_correct / len(sentences_true)

    return {"estimated_precision": estimated_precision, "estimated_recall": estimated_recall}
