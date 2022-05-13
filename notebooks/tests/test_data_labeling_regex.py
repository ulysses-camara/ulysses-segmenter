import sys

sys.path.append("..")

from . import conftest
from config import *


def run_tests(labels: list[list[int]]) -> None:
    correct = 0
    skipped = 0

    incorrect_cases_inds: list[str] = []

    if not conftest.TEST_CASES:
        return

    for idx, (
        expected_segment_count,
        expected_noise_count,
    ) in conftest.TEST_CASES.items():
        if idx >= len(labels):
            skipped += 1
            continue

        total_segments = int(len(labels[idx]) > 0)
        total_segments += sum(lab == SPECIAL_SYMBOLS[MARKER_VALID] for lab in labels[idx])
        total_noise = sum(lab == SPECIAL_SYMBOLS[MARKER_NOISE_START] for lab in labels[idx])

        if total_segments == expected_segment_count and total_noise == expected_noise_count:
            correct += 1
            continue

        incorrect_cases_inds.append(str(idx))

    n_tests = len(conftest.TEST_CASES)
    correct_prop = correct / n_tests

    print(
        f"Correct proportion: {100. * correct_prop:.2f}% ({correct} of {n_tests})"
        + (f", {skipped} tests skipped." if skipped > 0 else "")
    )

    assert correct == n_tests - skipped, f"Incorrect: {', '.join(incorrect_cases_inds)}"
