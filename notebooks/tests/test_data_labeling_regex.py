import sys
sys.path.append("..")

from . import conftest
from config import *


def run_tests(labels: list[list[int]]):
    correct = 0
    skipped = 0

    incorrect_cases_inds: list[str] = []
    
    if not conftest.TEST_CASES:
        return
    
    for idx, expected in conftest.TEST_CASES.items():
        if idx >= len(labels):
            skipped += 1
            continue
            
        total_segments = int(len(segments[idx]) > 0)
        total_segments += sum([lab == SPECIAL_SYMBOLS[MARKER_VALID] for lab in segments[idx]])

        if total_segments == expected:
            correct += 1
            continue

        incorrect_cases_inds.append(str(idx))
            
    correct_prop = correct / len(test_cases)
    
    print(
        f"Correct proportion: {100. * correct_prop:.2f}% ({correct} of {len(test_cases)})" +
        (f", {skipped} tests skipped." if skipped > 0 else "")
    )

    assert correct == len(test_cases) - skipped, f"Incorrect: {', '.join(incorrect_cases_inds)}"
