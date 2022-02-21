import os
import typing as t
import sys

import pandas as pd
import colorama

sys.path.append("..")
from config import *


REGISTERED_TEST_CASES_URI = "./registered_test_cases.csv"
REGISTERED_TEST_CASES_COLUMNS = ("document_index", "expected_segment_count", "expected_noise_count")
TEST_CASES: dict[int, tuple[int, int]] = dict()


def load_registered_cases(test_cases_uri: str = REGISTERED_TEST_CASES_URI) -> None:
    if not os.path.isfile(test_cases_uri):
        print(f"No test cases found at '{test_cases_uri}'.")
        return

    df = pd.read_csv(test_cases_uri, usecols=REGISTERED_TEST_CASES_COLUMNS, index_col=None)

    for i, (idx, *expected_values) in df.iterrows():
        TEST_CASES[idx] = expected_values

    print(f"Loaded {len(TEST_CASES)} test cases from '{test_cases_uri}'.")


def dump_registered_cases(test_cases_uri: str = REGISTERED_TEST_CASES_URI) -> None:
    items = []

    for key, vals in TEST_CASES.items():
        items.append((key, *vals))

    df = pd.DataFrame(items, columns=REGISTERED_TEST_CASES_COLUMNS)
    df.to_csv(test_cases_uri, index=False)
    print(f"Wrote {len(df)} test cases at '{test_cases_uri}'.")


def clear_registered_cases() -> None:
    TEST_CASES.clear()


def update_test_case(document_idx: int, expected_segment_count: t.Union[int, tuple[int, ...]]) -> None:
    if not hasattr(expected_segment_count, "__len__"):
        expected_segment_count = (expected_segment_count,)

    assert len(expected_segment_count) == len(REGISTERED_TEST_CASES_COLUMNS) - 1, f"Got length: {len(expected_segment_count)}"
    TEST_CASES[document_idx] = tuple(expected_segment_count)


def test_case_exists(document_idx: int) -> bool:
    return document_idx in TEST_CASES


def print_results(df, id_: int, keyword_filters: t.Union[set[str], str, None] = None, print_full_text: bool = True) -> tuple[int, int]:
    tokens = df["train"][id_]["tokens"]
    labels = df["train"][id_]["labels"]
    
    if print_full_text:
        print(" ".join(df["train"][id_]["tokens"]))
        print(64 * "_", end="\n\n")
    
    sentence: list[str] = []
    segment_count = int(len(tokens) > 0)
    noise_count = 0
    
    c_color = colorama.Fore.LIGHTWHITE_EX
    c_end = colorama.Fore.RESET
    
    if isinstance(keyword_filters, str):
        keyword_filters = {keyword_filters}
    
    if keyword_filters is not None and not isinstance(keyword_filters, set):
        keyword_filters = set(keyword_filters)
    
    for tok, lab in zip(tokens, labels):
        if lab == SPECIAL_SYMBOLS[MARKER_VALID]:
            if (not keyword_filters or not keyword_filters.isdisjoint(map(str.lower, sentence))):
                print(c_color, segment_count, c_end, " ".join(sentence), end="\n\n")
                
            sentence = []
            segment_count += 1
        
        if lab == SPECIAL_SYMBOLS[MARKER_NOISE_START]:
            tok = colorama.Fore.RED + tok
            noise_count += 1
            
        if lab == SPECIAL_SYMBOLS[MARKER_NOISE_END]:
            tok = colorama.Fore.RESET + tok
        
        sentence.append(tok)
       
    if sentence and (not keyword_filters or not keyword_filters.isdisjoint(map(str.lower, sentence))):
        print(c_color, segment_count, c_end, " ".join(sentence), end="\n\n")
    
    print(colorama.Fore.RESET)
    print(f"Idx/Segment count, noise count:   {id_}: {segment_count}, {noise_count}")
    
    return segment_count, noise_count
