"""Post-processing utility functions."""
import typing as t
import regex as re


__all__ = [
    "remove_spurious_whitespaces_",
]


reg_spurious_whitespaces_punct = re.compile(
    r"(?<=[(\[{/\\\u201C\u2018])\s+|\s+(?=[:?!,.;)\]}/\\\u2019\u201D])"
)
reg_spurious_whitespaces_dash_between_letters = re.compile(
    r"(?<=[a-záéíóúãẽõâêôüç])\s*-\s*(?=[a-záéíóúãẽõâêôüç])"
)
reg_spurious_whitespaces_dot_between_numbers = re.compile(r"(?<=[0-9])\s*\.\s*(?=[0-9])")
reg_spurious_whitespaces_between_pars = re.compile(r"(?<=§)\s+(?=§)")


def remove_spurious_whitespaces_(segs: t.List[str]) -> None:
    """Remove unnecessary whitespaces injected by the segmenter tokenizer."""
    for i, text in enumerate(segs):
        text = reg_spurious_whitespaces_punct.sub("", text)
        text = reg_spurious_whitespaces_dash_between_letters.sub("-", text)
        text = reg_spurious_whitespaces_dot_between_numbers.sub(".", text)
        text = reg_spurious_whitespaces_between_pars.sub("", text)
        segs[i] = text.strip()
