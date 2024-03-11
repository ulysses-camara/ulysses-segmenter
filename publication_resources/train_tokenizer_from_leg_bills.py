import os

import transformers
import pandas as pd
import regex


VOCAB_SIZE = 6000
TOKENIZER_OUTPUT_DIR = f"tokenizers/{VOCAB_SIZE}_subwords"
os.makedirs(TOKENIZER_OUTPUT_DIR, exist_ok=True)

# NOTE: any BERT-like tokenizer will do, since we won't reuse the pretrained tokens anyway.
# We're loading a pretrained tokenizer just for a quick setup of the architecture/special tokens.
tokenizer = transformers.AutoTokenizer.from_pretrained("neuralmind/bert-base-portuguese-cased")

UPPERCASE_LETTERS = "ÀÁÂÃÇÉÊẼÓÕÔÜÚÍA-Z\u0303\u0300\u0301\u0302\u0303\u0304\u0305\u0340\u0341\u0342\u0343"

# Data info + download link: https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
df = pd.read_csv("data/ulysses_segmenter_raw_data.txt", usecols=["content"], header=0, index_col=None).squeeze("columns")

df.dropna(inplace=True)
df.drop_duplicates(inplace=True)

good_inds = [i for i, text in enumerate(df) if isinstance(text, str) and len(text) >= 10]
df = df.iloc[good_inds]

RE_JUSTIFICATIVA = regex.compile(
    r"\s*(?:"
    + r"J\s*U\s*S\s*T\s*I\s*F\s*I\s*C\s*A?\s*T\s*I\s*V\s*A|"
    + r"J\s*u\s*s\s*t\s*i\s*f\s*i\s*c\s*a\s*t\s*i\s*v\s*a\s+(?=["
    + UPPERCASE_LETTERS
    + r"])|"
    + r"J\s*U\s*S\s*T\s*I\s*F\s*I\s*C\s*A\s*[CÇ]\s*[AÂÃÀÁ]\s*O|"
    + r"J\s*u\s*s\s*t\s*i\s*f\s*i\s*c\s*a\s*[cç]\s*[aãâàá]\s*o\s+(?=["
    + UPPERCASE_LETTERS
    + r"])"
    + r")"
)

RE_ANEXO = regex.compile(r"\s*A\s*N\s*E\s*X\s*O")

# NOTE: removing less relevant sections.
df = df.map(lambda item: RE_JUSTIFICATIVA.split(item)[0])
df = df.map(lambda item: RE_ANEXO.split(item)[0])

tokenizer = tokenizer.train_new_from_iterator(df, vocab_size=VOCAB_SIZE)

tokenizer.save_pretrained(TOKENIZER_OUTPUT_DIR)
