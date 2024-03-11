import itertools

import nltk

import baseline_utils
import approx_recall_and_precision


NATURAL_TEXT_SEG_ABBRV = {
    "art",
    "arts",
    "profa",
    "profᵃ",
    "dep",
    "sr",
    "sra",
    "srᵃ",
    "s.exª",
    "s.exa",
    "v.em.ª",
    "v.ex.ª",
    "v.mag.ª",
    "v.em.a",
    "v.ex.a",
    "v.mag.a",
    "v.m",
    "v.rev.ª",
    "v.rev.a",
    "v.s",
    "v.s.ª",
    "v.s.a",
    "v.a",
    "v.emª",
    "v.exª",
    "v.ema",
    "v.exa",
    "v.magª",
    "me",
    "ma",
    "v.sa",
    "v.ex.ªrev.ma",
    "v.ex.arev.ma",
    "v.p",
    "v.revª",
    "v.rev.ma",
    "v.m.cê",
    "v.sª",
    "dra",
    "drª",
    "profª",
    "ass",
    "obs",
    "par",
    "pag",
    "pág",
    "pars",
    "pags",
    "págs",
    "cap",
    "caps",
    "1º",
    "2º",
    "3º",
    "4º",
    "5º",
    "6º",
    "7º",
    "8º",
    "9º",
    "1o",
    "2o",
    "3o",
    "4o",
    "5o",
    "6o",
    "7o",
    "8o",
    "9o",
    "10o",
    "1",
    "2",
    "3",
    "4",
    "5",
    "6",
    "7",
    "8",
    "9",
    "10",
    "ph.d",
    "phd",
    "et al",
    "adv.º",
    "advº",
    "advª",
    "ind",
    "vol",
    "n",
    "nº",
    "núm",
    "nro",
    "nrº",
    "gab",
    "ex",
    "www",
    "gov",
    "http",
    "https",
    "com",
    "br",
    "org",
    "http://www",
    "https://www",
    "gov.br",
    "org.br",
    "leg",
    "http://www1",
    "https://www1",
    "parág",
    "m.d",
}

print("Number of additional abbreviations:", len(NATURAL_TEXT_SEG_ABBRV))

# Data info: https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
# Tokenizer info: https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#available-models

test_docs = baseline_utils.load_ground_truth_sentences(
    test_uri="data/dataset_ulysses_segmenter_v2_active_learning_curated_only",
    tokenizer_uri="tokenizers/6000_subwords",
    group_by_document=True,
)
segs_true = list(itertools.chain(*test_docs))


tokenizer = nltk.data.load("tokenizers/punkt/portuguese.pickle")
segs_pred = []
for doc in test_docs:
    segs_pred.extend(tokenizer.tokenize(" ".join(doc)))
print("Results without additional abbreviations:")
print(approx_recall_and_precision.estimate_seg_perf(sentences_pred=segs_pred, sentences_true=segs_true))

# NOTE: injecting additional abbreviations into NLTK model.
tokenizer._params.abbrev_types.update(NATURAL_TEXT_SEG_ABBRV)

segs_pred = []
for doc in test_docs:
    segs_pred.extend(tokenizer.tokenize(" ".join(doc)))
print("Results with additional abbreviations:")
print(approx_recall_and_precision.estimate_seg_perf(sentences_pred=segs_pred, sentences_true=segs_true))
