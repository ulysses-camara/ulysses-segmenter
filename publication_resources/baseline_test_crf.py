import typing as t

import argparse
import collections
import itertools

import scipy.stats
import sklearn_crfsuite
import sklearn
import sklearn.model_selection
import nltk
import numpy as np

import baseline_utils
import approx_recall_and_precision


def word_to_feat(word: str) -> dict[str, t.Any]:
    return {
        "lower": word.lower(),
        "isart": word.lower() in {"art", "artigo"},
        "ispar": word.lower() in {"parágrafo", "§"},
        "sig": "".join("D" if c.isdigit() else ("c" if c.islower() else "C") for c in word),
        "length": len(word) if len(word) < 4 else ("long" if len(word) > 6 else "normal"),
        "islower": word.islower(),
        "isupper": word.isupper(),
        "istitle": word == word.capitalize(),
        "isdigit": word.isdigit(),
    }


def dataset_to_feats(docs):
    feats = []
    labels = []
    all_tokens = []

    for doc in docs:
        cur_feats = []
        cur_labels = []
        cur_tokens = []
        for sent in doc:
            tokens = nltk.tokenize.word_tokenize(sent, language="portuguese")
            if tokens:
                cur_feats.extend([word_to_feat(tok) for tok in tokens])
                cur_labels.extend(len(tokens) * ["NO-OP"])
                cur_labels[-len(tokens)] = "START"
                cur_tokens.extend(tokens)

        cur_labels[0] = "NO-OP"

        assert len(cur_feats) == len(cur_labels) == len(cur_tokens)

        feats.append(cur_feats)
        labels.append(cur_labels)
        all_tokens.append(cur_tokens)

    return (feats, labels, all_tokens)


def get_stats(labels):
    class_counter = collections.Counter()
    for yi in labels:
        class_counter.update(yi)
    return class_counter


def build_sents(tokens_per_doc, preds_per_doc):
    assert len(tokens_per_doc) == len(preds_per_doc)

    sents = []
    is_start_count = 0
    n_docs = len(tokens_per_doc)

    for cur_tokens, cur_labels in zip(tokens_per_doc, preds_per_doc, strict=True):
        cur_sent = []
        for tok, lab in zip(cur_tokens, cur_labels, strict=True):
            if lab == "START":
                sents.append(" ".join(cur_sent))
                cur_sent.clear()
                is_start_count += 1

            cur_sent.append(tok)

        if cur_sent:
            sents.append(" ".join(cur_sent))

    assert len(sents) == is_start_count + n_docs

    return sents


def run(redo_hparam_search: bool = False):
    # Data info: https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data
    # Tokenizer info: https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#available-models

    train_docs = baseline_utils.load_ground_truth_sentences(
        test_uri="data/dataset_ulysses_segmenter_v2_active_learning_curated_only",
        tokenizer_uri="tokenizers/6000_subwords",
        split="train",
        group_by_document=True,
    )

    test_docs = baseline_utils.load_ground_truth_sentences(
        test_uri="data/dataset_ulysses_segmenter_v2_active_learning_curated_only",
        tokenizer_uri="tokenizers/6000_subwords",
        split="test",
        group_by_document=True,
    )

    train_feats, train_labels, _ = dataset_to_feats(train_docs)
    test_feats, _, test_tokens = dataset_to_feats(test_docs)
    flatten_test_docs = list(itertools.chain(*test_docs))

    if not redo_hparam_search:
        best_score = 0.610011381785407
        best_config = {"c1": 0.0717248764429347, "c2": 0.019492472144742503, "min_freq": 7}

    else:
        best_score = -np.inf
        best_config = None

        param_distributions = {
            "c1": scipy.stats.expon(scale=0.5),
            "c2": scipy.stats.expon(scale=0.5),
            "min_freq": scipy.stats.randint(low=5, high=50),
        }

        rng = np.random.RandomState(241500)

        sample_size = 400
        sample_inds = rng.choice(len(train_feats), replace=False, size=sample_size)
        sample_train_feats = [train_feats[i] for i in sample_inds]
        sample_train_labels = [train_labels[i] for i in sample_inds]

        assert len(sample_train_feats) == len(sample_train_labels) == sample_size
        assert len(sample_inds) == len(set(sample_inds))

        for fn in param_distributions.values():
            fn.random_state = np.random.RandomState(rng.randint(1, 2**31 - 1))

        for _ in np.arange(50):
            splitter = sklearn.model_selection.KFold(n_splits=3, shuffle=True, random_state=rng.randint(1, 2**31 - 1))
            hparams = {k: fn.rvs() for k, fn in param_distributions.items()}
            cur_eval_scores = []

            for inds_train, inds_eval in splitter.split(sample_train_feats, sample_train_labels):
                X_train = [train_feats[i] for i in inds_train]
                X_eval = [train_feats[i] for i in inds_eval]

                y_train = [train_labels[i] for i in inds_train]
                y_eval = [train_labels[i] for i in inds_eval]

                segmenter = sklearn_crfsuite.CRF(algorithm="lbfgs", **hparams)
                segmenter.fit(X_train, y_train)
                y_preds = segmenter.predict(X_eval)

                y_preds = np.asarray([int(yi == "START") for yi in itertools.chain(*y_preds)])
                y_eval = np.asarray([int(yi == "START") for yi in itertools.chain(*y_eval)])

                assert len(y_preds) == len(y_eval)

                eval_score = sklearn.metrics.f1_score(y_true=y_eval, y_pred=y_preds, average="binary")
                cur_eval_scores.append(eval_score)

            eval_score = float(np.mean(cur_eval_scores))
            print(hparams, eval_score)

            if best_score < eval_score:
                best_score = eval_score
                best_config = hparams.copy()

    print("Best config:")
    print(best_config, best_score, end="\n\n")

    segmenter = sklearn_crfsuite.CRF(algorithm="lbfgs", **best_config)
    segmenter.fit(train_feats, train_labels)

    y_preds = segmenter.predict(test_feats)
    sents_pred = build_sents(test_tokens, y_preds)

    # NOTE: removing whitespaces for comparison because CRF's output insert additional spaces before punctuation symbols.
    print(approx_recall_and_precision.estimate_seg_perf(sents_pred, flatten_test_docs, remove_whitespaces=True))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--redo-hparam-search", action="store_true", help="If enabled, redo hyper-parameter optimization.")
    args = parser.parse_args()
    run(**vars(args))
