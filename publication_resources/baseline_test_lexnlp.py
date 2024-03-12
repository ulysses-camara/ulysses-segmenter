import itertools

import lexnlp.nlp.en.segments.sentences as lexnlp

import baseline_utils
import approx_recall_and_precision


class SegmenterLexNLP:
    def __call__(self, document_texts):
        sents = []
        for text in document_texts:
            sents.extend(lexnlp.get_sentence_list(text))
        return sents


def run():
    test_docs = baseline_utils.load_ground_truth_sentences(
        test_uri="data/dataset_ulysses_segmenter_v2_active_learning_curated_only",
        tokenizer_uri="tokenizers/6000_subwords",
        split="test",
        group_by_document=True,
    )
    segs_true = list(itertools.chain(*test_docs))

    segmenter = SegmenterLexNLP()
    segs_pred = segmenter([" ".join(doc) for doc in test_docs])
    res = approx_recall_and_precision.estimate_seg_perf(sentences_pred=segs_pred, sentences_true=segs_true)
    print(res)


if __name__ == "__main__":
    run()
