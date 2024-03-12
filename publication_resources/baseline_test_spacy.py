import itertools

import spacy
import spacy.cli
import tqdm

import approx_recall_and_precision
import baseline_utils


class SegmenterSpacy:
    def __init__(self):
        self.model_name = "pt_core_news_lg"

        if not spacy.util.is_package(self.model_name):
            # Or use the following command:
            # python -m spacy download pt_core_news_lg
            spacy.cli.download(self.model_name)

        self.spacy_model = spacy.load(self.model_name)

    def __call__(self, document_texts):
        sents = []
        for text in tqdm.tqdm(document_texts):
            sents.extend(list(map(str, self.spacy_model(text).sents)))
        return sents


def run():
    test_docs = baseline_utils.load_ground_truth_sentences(
        test_uri="data/dataset_ulysses_segmenter_v2_active_learning_curated_only",
        tokenizer_uri="tokenizers/6000_subwords",
        split="test",
        group_by_document=True,
    )
    segs_true = list(itertools.chain(*test_docs))

    segmenter = SegmenterSpacy()
    segs_pred = segmenter([" ".join(doc) for doc in test_docs])
    res = approx_recall_and_precision.estimate_seg_perf(sentences_pred=segs_pred, sentences_true=segs_true)
    print(res)


if __name__ == "__main__":
    run()
