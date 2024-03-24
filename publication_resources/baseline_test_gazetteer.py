import itertools

import nltk
import numpy as np
import tqdm

import approx_recall_and_precision
import baseline_utils


class Gazetteer:
    def __init__(self):
        self.dictionary_unigram = {
            "Art",
            "Artigo",
            "I",
            "II",
            "III",
            "IV",
            "V",
            "VI",
            "VII",
            "VIII",
            "IX",
            "X",
            "XI",
            "XII",
            "XIII",
            "XIV",
            "XV",
            "XVI",
            "XVII",
            "XVIII",
            "XIX",
            "XX",
            "XXI",
            "XXII",
            "XXIII",
            "XXIV",
            "XXV",
            "XXVI",
            "XXVII",
            "XXVIII",
            "XXIX",
            "XXX",
            "Parágrafo"
            "§",
            "Sala",
            "Brasília",
            "TÍTULO"
            "TITULO"
            "CAPÍTULO",
            "CAPITULO",
            "SEÇÃO",
            "SECAO",
            "SUBSEÇÃO",
            "SUBSECAO",
            "SUB-SEÇÃO",
            "SUB-SECAO",
            "PROJETO",
            "PROPOSTA",
            "REQUERIMENTO",
            "SOLICITAÇÃO",
            "TVR",
            "MEDIDA",
            "MENSAGEM",
            "OFÍCIO",
            "Ofício",
            "JUSTIFICATIVA",
            "Justificativa",
            "Autor",
            "AUTOR",
            "Altera",
            "ALTERA",
            "Dispõe",
            "DISPÕE",
            "Acrescenta",
            "ACRESCENTA",
            "Dá",
            "DÁ",
            "Institui",
            "INSTITUI",
            "Estabelece",
            "ESTABELECE",
            "Aprova",
            "APROVA",
            "Susta",
            "SUSTA",
            "Autoriza",
            "AUTORIZA",
            "Autor",
            "AUTOR",
            "Modifica",
            "MODIFICA",
            "Regulamenta",
            "REGULAMENTA",
            "Cria",
            "CRIA",
            "Tipifica",
            "TIPIFICA",
            "Torna",
            "TORNA",
            "Denomina",
            "DENOMINA",
            "Inclui",
            "INCLUI",
            "Veda",
            "VEDA",
        }

        self.dictionary_bigram = {
            ("O", "Congresso"),
            ("O", "CONGRESSO"),
        }

        delims = ["-", ".", ")"]
        letters = "abcdefghijklmnopqrstuvwxyz"
        self.dictionary_bigram.update(itertools.product(letters.lower(), delims))
        self.dictionary_bigram.update(itertools.product(letters.upper(), delims))
        self.dictionary_bigram.update(itertools.product(map(str, range(1, 10)), delims))

    def __call__(self, document_texts):
        sents = []
        for text in tqdm.tqdm(document_texts):
            tokens = nltk.word_tokenize(text, language="portuguese")
            tokens.append("[END]")

            seg_start_inds = [0]

            for i, (cur_, next_) in enumerate(zip(tokens[:-1], tokens[1:])):
                if (cur_, next_) in self.dictionary_bigram or cur_ in self.dictionary_unigram:
                    seg_start_inds.append(i)

            tokens.pop()
            seg_start_inds.append(len(tokens))

            sents.extend([
                " ".join(tokens[i_start:i_end])
                for i_start, i_end in zip(seg_start_inds[:-1], seg_start_inds[1:])
                if (i_end - i_start) > 1
            ])

        return sents


def run():
    test_docs = baseline_utils.load_ground_truth_sentences(
        test_uri="data/dataset_ulysses_segmenter_v2_active_learning_curated_only",
        tokenizer_uri="tokenizers/6000_subwords",
        split="test",
        group_by_document=True,
    )
    segs_true = list(itertools.chain(*test_docs))

    segmenter = Gazetteer()
    segs_pred = segmenter([" ".join(doc) for doc in test_docs])

    res = approx_recall_and_precision.estimate_seg_perf(
        sentences_pred=segs_pred,
        sentences_true=segs_true,
        remove_whitespaces=True,
    )

    print(res)


if __name__ == "__main__":
    run()
