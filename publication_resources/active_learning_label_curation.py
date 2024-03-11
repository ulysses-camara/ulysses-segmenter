import argparse
import datetime

import segmentador
import datasets
import scipy.special
import numpy as np
import tqdm


def sample_instances(
    dt: datasets.Dataset,
    segmenter: segmentador.Segmenter,
    n: int,
    q: float = 0.01,
    *,
    ignore_label: int = -100,
) -> list[int]:
    instance_quantile_margins = np.full(len(dt), fill_value=np.inf)

    for i in tqdm.tqdm(range(len(dt))):
        inst = dt[i]

        token_logits = segmenter(
            inst,
            return_logits=True,
            regex_justificativa="__IGNORE__",  # Do not suppress Justification or Annex parts.
        ).logits

        probs = scipy.special.softmax(token_logits, axis=-1)
        token_margins = np.diff(np.sort(probs, axis=-1)[:, [-2, -1]], axis=-1).ravel()
        true_labels = np.asarray(inst["labels"], dtype=int)

        assert len(token_logits) == len(inst["input_ids"])
        assert len(token_logits) == len(true_labels)

        try:
            is_not_middle_subword = true_labels != ignore_label  # Ignore margins from middle subwords; they don't have labels.
            token_margins = token_margins[is_not_middle_subword]

        except IndexError:
            token_margins = [np.inf, np.inf]

        try:
            instance_quantile_margins[i] = float(np.quantile(token_margins, q))
        except IndexError:
            instance_quantile_margins[i] = 0.0

    # NOTE: In Python, it is faster simply to argsort + slice instead of using a heap.
    #       Both implementation are equivalent in practice.
    ids_to_fetch = np.argsort(instance_quantile_margins)
    ids_to_fetch = ids_to_fetch[:n]

    return dt.select(ids_to_fetch)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("segmenter_uri")
    parser.add_argument("dataset_uri")
    parser.add_argument("n", help="Number of instances to select.", type=int)
    parser.add_argument("--dataset-uri", default="data/df_tokenized_split_0_120000_6000")
    parser.add_argument("--quantile", help="Cut-off quantile for token margins.", type=float, default=0.01)
    parser.add_argument("--device", default="cuda:0")
    parser.add_argument(
        "--ignore-label",
        help="Label to ignore (corresponding to subwords in the middle of words)",
        type=int,
        default=-100,
    )

    now = datetime.datetime.now()
    now = now.isoformat().split(".")[0]

    parser.add_argument("--output-uri", default=f"lowest_margin_instances_{now}")

    args = parser.parse_args()

    segmenter = segmentador.BERTSegmenter(uri_model=args.segmenter_uri, device=args.device)

    dt_input = datasets.Dataset.load_from_disk(args.dataset_uri)

    dt_output = sample_instances(
        dt=dt_input,
        segmenter=segmenter,
        n=args.n,
        q=args.quantile,
        ignore_label=args.ignore_label,
    )

    dt_output.save_to_disk(args.output_uri)
    print(f"Saved output dataset at '{args.output_uri}'.")
