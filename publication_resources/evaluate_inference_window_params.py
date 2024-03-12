import typing as t
import os
import pathlib
import pickle
import collections
import json

import datasets
import regex
import tqdm
import numpy as np
import sklearn.metrics

import segmentador
import utils


re_non_letters_or_num = regex.compile(r"[^A-Z0-9]+", regex.IGNORECASE)
ConfigType = collections.namedtuple("Config", ["model_class", "window_size", "window_shift"])
RESULTS_CACHE_DIR = "inference_cache_results"

os.makedirs(RESULTS_CACHE_DIR, exist_ok=True)


ResultTypeKey = tuple[str, str, int, int]
ResultTypeValue = dict[str, float]
ResultType = dict[ResultTypeKey, ResultTypeValue]


def compute_metrics(
    eval_pred,
    invalid_label_index: int = -100,
    available_labels: tuple[int, ...] = (0, 1, 2, 3),
    eps: float = 1e-8,
) -> dict[str, float]:
    """Compute per-class and macro validation scores for segmenter models."""
    pred_logits, labels = eval_pred
    predictions = np.argmax(pred_logits, axis=-1)

    true_predictions: list[int] = [
        pp for (p, l) in zip(predictions, labels) for (pp, ll) in zip(p, l) if ll != invalid_label_index
    ]

    true_labels: list[int] = [ll for l in labels for ll in l if ll != invalid_label_index]

    conf_mat = sklearn.metrics.confusion_matrix(
        y_true=true_labels,
        y_pred=true_predictions,
        labels=available_labels,
    )

    per_cls_precision = conf_mat.diagonal() / (eps + conf_mat.sum(axis=0))
    per_cls_recall = conf_mat.diagonal() / (eps + conf_mat.sum(axis=1))

    macro_precision = float(np.mean(per_cls_precision))
    macro_recall = float(np.mean(per_cls_recall))
    macro_f1 = 2.0 * macro_precision * macro_recall / (eps + macro_precision + macro_recall)

    overall_accuracy = float(np.sum(conf_mat.diagonal())) / len(true_labels)

    res: dict[str, float] = {
        **{f"per_cls_precision_{cls_i}": score for cls_i, score in enumerate(per_cls_precision)},
        **{f"per_cls_recall_{cls_i}": score for cls_i, score in enumerate(per_cls_recall)},
        "macro_precision": macro_precision,
        "macro_recall": macro_recall,
        "macro_f1": macro_f1,
        "overall_accuracy": overall_accuracy,
    }

    return res


def validate_single_model(
    model: segmentador.Segmenter,
    dt: dict[str, list[t.Any]],
    moving_window_sizes: tuple[int, ...],
    window_shift_sizes: tuple[float, ...],
    batch_size: int = 128,
) -> dict[tuple[int, int], ResultTypeValue]:
    all_res: dict[tuple[int, int], ResultTypeValue] = {}

    for moving_window_size in moving_window_sizes:
        for window_shift_size in window_shift_sizes:
            lock = True
            cur_batch_size = batch_size
            config = ConfigType(model.__class__.__name__, moving_window_size, window_shift_size)
            print("Starting config:", config)

            while lock:
                try:
                    cur_ret = model(
                        dt,
                        batch_size=cur_batch_size,
                        moving_window_size=moving_window_size,
                        window_shift_size=window_shift_size,
                        return_logits=True,
                        show_progress_bar=False,
                    )
                    lock = False

                except RuntimeError:
                    cur_batch_size //= 2
                    print(f"Insufficient CUDA memory, retrying with batch_size={cur_batch_size}")

                if cur_batch_size <= 0:
                    print("Insufficient CUDA memory to ANY batch size, failed.")
                    lock = False

            cur_res = compute_metrics(([cur_ret.logits], [dt["labels"]]))
            all_res[(moving_window_size, window_shift_size)] = cur_res
            print("Completed config:", config)

    return all_res


def aggregate_result(res_base, new_res, key_prefix: tuple[str, ...]) -> None:
    res_base.update({(*key_prefix, *key): val for key, val in new_res.items()})


def validate_all_models(
    segmenter_cls,
    dt: dict[str, list[t.Any]],
    uri_models: list[str],
    uri_tokenizers: t.Optional[t.Union[str, list[str]]] = None,
    moving_window_sizes: tuple[int, ...] = (128, 256, 512, 1024),
    window_shift_sizes: tuple[float, ...] = (0.25, 0.5, 1.0),
    inference_pooling_operations: tuple[str, ...] = ("max", "sum", "gaussian", "asymmetric-max"),
    device: str = "cuda:0",
    batch_size: int = 128,
) -> ResultType:
    all_res: ResultType = {}

    uri_tokenizer = None

    for i, uri_model in enumerate(tqdm.tqdm(uri_models)):

        if uri_tokenizers is not None:
            uri_tokenizer = uri_tokenizers if isinstance(uri_tokenizers, str) else uri_tokenizers[i]

        try:
            for inference_pooling_operation in inference_pooling_operations:
                cache_data = (uri_model, inference_pooling_operation)
                cache_data_path = re_non_letters_or_num.sub("_", "_".join(cache_data))
                cache_uri = os.path.join(
                    RESULTS_CACHE_DIR,
                    f"cache_{segmenter_cls.__name__}_{cache_data_path}.pickle",
                )

                if os.path.isfile(cache_uri):
                    cur_res = load_results_from_file(input_uri=cache_uri)
                    aggregate_result(all_res, cur_res, cache_data)
                    print(f"Found cached '{cache_uri}'.")
                    continue

                seg_model = segmenter_cls(
                    uri_model=uri_model,
                    uri_tokenizer=uri_tokenizer,
                    inference_pooling_operation=inference_pooling_operation,
                    device=device,
                )

                cur_res = validate_single_model(
                    model=seg_model,
                    dt=dt,
                    moving_window_sizes=moving_window_sizes,
                    window_shift_sizes=window_shift_sizes,
                    batch_size=batch_size,
                )

                save_results_in_file(res=cur_res, output_uri=cache_uri)

                aggregate_result(all_res, cur_res, cache_data)

        except OSError:
            print(f"Unable to load '{uri_model}'.")

    return all_res


def save_results_in_file(res: ResultType, output_uri: str, overwrite: bool = False) -> None:
    if not overwrite and os.path.isfile(output_uri):
        raise FileExistsError("Can not overwrite file if 'overwrite=False'.")

    result_output_dir = os.path.dirname(output_uri)

    if not os.path.isdir(result_output_dir):
        pathlib.Path(result_output_dir).mkdir(exist_ok=True, parents=True)

    with open(output_uri, "wb") as f_out:
        pickle.dump(res, file=f_out, protocol=pickle.HIGHEST_PROTOCOL)


def load_results_from_file(input_uri: str) -> ResultType:
    with open(input_uri, "rb") as f_in:
        return pickle.load(file=f_in)


def run():
    # NOTE: Models + data download link:
    # https://github.com/ulysses-camara/ulysses-segmenter/tree/master?tab=readme-ov-file#available-models
    # https://github.com/ulysses-camara/ulysses-segmenter/tree/master?tab=readme-ov-file#train-and-evaluation-data

    dt = datasets.load_from_disk("data/dataset_ulysses_segmenter_v2_active_learning_curated_only/test")
    dt = dt.to_dict()
    dt = utils.flatten_dict(dt)

    results_bert = validate_all_models(
        segmenter_cls=segmentador.BERTSegmenter,
        dt=dt,
        uri_models=[f"{layer_count}_layer_6000_vocab_size_bert_v2" for layer_count in reversed((2, 4, 6))],
        device="cuda:0",
        batch_size=64,
    )

    results_lstm = validate_all_models(
        segmenter_cls=segmentador.LSTMSegmenter,
        dt=dt,
        uri_models=[
            f"{hidden_layer_size}_hidden_dim_6000_vocab_size_1_layer_lstm_v2.pt" for hidden_layer_size in (128, 256, 512)
        ],
        moving_window_sizes=(128, 256, 512, 1024, 2048, 4096),
        uri_tokenizers="tokenizers/6000_subwords",
        device="cuda:0",
        batch_size=64,
    )

    print(len(results_bert), len(results_lstm))
    all_results = {**results_bert, **results_lstm}

    print(all_results)

    with open("all_results_inference.json", "w", encoding="utf-8") as f_out:
        json.dump(all_results, f_out)


if __name__ == "__main__":
    run()
