import collections
import itertools
import json
import os

import matplotlib.pyplot as plt
import seaborn as sns
import datasets
import torch
import pandas as pd
import regex as re
import numpy as np
import sklearn.model_selection
import segmentador
import tqdm

import utils


STATES = {
    "São Paulo": "SP",
    "Minas Gerais": "MG",
    "Espírito Santo": "ES",
    "Rio de Janeiro": "RJ",
    "Rio Grande do Sul": "RS",
    "Santa Catarina": "SC",
    "Paraná": "PR",
    "Goiás": "GO",
    "Mato Grosso": "MT",
    "Mato Grosso do Sul": "MS",
    "Bahia": "BA",
    "Pernambuco": "PE",
    "Ceará": "CE",
    "Paraíba": "PB",
    "Rio Grande do Norte": "RN",
    "Maranhão": "MA",
    "Alagoas": "AL",
    "Sergipe": "SE",
    "Piauí": "PI",
    "Acre": "AC",
    "Amazonas": "AM",
    "Amapá": "AP",
    "Roraima": "RR",
    "Rondônia": "RO",
    "Pará": "PA",
    "Tocantins": "TO",
}


assert len(STATES) == 26


def load():
    reg_states = re.compile("(?<=ESTADO D[AEO] )" + "(?:" + "|".join(sorted(STATES.keys())).upper() + ")")

    tsv = pd.read_csv("data/dataset_ulysses_segmenter_train_v3.tsv", sep="\t", index_col=0)
    tsv = tsv.groupby("document_id").agg(lambda x: " ".join(x).upper())

    dist_by_state = collections.Counter()
    indices_by_state = collections.defaultdict(list)

    for i, (content,) in tsv.iterrows():
        detected_states = reg_states.findall(content)
        if not detected_states:
            continue
        state, freqs = np.unique(detected_states, return_counts=True)

        if state.size > 1 and freqs[0] == freqs[1]:
            continue

        dist_by_state[state[0]] += 1
        indices_by_state[state[0]].append(i)

    selected_states = {k for k, v in dist_by_state.most_common(100) if v >= 19}
    indices_by_state = {k: v for k, v in indices_by_state.items() if k in selected_states}

    dt = datasets.Dataset.load_from_disk("data/dataset_ulysses_segmenter_train_v3")

    return (dt, indices_by_state)


def check_misclass_quantiles(dt, indices_by_state):
    test_inference_kwargs = {
        "return_logits": True,
        "show_progress_bar": True,
        "window_shift_size": 1.0,
        "moving_window_size": 1024,
        "apply_postprocessing": False,
        "batch_size": 16,
    }

    segmenter = segmentador.BERTSegmenter(uri_model="4_layer_6000_vocab_size_bert_v2", device="cuda:0")
    misclass_fracs = []

    for inds in tqdm.tqdm(indices_by_state.values()):
        for i in inds[:20]:
            dt_cur = dt.select([i]).to_dict()
            labels_true = np.asarray(dt_cur.pop("labels")[0])
            labels_pred = segmenter(dt_cur, **test_inference_kwargs, return_labels=True).labels
            assert labels_true.shape == labels_pred.shape
            labels_pred = labels_pred[labels_true != -100]
            labels_true = labels_true[labels_true != -100]
            assert labels_pred.size == labels_true.size
            misclass_inds_frac = np.flatnonzero(labels_pred != labels_true) / labels_true.size
            misclass_fracs.extend(misclass_inds_frac.tolist())

    fig, ax = plt.subplots(1, figsize=(10, 5), layout="tight")
    sns.histplot(y=misclass_fracs, bins=50, ax=ax, stat="probability", kde=False)
    ax.set_ylabel("Misclassification relative location (w.r.t. instance size)")
    sns.despine(ax=ax)
    fig.savefig("misclass_state_leg_dist.pdf", format="pdf", bbox_inches="tight")
    # plt.show()


def undersample_inds(inds_by_class, rng, m: int):
    return sorted({k: rng.choice(v, size=m, replace=False) for k, v in inds_by_class.items()}.items())


def compute_undersampling_stats(dt, indices_by_state, m):
    rng = np.random.RandomState(1485)
    token_counts = []

    for _ in tqdm.tqdm(np.arange(100)):
        undersampled_inds = undersample_inds(indices_by_state, rng, m=m)
        token_count_per_state = [list(itertools.chain(map(len, dt.select(inds)["input_ids"]))) for _, inds in undersampled_inds]
        token_counts.extend(list(itertools.chain(*token_count_per_state)))
        print(sum(list(itertools.chain(*token_count_per_state))))

    print("length :", len(token_counts))
    print("avg len:", round(np.mean(token_counts), 2))
    print("std len:", round(np.std(token_counts), 2))


def kfold_with_undersampling(dt, indices_by_state):
    lens_by_state = {k: len(v) for k, v in indices_by_state.items()}
    m = int(min(lens_by_state.values()))
    rng = np.random.RandomState(1485)

    pbar = tqdm.tqdm(range(10))

    output_dir = "results/state_legislation_finetuning/first_experiment__per_state"
    output_name = f"test_results_state_leg_{m}_inst_per_state.json"
    output_name = os.path.join(output_dir, output_name)

    all_res = collections.defaultdict(list)

    if os.path.exists(output_name):
        with open(output_name, "r", encoding="utf-8") as f_in:
            all_res.update(json.load(f_in))

        assert all_res

    test_inference_kwargs = {
        "return_logits": True,
        "show_progress_bar": True,
        "window_shift_size": 1.0,
        "moving_window_size": 1024,
        "apply_postprocessing": False,
        "batch_size": 16,
    }

    def compute_metrics_(
        labels, logits, group_ids: list[int], sorted_state_names: list[str], prefix: str, all_res: dict[str, list[float]]
    ) -> None:
        assert len(labels) == len(logits) == len(group_ids), (len(labels), len(logits), len(group_ids))
        assert len(sorted_state_names) == len(indices_by_state)

        metrics_micro = utils.fn_compute_metrics(labels=labels, logits=logits)

        for l, v in metrics_micro.items():
            all_res[f"{prefix}{l}"].append(v)

        group_ids = np.asarray(group_ids)

        for i, state in enumerate(sorted_state_names):
            cur_inst_ids = np.flatnonzero(group_ids == i)
            cur_labels = labels[cur_inst_ids]
            cur_logits = logits[cur_inst_ids, :]
            assert len(cur_labels) == len(cur_logits) == len(cur_inst_ids)
            metrics_state = utils.fn_compute_metrics(labels=cur_labels, logits=cur_logits)
            for l, v in metrics_state.items():
                all_res[f"{prefix}{l}@{state}"].append(v)

    for repetition_id in pbar:
        undersampled_state_to_inds = undersample_inds(indices_by_state, rng=rng, m=m)
        sorted_state_names = [state_name for state_name, _ in undersampled_state_to_inds]

        assert isinstance(undersampled_state_to_inds, list)
        assert len(sorted_state_names) == len(indices_by_state)

        splitter = sklearn.model_selection.KFold(n_splits=5, shuffle=True, random_state=rng.randint(1, 2**32 - 1))

        if repetition_id < len(all_res["after_macro_f1"]) // 5:
            print(f"Skipped {repetition_id=}.")
            continue

        for inds_train_aux, inds_test_aux in splitter.split(range(m)):
            inds_train = list(
                itertools.chain(*[inds_orig[inds_train_aux].tolist() for (_, inds_orig) in undersampled_state_to_inds])
            )
            inds_test = list(
                itertools.chain(*[inds_orig[inds_test_aux].tolist() for (_, inds_orig) in undersampled_state_to_inds])
            )

            test_group_ids = list(itertools.chain(*[len(inds_test_aux) * [i] for i in range(len(indices_by_state))]))

            assert len(inds_train) == len(set(inds_train))
            assert len(inds_test) == len(set(inds_test))
            assert set(inds_test).isdisjoint(inds_train)
            assert len(inds_test) == len(test_group_ids)

            split_train = dt.select(inds_train)
            split_test = dt.select(inds_test).to_dict()
            (split_test_flatten, test_group_ids) = utils.flatten_dict(split_test, group_ids=test_group_ids)
            test_labels = np.asarray(split_test_flatten.pop("labels"))

            assert len(test_group_ids) == len(test_labels)
            assert len(split_train["input_ids"]) >= len(split_test["input_ids"]), (len(split_train), len(split_test))
            assert len(split_test["labels"]) == len(inds_test_aux) * len(indices_by_state)

            seg_model = segmentador.BERTSegmenter(uri_model="4_layer_6000_vocab_size_bert_v2", device="cuda:0")

            with torch.no_grad():
                seg_model.eval()
                test_logits_orig = seg_model(split_test_flatten, **test_inference_kwargs).logits

            compute_metrics_(
                labels=test_labels,
                logits=test_logits_orig,
                sorted_state_names=sorted_state_names,
                group_ids=test_group_ids,
                prefix="before_",
                all_res=all_res,
            )

            finetuned_segmenter = utils.train(segmenter_name=seg_model, split_train=split_train, pbar=pbar)
            with torch.no_grad():
                seg_model.eval()
                test_logits = finetuned_segmenter(split_test_flatten, **test_inference_kwargs).logits

            compute_metrics_(
                labels=test_labels,
                logits=test_logits,
                sorted_state_names=sorted_state_names,
                group_ids=test_group_ids,
                prefix="after_",
                all_res=all_res,
            )

        with open(output_name, "w", encoding="utf-8") as f_out:
            json.dump(all_res, f_out)


def kfold(dt, random_init: bool = False):
    rng = np.random.RandomState(148521)
    output_dir = "results/state_legislation_finetuning/second_experiment__all_extra_data"

    if random_init:
        output_name = "test_results_state_leg_all_data_random_init.json"
    else:
        output_name = "test_results_state_leg_all_data.json"

    output_name = os.path.join(output_dir, output_name)
    print(output_name)

    all_res = collections.defaultdict(list)

    if os.path.exists(output_name):
        with open(output_name, "r", encoding="utf-8") as f_in:
            all_res.update(json.load(f_in))

        assert all_res

    test_inference_kwargs = {
        "return_logits": True,
        "show_progress_bar": True,
        "window_shift_size": 1.0,
        "moving_window_size": 1024,
        "apply_postprocessing": False,
        "batch_size": 16,
    }

    def compute_metrics_(labels, logits, prefix: str, all_res: dict[str, list[float]]) -> None:
        assert len(labels) == len(logits), (len(labels), len(logits))
        metrics_micro = utils.fn_compute_metrics(labels=labels, logits=logits)
        for l, v in metrics_micro.items():
            all_res[f"{prefix}{l}"].append(v)

    n_splits = 5
    splitter = sklearn.model_selection.KFold(n_splits=n_splits, shuffle=True, random_state=rng.randint(1, 2**32 - 1))
    pbar = tqdm.tqdm(splitter.split(dt), total=n_splits)

    for inds_train, inds_test in pbar:
        assert len(inds_train) == len(set(inds_train))
        assert len(inds_test) == len(set(inds_test))
        assert set(inds_test).isdisjoint(inds_train)

        split_train = dt.select(inds_train)
        split_test = dt.select(inds_test).to_dict()

        split_test_flatten = utils.flatten_dict(split_test)
        test_labels = np.asarray(split_test_flatten.pop("labels"))

        seg_model = segmentador.BERTSegmenter(
            uri_model="4_layer_6000_vocab_size_bert_v2",
            device="cuda:0",
            init_from_pretrained_weights=not random_init,
        )

        with torch.no_grad():
            seg_model.eval()
            test_logits_orig = seg_model(split_test_flatten, **test_inference_kwargs).logits

        compute_metrics_(
            labels=test_labels,
            logits=test_logits_orig,
            prefix="before_",
            all_res=all_res,
        )

        finetuned_segmenter = utils.train(segmenter_name=seg_model, split_train=split_train, pbar=pbar, n_epochs=5)

        with torch.no_grad():
            seg_model.eval()
            test_logits = finetuned_segmenter(split_test_flatten, **test_inference_kwargs).logits

        compute_metrics_(
            labels=test_labels,
            logits=test_logits,
            prefix="after_",
            all_res=all_res,
        )

        with open(output_name, "w", encoding="utf-8") as f_out:
            json.dump(all_res, f_out)


def run():
    dt, indices_by_state = load()
    check_misclass_quantiles(dt, indices_by_state)
    kfold_with_undersampling(dt, indices_by_state)
    kfold(dt)
    kfold(dt, random_init=True)


if __name__ == "__main__":
    run()
