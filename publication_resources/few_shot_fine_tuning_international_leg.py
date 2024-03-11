import collections
import os
import json
import functools

import torch
import datasets
import numpy as np
import tqdm

import utils


def run_few_shot_finetuning(input_: dict[str, list[int]], lang: str, *, random_init: bool = False):
    if random_init:
        lang = f"{lang}-random"

    n = len(input_["labels"])
    rng = np.random.RandomState(19888012)

    output_dir = f"results/few_shot_fine_tuning_international_leg/{lang}"
    output_dir = os.path.abspath(output_dir)
    os.makedirs(output_dir, exist_ok=True)

    df_test_uri = os.path.abspath("data/dataset_ulysses_segmenter_v2_active_learning_curated_only/test")
    df_test_ptbr = datasets.Dataset.load_from_disk(df_test_uri)
    df_test_ptbr = df_test_ptbr.map(functools.partial(utils.fn_pad_and_truncate, max_length=1024))
    df_test_ptbr.set_format("torch")
    df_test_ptbr = df_test_ptbr.to_dict()

    (df_test_ptbr, _) = utils.split_train_test(df_test_ptbr, m=500, random_state=1056, shifts=1)
    assert len(df_test_ptbr["labels"]) == 500
    df_test_ptbr_flatten = utils.flatten_dict(df_test_ptbr)
    labels_orig = np.asarray(df_test_ptbr_flatten.pop("labels"))

    for k in [0, 10, 100]:
        cur_res = collections.defaultdict(list)
        pbar = tqdm.tqdm(np.arange(10 if k > 0 else 1))
        rng = np.random.RandomState(1835 + k)
        output_uri = os.path.join(output_dir, f"{lang}_{k}.json")

        if os.path.exists(output_uri):
            print(f"Found cached '{output_uri}' result, skipping.")
            continue

        for _ in pbar:
            (split_train, split_test) = utils.split_train_test(input_, m=k, random_state=rng.randint(1, 2**32 - 1), shifts=3)

            assert len(split_test["input_ids"]) >= n - 3 * k, (len(split_test), len(split_test["input_ids"]))
            assert len(split_train["input_ids"]) == k, (len(split_train), len(split_train["input_ids"]))

            input_flatten = utils.flatten_dict(split_test)
            labels = np.asarray(input_flatten.pop("labels"))

            finetuned_segmenter = utils.train("4_layer_6000_vocab_size_bert_v2", split_train, pbar, random_init=random_init)

            with torch.no_grad():
                logits_lang = finetuned_segmenter(input_flatten, return_logits=True, show_progress_bar=True).logits

            res_k_shot = utils.fn_compute_metrics(labels=labels, logits=logits_lang)
            for l, v in res_k_shot.items():
                cur_res[l].append(v)

            with torch.no_grad():
                logits_orig = finetuned_segmenter(df_test_ptbr_flatten, return_logits=True, show_progress_bar=True).logits

            res_orig = utils.fn_compute_metrics(labels=labels_orig, logits=logits_orig)
            for l, v in res_orig.items():
                cur_res[f"{l}_orig"].append(v)

            pbar.set_description(f"{cur_res['cls_1_f1'][-1]=:.6f}")

        with open(output_uri, "w", encoding="utf-8") as f_out:
            json.dump(cur_res, f_out)


def run():
    lang_to_dataset_map = {
        "french": "french_code_civil",
        "italian": "italian_legislative_decrees",
        "german": "german_laws_and_regulations",
        "english": "us_congressional_bills",
    }

    for lang, dt_name in sorted(lang_to_dataset_map.items()):
        dt = datasets.Dataset.load_from_disk(f"data/{dt_name}")

        print(f"{lang.capitalize()} dataset:")
        print(dt)

        run_few_shot_finetuning(dt, lang=lang)
        run_few_shot_finetuning(dt, lang=lang, random_init=True)


if __name__ == "__main__":
    run()
