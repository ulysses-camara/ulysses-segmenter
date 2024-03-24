import subprocess
import os

import bs4
import datasets
import pandas as pd

import eval_models


def test(*, output_xml: str, to_lower: bool):
    parser = bs4.BeautifulSoup(output_xml, "lxml")
    segs = [item.text.strip() for item in parser.find_all("text")]

    assert len(segs)
    print("Test segments:", len(segs))

    # NOTE: Segmenter download link available in:
    # https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#available-models
    segs_true = eval_models.load_ground_truth_sentences(
        test_uri="data/dataset_ulysses_segmenter_v2_active_learning_curated_only",
        tokenizer_uri="tokenizers/6000_subwords",
        group_by_document=False,
    )

    if to_lower:
        segs_true = [str(item).lower() for item in segs_true]
    
    res = eval_models.estimated_test_perf(sentences_pred=segs, sentences_true=segs_true)
    print(res)
    print("Precision (%):", round(100.0 * res["precision"], 3))
    print("Recall    (%):", round(100.0 * res["recall"], 3))
    print("F1        (%):", round(100.0 * 2.0 * (res["precision"] * res["recall"]) / (res["precision"] + res["recall"]), 3))
    print("GeoAvg    (%):", round(100.0 * (res["precision"] * res["recall"]) ** 0.50, 3))


def convert_tsv_data_to_topictiling_format(*, data_uri: str, split: str, to_lower: bool):
    assert split in {"train", "test"}

    df = pd.read_csv(
        data_uri,
        index_col=None,
        sep="\t",
        usecols=["document_id", "text"] if split == "test" else ["text"],
    )
    df = df.dropna()

    if split == "test":
        df = df.groupby("document_id").agg(lambda x: " ".join(x))

    df = df.squeeze().tolist()
    data_urn = os.path.basename(os.path.dirname(data_uri.rstrip("/")))

    assert len(df)

    os.makedirs("topictiling_data", exist_ok=True)
    suffix = "lower" if to_lower else "normal"
    output_uri = f"topictiling_data/input_{data_urn}_{split}_{suffix}.txt"
    
    with open(output_uri, "w") as f_out:
        f_out.write(f"{len(df)}\n")
        f_out.write("\n".join(df))

    return output_uri


def train(train_input_uri: str) -> str:
    # NOTE: Download URL for TopicTiling
    # https://github.com/riedlma/topictiling

    # NOTE: Download URLs to the Gibbs sampler (also extract/compile):
    # Java : https://jgibblda.sourceforge.net/
    # C/C++: https://gibbslda.sourceforge.net/

    # NOTE: Download URLs for train/test data:
    # https://github.com/ulysses-camara/ulysses-segmenter?tab=readme-ov-file#train-and-evaluation-data

    # Training
    print("Train input URI:", train_input_uri)
    subprocess.run(" ".join([
        "./GibbsLDA++-0.2/src/lda",
        "-est",  #  Estimate the LDA model from scratch.
        "-alpha 0.50",  # 50 / n_topics 
        "-beta 0.01",  # As per paper.
        "-ntopics 100",  # As per paper.
        "-savestep 90000000",  # Don't save intermediate models.
        "-niters 500",
        f"-dfile '{train_input_uri}'",
    ]), check=True, shell=True)


def inference(test_input_uri: str):
    url, urn = os.path.split(test_input_uri)
    url = os.path.abspath(url)
    data_dir = os.path.abspath("./topictiling_data")

    print("Test input URI:", test_input_uri)
    # NOTE: unfortunately, the original package name contains a dot, 'topictiling_v1.0',
    # which causes problem when specifying the path to Java. I had to rename it to
    # 'topictiling_v1' in order to be able to execute the comman below.
    output_xml = subprocess.run(" ".join([
        "java",
        "-Xmx10G",
        "-cp $(echo topictiling_v1/dependency/*jar| tr ' ' ':'):topictiling_v1/de.tudarmstadt.langtech.semantics.segmentation.topictiling-0.0.2.jar",
        "de.tudarmstadt.langtech.semantics.segmentation.segmenter.RunTopicTilingOnFile",
        "-i 100",  # Inference iterations.
        "-ri 5",  # Repeated inferences.
        f"-tmd {data_dir}",
        "-tmn model-final",
        f"-fd {url}",
        f"-fp {urn}",
    ]), capture_output=True, text=True, check=True, shell=True).stdout

    assert output_xml

    return output_xml


if __name__ == "__main__":
    """Results obtained:

    dataset_name='dataset_ulysses_segmenter_v2_active_learning_curated_only_tsv' to_lower=True
    {'precision': 0.006626506024096385, 'recall': 0.0015015015015015015}
    Precision (%): 0.663
    Recall    (%): 0.15
    F1        (%): 0.245
    GeoAvg    (%): 0.315

    dataset_name='dataset_ulysses_segmenter_v2_active_learning_curated_only_tsv' to_lower=False
    {'precision': 0.32585011554968635, 'recall': 0.06736281736281736}
    Precision (%): 32.585
    Recall    (%): 6.736
    F1        (%): 11.165
    GeoAvg    (%): 14.816

    dataset_name='dataset_ulysses_segmenter_v1_weak_supervision_tsv' to_lower=False
    {'precision': 0.3129404228058937, 'recall': 0.06668031668031668}
    Precision (%): 31.294
    Recall    (%): 6.668
    F1        (%): 10.994
    GeoAvg    (%): 14.445
    """
    dataset_names = [
        "dataset_ulysses_segmenter_v2_active_learning_curated_only_tsv",
        "dataset_ulysses_segmenter_v1_weak_supervision_tsv",
    ]

    for dataset_name in dataset_names:
        for to_lower in (True, False):
            print(f"{dataset_name=} {to_lower=}")

            train_input_uri = convert_tsv_data_to_topictiling_format(
                data_uri=f"data/{dataset_name}/train.tsv",
                split="train",
                to_lower=to_lower,
            )

            test_input_uri = convert_tsv_data_to_topictiling_format(
                data_uri=f"data/dataset_ulysses_segmenter_v2_active_learning_curated_only_tsv/test.tsv",
                split="test",
                to_lower=to_lower,
            )

            train(train_input_uri)
            output_xml = inference(test_input_uri)
            test(output_xml=output_xml, to_lower=to_lower)
