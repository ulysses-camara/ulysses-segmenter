import datasets
import transformers


def load_ground_truth_sentences(test_uri: str, tokenizer_uri: str, split="test", group_by_document: bool = False):
    df_test = datasets.DatasetDict.load_from_disk(test_uri)[split]
    tokenizer = transformers.AutoTokenizer.from_pretrained(tokenizer_uri)

    sentences = [[]]
    doc_start_id = tokenizer.vocab["[CLS]"]
    inst_ids = [0]

    for i, (inst_labels, inst_input_ids) in enumerate(zip(df_test["labels"], df_test["input_ids"])):
        for label, input_id in zip(inst_labels, inst_input_ids):
            if label == 1 or input_id == doc_start_id:
                sentences.append([])
                inst_ids.append(i)
            sentences[-1].append(input_id)

    sentences = tokenizer.batch_decode(sentences, skip_special_tokens=True)

    if not group_by_document:
        return sentences

    new_sentences = []
    prev_id = -1

    for sent, cur_id in zip(sentences, inst_ids, strict=True):
        if prev_id != cur_id:
            new_sentences.append([])
            prev_id = cur_id
        new_sentences[-1].append(sent)

    return new_sentences
