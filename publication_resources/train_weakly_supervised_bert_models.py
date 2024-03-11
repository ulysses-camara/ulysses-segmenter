import os

import numpy as np
import transformers
import torch
import torch.nn
import datasets
import sklearn.metrics

import segmentador


OUTPUT_DIR = "./weak_supervision_outputs"

USE_FP16 = False
NUM_TRAIN_EPOCHS = 4
NUM_HIDDEN_LAYERS = 4
GRAD_ACCUMULATION_STEPS = 16
VOCAB_SIZE = 6000


TRAINED_MODEL_SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    "segmenter_model_v2",
    f"{NUM_HIDDEN_LAYERS}_{VOCAB_SIZE}_layer_model",
)

TRAINER_STATE_SAVE_PATH = os.path.join(
    OUTPUT_DIR,
    "saved_trainer_states",
    f"{NUM_HIDDEN_LAYERS}_{VOCAB_SIZE}_layer_model",
)


def init_model_with_random_parameters():
    # NOTE: initializing w/ any BERT-like will do, since we are not using the pretrained parameters.
    return segmentador.Segmenter(
        device="cuda:0",
        uri_model="google-bert/bert-base-cased",
        init_from_pretrained_weights=False,
        uri_tokenizer=f"tokenizers/{VOCAB_SIZE}_subwords",
        num_hidden_layers=NUM_HIDDEN_LAYERS,
    )


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
        pp
        for (p, l) in zip(predictions, labels)
        for (pp, ll) in zip(p, l)
        if ll != invalid_label_index
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


def train(seg_model, df_tokenized_split):
    save_steps = int(df_tokenized_split["train"].num_rows / GRAD_ACCUMULATION_STEPS * 0.10)

    training_args = transformers.TrainingArguments(
        output_dir=os.path.join(OUTPUT_DIR, "segmenter_checkpoint", f"{NUM_HIDDEN_LAYERS}_{VOCAB_SIZE}_layer_model"),
        logging_dir=os.path.join(OUTPUT_DIR, "loggings", f"{NUM_HIDDEN_LAYERS}_{VOCAB_SIZE}_layer_model"),
        fp16=USE_FP16 and torch.cuda.is_available(),
        load_best_model_at_end=True,
        metric_for_best_model="eval_loss",
        greater_is_better=False,
        evaluation_strategy="steps",
        logging_strategy="steps",
        save_strategy="steps",
        save_steps=save_steps,
        eval_steps=save_steps,
        logging_steps=save_steps,
        save_total_limit=5,
        per_device_train_batch_size=8,
        per_device_eval_batch_size=8,
        gradient_accumulation_steps=GRAD_ACCUMULATION_STEPS,
        learning_rate=5e-5,
        max_grad_norm=1.0,
        num_train_epochs=NUM_TRAIN_EPOCHS,
        lr_scheduler_type="linear",
        warmup_ratio=0.1,
        weight_decay=0.0,
        report_to="all",
    )

    data_collator = transformers.DataCollatorForTokenClassification(
        seg_model.tokenizer,
        pad_to_multiple_of=8 if USE_FP16 else 1,
    )

    trainer = transformers.Trainer(
        model=seg_model.model,
        tokenizer=seg_model.tokenizer,
        args=training_args,
        data_collator=data_collator,
        train_dataset=df_tokenized_split["train"],
        eval_dataset=df_tokenized_split["eval"],
        compute_metrics=compute_metrics,
    )

    train_results = trainer.train()

    train_metrics = train_results.metrics
    trainer.log_metrics(split="all", metrics=train_metrics)
    trainer.save_metrics(split="all", metrics=train_metrics)

    trainer.save_model(TRAINED_MODEL_SAVE_PATH)
    trainer.save_state()


def run():
    seg_model = init_model_with_random_parameters()
    df_tokenized_split = datasets.DatasetDict.load_from_disk("data/df_tokenized_split_0_120000_6000")
    train(seg_model, df_tokenized_split)


if __name__ == "__main__":
    run()
