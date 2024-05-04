import typing as t
import functools

import torch
import torch.nn
import tqdm


def text_to_ids(
    segments: t.List[t.List[str]],
    tokenizer,
) -> t.Tuple[t.List[t.List[int]], t.List[t.List[int]]]:
    input_ids: t.List[t.List[int]] = []
    labels: t.List[t.List[int]] = []

    for doc_segs in segments:
        for j, seg in enumerate(doc_segs):
            if j == 0: seg = f"{tokenizer.pad_token} {seg}"
            if j == len(doc_segs) - 1: seg = f"{seg} {tokenizer.sep_token}"

            cur_tokens = tokenizer.tokenize(seg, add_special_tokens=False)

            cur_input_ids = tokenizer.convert_tokens_to_ids(cur_tokens)

            cur_labels = [-100 if tok.startswith("##") else 0 for tok in cur_tokens]
            cur_labels[0] = 1

            if j == 0:
                cur_labels[0] = -100
                cur_labels[1] = 1

            if j == len(doc_segs) - 1:
                cur_labels[-1] = -100

            input_ids.append(cur_input_ids)
            labels.append(cur_labels)

    return (input_ids, labels)


def ids_to_insts(
    seg_input_ids: t.List[int],
    seg_labels: t.List[int],
    inst_length: int,
    pad_id: int,
) -> t.Tuple[torch.Tensor, torch.Tensor]:
    all_input_ids: t.List[torch.Tensor] = [[]]
    all_labels: t.List[torch.Tensor] = [[]]

    for inst, labs in zip(seg_input_ids, seg_labels):
        left_size = inst_length - len(all_input_ids[-1])

        (left_inst, right_inst) = (inst[:left_size], inst[left_size:])
        (left_labs, right_labs) = (labs[:left_size], labs[left_size:])

        all_input_ids[-1].extend(left_inst)
        all_labels[-1].extend(left_labs)

        if right_inst:
            all_input_ids.append(right_inst)
            all_labels.append(right_labs)

    half_length = inst_length // 2

    for i in range(len(all_input_ids) - 1):
        left_ids, right_ids = (all_input_ids[i][half_length:], all_input_ids[i + 1][:half_length])
        left_labs, right_labs = (all_labels[i][half_length:], all_labels[i + 1][:half_length])
        all_input_ids.append(left_ids + right_ids)
        all_labels.append(left_labs + right_labs)

    for ids, labs in zip(all_input_ids, all_labels):
        if len(ids) < inst_length:
            ids.extend((inst_length - len(ids)) * [pad_id])
            labs.extend((inst_length - len(labs)) * [-100])

    all_input_ids = torch.vstack([torch.Tensor(item) for item in all_input_ids])
    all_labels = torch.vstack([torch.Tensor(item) for item in all_labels])

    all_input_ids = all_input_ids.long()
    all_labels = all_labels.long()

    return (all_input_ids, all_labels)



def finetune(
    model: torch.nn.Module,
    tokenizer,
    segments: t.List[t.List[str]],
    is_complete_input: bool,
    *,
    lr: int = 1e-4,
    max_epochs: int = 10,
    batch_size: int = 3,
    grad_acc_its: int = 1,
    device: str | torch.device = "cuda:0",
    inst_length: int = 1024,
    show_progress_bar: bool = True,
    focus_on_misclassifications: bool = False,
    early_stopping_accuracy_threshold: float = 1.0,
):
    if len(segments) == 0:
        return model

    if isinstance(segments[0], str):
        segments = [segments]

    (seg_input_ids, seg_labels) = text_to_ids(segments=segments, tokenizer=tokenizer)

    (input_ids, labels) = ids_to_insts(
        seg_input_ids=seg_input_ids,
        seg_labels=seg_labels,
        inst_length=inst_length,
        pad_id=tokenizer.pad_token_id,
    )

    assert len(input_ids) == len(labels)

    n = len(input_ids)

    loss_fn = torch.nn.CrossEntropyLoss(ignore_index=-100, reduction="none")
    optim = torch.optim.Adam(model.parameters(), lr=lr)

    indices = torch.arange(n)
    weights = torch.ones(n, dtype=torch.double)

    dl_train = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(indices, weights),
        shuffle=True,
        batch_size=batch_size,
        drop_last=False,
    )

    model.train()
    model.to(device)

    for i in range(1, 1 + max_epochs):
        mov_loss = None
        optim.zero_grad()
        pbar = tqdm.tqdm(dl_train, disable=not show_progress_bar)
        total_misclass = 0
        total_insts = 0

        for j, (batch_inds, batch_weight) in enumerate(pbar, 1):
            batch_inds = batch_inds.squeeze()
            batch_inds = torch.atleast_1d(batch_inds)

            cur_input_ids = input_ids[batch_inds, :]
            batch_weight = batch_weight.to(device)

            batch = {"input_ids": cur_input_ids}

            if is_complete_input:
                batch["token_type_ids"] = torch.zeros_like(cur_input_ids)
                batch["attention_mask"] = torch.ones_like(cur_input_ids)

            batch = {k: torch.atleast_2d(v).to(device) for k, v in batch.items()}

            y_true = labels[batch_inds, :].to(device)
            y_true = y_true.view(-1)

            output = model(**batch)
            logits = output["logits"].view(-1, 4)

            assert logits.shape[0] == y_true.numel(), (logits.shape, y_true.shape)

            loss = loss_fn(logits, y_true)
            loss = loss.reshape(-1, inst_length).mean(axis=-1)
            loss = (loss * batch_weight).sum() / (batch_weight.sum() * grad_acc_its)
            loss.backward()

            if j % grad_acc_its == 0 or j == n:
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optim.step()
                optim.zero_grad()

            with torch.no_grad():
                y_preds = logits.detach().argmax(axis=-1)
                assert y_preds.shape == y_true.shape
                insts_with_misclass = ((y_preds != y_true) * (y_true != -100)).view(-1, inst_length)
                insts_with_misclass = insts_with_misclass.any(axis=-1)
                assert insts_with_misclass.ndim == 1 and int(insts_with_misclass.numel()) == len(batch_inds)
                insts_with_misclass = torch.nonzero(insts_with_misclass)
                insts_with_misclass = insts_with_misclass.cpu()
                insts_with_misclass = batch_inds[insts_with_misclass]

                if focus_on_misclassifications:
                    weights[insts_with_misclass] += 1.0

                total_misclass += int(insts_with_misclass.numel())
                total_insts += int(batch_inds.numel())

            accuracy = 1.0 - total_misclass / total_insts
            pbar.set_description(f"Epoch: {i} of {max_epochs}. Acc.: {100.0 * accuracy:.2f}%")

        if accuracy >= early_stopping_accuracy_threshold:
            break

    model.eval()
    return model
