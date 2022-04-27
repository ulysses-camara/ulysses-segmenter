# Adapted from: https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py
import argparse
import os
from datetime import datetime

import transformers
import sklearn.metrics
import numpy as np
import torch
import torch.nn as nn
import tqdm.auto


os.makedirs("head_pruning", exist_ok=True)


def segment_cls_f1_fn(y_pred, y_true, invalid_ind: int = -100):
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()

    non_pad_tokens = y_true != invalid_ind

    y_pred = y_pred[non_pad_tokens]
    y_true = y_true[non_pad_tokens]

    return sklearn.metrics.f1_score(y_pred, y_true, average=None)[1]


def compute_heads_importance(
    model,
    eval_dataloader,
    compute_importance=True,
    head_mask=None,
    actually_pruned=False,
    device="cpu",
    dont_normalize_importance_by_layer=False,
    dont_normalize_global_importance=False,
    output_dir="head_pruning",
):
    """This method shows how to compute:
    - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    os.makedirs(output_dir, exist_ok=True)

    output_uri_b = os.path.join(output_dir, "head_importance.npy")

    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)

    if head_mask is None:
        head_mask = torch.ones(n_layers, n_heads).to(device)

    head_mask.requires_grad_(requires_grad=True)
    # If actually pruned attention multi-head, set head mask to None to avoid shape mismatch
    if actually_pruned:
        head_mask = None

    preds = None
    labels = None
    tot_tokens = 0.0

    for step, inputs in enumerate(tqdm.auto.tqdm(eval_dataloader, desc="Iteration")):
        for k, v in inputs.items():
            inputs[k] = v.to(device)

        # Do a forward pass (not with torch.no_grad() since we need gradients for importance score - see below)
        outputs = model(**inputs, head_mask=head_mask, output_attentions=True)
        loss, logits, all_attentions = (
            outputs[0],
            outputs[1],
            outputs[-1],
        )  # Loss and logits are the first, attention the last
        loss.backward()  # Backpropagate to populate the gradients in the head mask

        if compute_importance:
            head_importance += head_mask.grad.abs().detach()

        # Also store our logits/labels if we want to compute metrics afterwards
        if preds is None:
            preds = logits.detach().cpu().numpy()
            labels = inputs["labels"].detach().cpu().numpy()
        else:
            preds = np.append(preds, logits.detach().cpu().numpy(), axis=0)
            labels = np.append(labels, inputs["labels"].detach().cpu().numpy(), axis=0)

        tot_tokens += inputs["attention_mask"].float().detach().sum().data

    # Normalize
    head_importance /= tot_tokens
    # Layerwise importance normalization
    if not dont_normalize_importance_by_layer:
        exponent = 2
        norm_by_layer = torch.pow(torch.pow(head_importance, exponent).sum(-1), 1 / exponent)
        head_importance /= norm_by_layer.unsqueeze(-1) + 1e-20

    if not dont_normalize_global_importance:
        head_importance = (head_importance - head_importance.min()) / (
            head_importance.max() - head_importance.min()
        )

    # Print/save matrices
    np.save(output_uri_b, head_importance.detach().cpu().numpy())

    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel(), device=device
    )
    head_ranks = head_ranks.view_as(head_importance)

    return head_importance, preds, labels


def mask_heads(
    model,
    eval_dataloader,
    masking_threshold: float = 0.9,
    masking_amount: float = 0.1,
    output_dir: str = "head_pruning",
    device: str = "cpu",
):
    """This method shows how to mask head (set some heads to zero), to test the effect on the network,
    based on the head importance scores, as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    os.makedirs(output_dir, exist_ok=True)
    output_uri = os.path.join(output_dir, "head_mask.npy")

    head_importance, preds, labels = compute_heads_importance(
        model,
        eval_dataloader,
        device=device,
    )

    preds = np.argmax(preds, axis=-1)
    original_score = segment_cls_f1_fn(preds, labels)

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * masking_amount))

    current_score = original_score
    while current_score >= original_score * masking_threshold:
        print(f"{current_score=:.4f}, {original_score=:.4f}")
        head_mask = new_head_mask.clone()  # save current head mask
        # heads from least important to most - keep only not-masked heads
        head_importance[head_mask == 0.0] = float("Inf")
        current_heads_to_mask = head_importance.view(-1).sort()[1]

        if len(current_heads_to_mask) <= num_to_mask:
            break

        new_head_mask.requires_grad = False
        current_heads_to_mask.requires_grad = False

        # mask heads
        current_heads_to_mask = current_heads_to_mask[:num_to_mask]
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        new_head_mask = new_head_mask.clone().detach()

        # Compute metric and head importance again
        head_importance, preds, labels = compute_heads_importance(
            model,
            eval_dataloader,
            head_mask=new_head_mask,
            device=device,
        )
        preds = np.argmax(preds, axis=-1)
        current_score = segment_cls_f1_fn(preds, labels)

    np.save(output_uri, head_mask.detach().cpu().numpy())

    return head_mask


def prune_heads(
    model,
    eval_dataloader,
    head_mask,
    device: str = "cpu",
    enforce_equal_number_of_heads: bool = False,
):
    """This method shows how to prune head (remove heads weights) based on
    the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, preds, labels = compute_heads_importance(
        model,
        eval_dataloader,
        compute_importance=False,
        head_mask=head_mask,
        device=device,
    )
    preds = np.argmax(preds, axis=-1)
    score_masking = segment_cls_f1_fn(preds, labels)
    original_time = datetime.now() - before_time

    original_num_params = sum(p.numel() for p in model.parameters())
    heads_to_prune = dict(
        (layer, (1 - head_mask[layer].long()).nonzero().squeeze().tolist())
        for layer in range(len(head_mask))
    )

    assert sum(len(h) for h in heads_to_prune.values()) == (1 - head_mask.long()).sum().item()

    if enforce_equal_number_of_heads:
        min_head_count = min(map(len, heads_to_prune.values()))
        for layer, head_ids in heads_to_prune.items():
            if len(head_ids) > min_head_count:
                heads_to_prune[layer] = head_ids[:min_head_count]

    model.prune_heads(heads_to_prune)

    pruned_num_params = sum(p.numel() for p in model.parameters())

    before_time = datetime.now()
    _, preds, labels = compute_heads_importance(
        model,
        eval_dataloader,
        compute_importance=False,
        head_mask=None,
        actually_pruned=True,
        device=device,
    )
    preds = np.argmax(preds, axis=-1)
    score_pruning = segment_cls_f1_fn(preds, labels)
    new_time = datetime.now() - before_time

    print(
        f"Pruning: original num of params: {original_num_params:.2e}, "
        f"after pruning {pruned_num_params:.2e} "
        f"({pruned_num_params / original_num_params * 100:.1f}%)"
    )
    print(
        f"Pruning: score with masking: {score_masking:.4f} "
        f"score with pruning: {score_pruning:.4f}"
    )
    print(
        "Pruning: speed ratio (new timing / original timing): "
        f"{original_time / new_time * 100:.2f}%"
    )
