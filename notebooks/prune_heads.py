# Adapted from: https://github.com/huggingface/transformers/blob/main/examples/research_projects/bertology/run_bertology.py
import argparse
import logging
import os
from datetime import datetime

import transformers
import sklearn.metrics
import numpy as np
import torch
import torch.nn as nn
import tqdm.auto


os.makedirs("head_pruning", exist_ok=True)


logger = logging.getLogger("head_pruning/head_pruning_logs")
logger.setLevel(logging.INFO)


# f1_fn = torchmetrics.F1Score(num_classes=4, average=None)
def segment_cls_f1_fn(y_pred, y_true, invalid_ind: int = -100):
    y_pred = y_pred.ravel()
    y_true = y_true.ravel()

    non_pad_tokens = y_true != invalid_ind

    y_pred = y_pred[non_pad_tokens]
    y_true = y_true[non_pad_tokens]

    return sklearn.metrics.f1_score(y_pred, y_true, average=None)[1]


def entropy(p):
    """Compute the entropy of a probability distribution"""
    plogp = p * torch.log(p)
    plogp[p == 0] = 0
    return -plogp.sum(dim=-1)


def print_2d_tensor(tensor):
    """Print a 2D tensor"""
    logger.info("lv, h >\t" + "\t".join(f"{x + 1}" for x in range(len(tensor))))
    for row in range(len(tensor)):
        if tensor.dtype != torch.long:
            logger.info(
                f"layer {row + 1}:\t" + "\t".join(f"{x:.5f}" for x in tensor[row].cpu().data)
            )
        else:
            logger.info(f"layer {row + 1}:\t" + "\t".join(f"{x:d}" for x in tensor[row].cpu().data))


def compute_heads_importance(
    model,
    eval_dataloader,
    compute_entropy=True,
    compute_importance=True,
    head_mask=None,
    actually_pruned=False,
    device="cpu",
    dont_normalize_importance_by_layer=False,
    dont_normalize_global_importance=False,
    output_dir="head_pruning",
):
    """This method shows how to compute:
    - head attention entropy
    - head importance scores according to http://arxiv.org/abs/1905.10650
    """
    os.makedirs(output_dir, exist_ok=True)

    output_uri_a = os.path.join(output_dir, "attn_entropy.npy")
    output_uri_b = os.path.join(output_dir, "head_importance.npy")

    # Prepare our tensors
    n_layers, n_heads = model.config.num_hidden_layers, model.config.num_attention_heads
    head_importance = torch.zeros(n_layers, n_heads).to(device)
    attn_entropy = torch.zeros(n_layers, n_heads).to(device)

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

        if compute_entropy:
            for layer, attn in enumerate(all_attentions):
                masked_entropy = entropy(attn.detach()) * inputs[
                    "attention_mask"
                ].float().unsqueeze(1)
                attn_entropy[layer] += masked_entropy.sum(-1).sum(0).detach()

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
    attn_entropy /= tot_tokens
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
    np.save(output_uri_a, attn_entropy.detach().cpu().numpy())
    np.save(output_uri_b, head_importance.detach().cpu().numpy())

    logger.info("Attention entropies")
    print_2d_tensor(attn_entropy)
    logger.info("Head importance scores")
    print_2d_tensor(head_importance)
    logger.info("Head ranked by importance scores")
    head_ranks = torch.zeros(head_importance.numel(), dtype=torch.long, device=device)
    head_ranks[head_importance.view(-1).sort(descending=True)[1]] = torch.arange(
        head_importance.numel(), device=device
    )
    head_ranks = head_ranks.view_as(head_importance)
    print_2d_tensor(head_ranks)

    return attn_entropy, head_importance, preds, labels


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

    _, head_importance, preds, labels = compute_heads_importance(
        model,
        eval_dataloader,
        compute_entropy=False,
        device=device,
    )

    preds = np.argmax(preds, axis=-1)
    original_score = segment_cls_f1_fn(preds, labels)
    logger.info(
        "Pruning: original score: %f, threshold: %f",
        original_score,
        original_score * masking_threshold,
    )

    new_head_mask = torch.ones_like(head_importance)
    num_to_mask = max(1, int(new_head_mask.numel() * masking_amount))

    current_score = original_score
    while current_score >= original_score * masking_threshold:
        print(current_score, original_score)
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
        logger.info("Heads to mask: %s", str(current_heads_to_mask.tolist()))
        new_head_mask = new_head_mask.view(-1)
        new_head_mask[current_heads_to_mask] = 0.0
        new_head_mask = new_head_mask.view_as(head_mask)
        new_head_mask = new_head_mask.clone().detach()
        print_2d_tensor(new_head_mask)

        # Compute metric and head importance again
        _, head_importance, preds, labels = compute_heads_importance(
            model,
            eval_dataloader,
            compute_entropy=False,
            head_mask=new_head_mask,
            device=device,
        )
        preds = np.argmax(preds, axis=-1)
        current_score = segment_cls_f1_fn(preds, labels)
        logger.info(
            "Masking: current score: %f, remaining heads %d (%.1f percents)",
            current_score,
            new_head_mask.sum(),
            new_head_mask.sum() / new_head_mask.numel() * 100,
        )

    logger.info("Final head mask")
    print_2d_tensor(head_mask)
    np.save(output_uri, head_mask.detach().cpu().numpy())

    return head_mask


def prune_heads(model, eval_dataloader, head_mask, device: str = "cpu", enforce_equal_number_of_heads: bool = False):
    """This method shows how to prune head (remove heads weights) based on
    the head importance scores as described in Michel et al. (http://arxiv.org/abs/1905.10650)
    """
    # Try pruning and test time speedup
    # Pruning is like masking but we actually remove the masked weights
    before_time = datetime.now()
    _, _, preds, labels = compute_heads_importance(
        model,
        eval_dataloader,
        compute_entropy=False,
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
    _, _, preds, labels = compute_heads_importance(
        model,
        eval_dataloader,
        compute_entropy=False,
        compute_importance=False,
        head_mask=None,
        actually_pruned=True,
        device=device,
    )
    preds = np.argmax(preds, axis=-1)
    score_pruning = segment_cls_f1_fn(preds, labels)
    new_time = datetime.now() - before_time

    print(
        "Pruning: original num of params: %.2e, after pruning %.2e (%.1f percents)",
        original_num_params,
        pruned_num_params,
        pruned_num_params / original_num_params * 100,
    )
    print("Pruning: score with masking: %f score with pruning: %f", score_masking, score_pruning)
    print(
        "Pruning: speed ratio (new timing / original timing): %f percents",
        original_time / new_time * 100,
    )
