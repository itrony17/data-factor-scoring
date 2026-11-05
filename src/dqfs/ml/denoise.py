"""Masked-cell denoising (TabMAE-style) training and signal extraction.

The trainer corrupts a fraction of observed cells, then trains a branch to
reconstruct them. After training:

* :func:`row_error_signals` scores every row by the mean per-column
  z-normalized reconstruction error of its corrupted cells - rows the
  ensemble reconstructs badly are statistically anomalous, i.e. suspect;
* :func:`permutation_importance` measures how much total reconstruction
  loss rises when a source column's values are shuffled - a per-column
  deep importance score used for ML-driven factor weighting.
"""

from __future__ import annotations

import copy
import logging
import time
from dataclasses import dataclass, field

import numpy as np

from dqfs.ml.network import Adam, BaseTabularNet, ForwardOutput, softmax
from dqfs.ml.tokenizer import TokenizedBatch

logger = logging.getLogger(__name__)


@dataclass
class MaskPair:
    """Boolean corruption masks for one batch."""

    num: np.ndarray
    cat: np.ndarray


@dataclass
class DenoiseReport:
    """Summary of a denoising training run."""

    history: list[float] = field(default_factory=list)
    final_loss: float = 0.0
    seconds: float = 0.0


# ---------------------------------------------------------------- masking
def make_mask(
    batch: TokenizedBatch, corruption_rate: float, rng: np.random.Generator
) -> MaskPair:
    """Sample corruption masks over observed cells.

    Every row with at least one observed cell gets at least one corrupted
    cell, so every row contributes to the loss and to row error signals.
    """
    m_num = (rng.random(batch.X_num.shape) < corruption_rate) & batch.M_num
    m_cat = (rng.random(batch.X_cat.shape) < corruption_rate) & batch.M_cat
    empty = ~(m_num.any(axis=1) | m_cat.any(axis=1))
    for r in np.where(empty)[0]:
        obs_num = np.where(batch.M_num[r])[0]
        obs_cat = np.where(batch.M_cat[r])[0]
        if obs_num.size and (not obs_cat.size or rng.random() < 0.5):
            m_num[r, obs_num[rng.integers(obs_num.size)]] = True
        elif obs_cat.size:
            m_cat[r, obs_cat[rng.integers(obs_cat.size)]] = True
    return MaskPair(num=m_num, cat=m_cat)


def slice_batch(batch: TokenizedBatch, idx: np.ndarray) -> TokenizedBatch:
    """Row-slice a tokenized batch (used for minibatching)."""
    return TokenizedBatch(
        X_num=batch.X_num[idx],
        M_num=batch.M_num[idx],
        X_cat=batch.X_cat[idx],
        M_cat=batch.M_cat[idx],
        numeric_columns=batch.numeric_columns,
        categorical_columns=batch.categorical_columns,
        vocab_sizes=batch.vocab_sizes,
    )


# ------------------------------------------------------------------ loss
def apply_input_mask(batch: TokenizedBatch, mask: MaskPair) -> TokenizedBatch:
    """Hide corrupted cells from the network input.

    Masked numeric cells become 0 (the same encoding as missing) and masked
    categorical cells fall back to the reserved 0 code, so the network must
    reconstruct them from the remaining context - a true denoising
    objective instead of a copy shortcut.
    """
    X_num = np.where(mask.num, 0.0, batch.X_num)
    X_cat = np.where(mask.cat, 0, batch.X_cat)
    return TokenizedBatch(
        X_num=X_num, M_num=batch.M_num, X_cat=X_cat, M_cat=batch.M_cat,
        numeric_columns=batch.numeric_columns,
        categorical_columns=batch.categorical_columns,
        vocab_sizes=batch.vocab_sizes,
    )


def compute_losses(
    out: ForwardOutput, batch: TokenizedBatch, mask: MaskPair
) -> tuple[float, float, float, np.ndarray, list[np.ndarray]]:
    """Masked reconstruction loss and gradients w.r.t. network outputs.

    Returns:
        ``(total, loss_num, loss_cat, d_pred_num, d_logits_cat)`` where the
        gradients are already normalized by the number of corrupted cells.
    """
    # numeric: MSE over corrupted observed cells
    err = out.pred_num - batch.X_num
    sel = mask.num
    k_num = int(sel.sum())
    if k_num:
        loss_num = float((err[sel] ** 2).mean())
        d_pred_num = np.zeros_like(out.pred_num)
        d_pred_num[sel] = 2.0 * err[sel] / k_num
    else:
        loss_num = 0.0
        d_pred_num = np.zeros_like(out.pred_num)

    # categorical: CE over corrupted observed cells, equal weight per column
    d_logits: list[np.ndarray] = []
    cat_losses: list[float] = []
    for j, logits in enumerate(out.logits_cat):
        sel_j = mask.cat[:, j]
        dj = np.zeros_like(logits)
        if sel_j.any():
            probs = softmax(logits[sel_j], axis=1)
            target = batch.X_cat[sel_j, j]
            rows = np.arange(len(target))
            nll = -np.log(probs[rows, target] + 1e-12)
            cat_losses.append(float(nll.mean()))
            dp = probs
            dp[rows, target] -= 1.0
            dj[sel_j] = dp / len(target)
        d_logits.append(dj)
    loss_cat = float(np.mean(cat_losses)) if cat_losses else 0.0
    return loss_num + loss_cat, loss_num, loss_cat, d_pred_num, d_logits


# --------------------------------------------------------------- training
def train_denoising(
    net: BaseTabularNet,
    batch: TokenizedBatch,
    epochs: int,
    batch_size: int,
    corruption_rate: float,
    lr: float = 3e-3,
    seed: int = 0,
) -> DenoiseReport:
    """Train ``net`` end-to-end with masked-cell reconstruction.

    Trunk, attention/step layers and reconstruction heads are all updated
    (the supervised target head is trained separately out-of-fold by the
    ensemble).
    """
    rng = np.random.default_rng(seed)
    opt = Adam(net.P, lr=lr)
    n = batch.n_rows
    report = DenoiseReport()
    started = time.perf_counter()
    for _epoch in range(epochs):
        order = rng.permutation(n)
        losses: list[float] = []
        for start in range(0, n, batch_size):
            idx = order[start:start + batch_size]
            sub = slice_batch(batch, idx)
            mask = make_mask(sub, corruption_rate, rng)
            sub_masked = apply_input_mask(sub, mask)
            out = net.forward(sub_masked, train=True)
            total, _, _, d_pred_num, d_logits = compute_losses(out, sub, mask)
            net.zero_grad()
            net.backward(out, d_pred_num, d_logits)
            opt.step(net.P, net.G)
            losses.append(total)
        report.history.append(float(np.mean(losses)))
    report.final_loss = report.history[-1] if report.history else 0.0
    report.seconds = time.perf_counter() - started
    logger.info(
        "denoising done: %d epochs, final loss %.4f (%.1fs)",
        epochs, report.final_loss, report.seconds,
    )
    return report


def _target_loss_grad(
    target_pred: np.ndarray, y: np.ndarray, binary: bool
) -> tuple[float, np.ndarray]:
    """Supervised loss value and gradient dL/d(target_pred)."""
    pred = target_pred.reshape(-1)
    if binary:
        z = np.clip(pred, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-z))
        loss = float(-(y * np.log(p + 1e-12) + (1 - y) * np.log(1 - p + 1e-12)).mean())
        grad = ((p - y) / len(y)).reshape(-1, 1)
        return loss, grad
    resid = pred - y
    loss = float((resid**2).mean())
    grad = (2.0 * resid / len(y)).reshape(-1, 1)
    return loss, grad


def train_target_head(
    net: BaseTabularNet,
    batch: TokenizedBatch,
    y: np.ndarray,
    train_idx: np.ndarray,
    epochs: int,
    lr: float,
    binary: bool,
    seed: int = 0,
) -> float:
    """Train ONLY the supervised target head (trunk frozen) on ``train_idx``.

    Returns:
        Final training loss.
    """
    head_params = {k: v for k, v in net.P.items() if k.startswith("head_target")}
    del seed  # full-batch head training is deterministic
    opt = Adam(head_params, lr=lr)
    loss = 0.0
    for _ in range(epochs):
        out = net.forward(batch, train=False)
        assert out.target_pred is not None
        loss, grad_full = _target_loss_grad(out.target_pred[train_idx], y[train_idx], binary)
        d_target = np.zeros_like(out.target_pred)
        d_target[train_idx] = grad_full
        net.zero_grad()
        zero_logits = [np.zeros_like(lg) for lg in out.logits_cat]
        net.backward(out, np.zeros_like(out.pred_num), zero_logits, d_target)
        opt.step(head_params, net.G)
    return loss


# --------------------------------------------------------------- signals
def _zscore_selected(err: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """Per-column z-score of cell errors over corrupted cells only."""
    z = np.zeros_like(err)
    for j in range(err.shape[1]):
        sel = mask[:, j]
        if not sel.any():
            continue
        vals = err[sel, j]
        mu, sd = float(vals.mean()), float(vals.std())
        z[sel, j] = (vals - mu) / (sd + 1e-9)
    return z


def rank_normalize(values: np.ndarray) -> np.ndarray:
    """Map values to uniform ranks in [0, 1] (robust to outliers)."""
    n = len(values)
    if n <= 1:
        return np.zeros(n)
    order = np.argsort(values, kind="stable")
    ranks = np.empty(n, dtype=np.float64)
    ranks[order] = np.arange(n, dtype=np.float64)
    return ranks / (n - 1)


def zscore(values: np.ndarray) -> np.ndarray:
    """Standardize a vector; constant input maps to zeros."""
    mu, sd = float(values.mean()), float(values.std())
    if sd < 1e-12:
        return np.zeros_like(values)
    return (values - mu) / sd


def row_error_signals(
    net: BaseTabularNet,
    batch: TokenizedBatch,
    corruption_rate: float,
    seed: int = 0,
) -> np.ndarray:
    """Per-row mean z-normalized reconstruction error, rank-normalized.

    Every corrupted cell contributes its z-scored error (numeric absolute
    error, categorical cross-entropy); a row's signal is the mean over its
    own corrupted cells, then ranks are mapped to [0, 1].
    """
    rng = np.random.default_rng(seed)
    mask = make_mask(batch, corruption_rate, rng)
    masked = apply_input_mask(batch, mask)
    out = net.forward(masked, train=False)

    z_num = _zscore_selected(np.abs(out.pred_num - batch.X_num), mask.num)
    z_cat = np.zeros((batch.n_rows, batch.n_cat))
    for j, logits in enumerate(out.logits_cat):
        sel_j = mask.cat[:, j]
        if not sel_j.any():
            continue
        probs = softmax(logits[sel_j], axis=1)
        target = batch.X_cat[sel_j, j]
        rows = np.arange(len(target))
        ce = -np.log(probs[rows, target] + 1e-12)
        z_cat[sel_j, j] = (ce - ce.mean()) / (ce.std() + 1e-9)

    all_z = np.concatenate([z_num, z_cat], axis=1)
    all_mask = np.concatenate([mask.num, mask.cat], axis=1)
    counts = all_mask.sum(axis=1)
    sums = (all_z * all_mask).sum(axis=1)
    row_mean = np.where(counts > 0, sums / np.maximum(counts, 1), 0.0)
    return rank_normalize(row_mean)


def permutation_importance(
    net: BaseTabularNet,
    tokenizer,  # FeatureTokenizer (avoids a circular import)
    df,
    batch: TokenizedBatch,
    corruption_rate: float,
    seed: int = 0,
    max_columns: int = 60,
) -> dict[str, float]:
    """Per-column deep importance via permutation of raw source columns.

    Returns:
        Mapping column -> importance in [0, 1], normalized by the maximum.
    """
    rng = np.random.default_rng(seed)
    mask = make_mask(batch, corruption_rate, rng)

    def total_loss(b: TokenizedBatch) -> float:
        out = net.forward(apply_input_mask(b, mask), train=False)
        total, _, _, _, _ = compute_losses(out, b, mask)
        return total

    base = total_loss(batch)
    columns = [c for c in batch.numeric_columns + batch.categorical_columns][:max_columns]
    raw: dict[str, float] = {}
    for col in columns:
        df_perm = df.copy()
        df_perm[col] = rng.permutation(df[col].to_numpy(copy=True))
        permuted = tokenizer.transform(df_perm)
        loss = total_loss(permuted)
        raw[col] = max((loss - base) / max(base, 1e-9), 0.0)
    top = max(raw.values(), default=0.0)
    if top <= 0:
        return {c: 0.0 for c in raw}
    return {c: v / top for c, v in raw.items()}


def copy_net(net: BaseTabularNet) -> BaseTabularNet:
    """Deep-copy a trained network (fresh Adam state via new optimizer)."""
    return copy.deepcopy(net)
