"""Tests for dqfs.ml.network - shapes, training, and GRADIENT CHECKS.

The finite-difference gradient check is the keystone: it verifies the
hand-derived backpropagation of both branches (attention path, residual
blocks, LayerNorm, embeddings, heads) against numerical partial
derivatives of the exact loss the trainer uses.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from dqfs.ml.denoise import compute_losses, make_mask, train_denoising
from dqfs.ml.network import (
    Adam,
    AttentiveTabularNet,
    ResidualMlpNet,
    gelu,
    gelu_grad,
    softmax,
)
from dqfs.ml.tokenizer import FeatureTokenizer


@pytest.fixture()
def batch() -> object:
    rng = np.random.default_rng(7)
    df = pd.DataFrame(
        {
            "a": rng.normal(size=12),
            "b": rng.normal(size=12) * 3 + 1,
            "c": rng.choice(["x", "y", "z"], size=12),
        }
    )
    df.loc[3, "a"] = np.nan
    df.loc[5, "c"] = None
    tok = FeatureTokenizer(numeric=["a", "b"], categorical=["c"])
    return tok.fit_transform(df)


# --------------------------------------------------------------- units
def test_gelu_matches_grad_numerically() -> None:
    x = np.linspace(-3, 3, 41)
    eps = 1e-6
    num = (gelu(x + eps) - gelu(x - eps)) / (2 * eps)
    assert np.allclose(num, gelu_grad(x), atol=1e-6)


def test_softmax_sums_to_one() -> None:
    x = np.random.default_rng(0).normal(size=(5, 7)) * 10
    s = softmax(x, axis=1)
    assert np.allclose(s.sum(axis=1), 1.0)
    assert (s >= 0).all()


def test_adam_minimizes_quadratic() -> None:
    params = {"w": np.array([5.0])}
    opt = Adam(params, lr=0.1)
    for _ in range(200):
        opt.step(params, {"w": 2 * params["w"]})
    assert abs(params["w"][0]) < 1e-2


def test_attention_masks_are_probability_distributions(batch) -> None:
    net = AttentiveTabularNet(batch, hidden=8, steps=3, emb_dim=4, seed=1)
    out = net.forward(batch)
    # re-derive masks via a fresh forward inspecting cache
    for sc in out.cache["steps"]:
        assert np.allclose(sc["m"].sum(axis=1), 1.0, atol=1e-9)
        assert (sc["m"] >= 0).all()


def test_forward_output_shapes(batch) -> None:
    for Net in (AttentiveTabularNet, ResidualMlpNet):
        net = Net(batch, hidden=8, emb_dim=4, with_target=True, seed=2)
        out = net.forward(batch)
        assert out.pred_num.shape == (12, 2)
        assert out.logits_cat[0].shape == (12, 4)  # 3 categories + reserved
        assert out.target_pred is not None and out.target_pred.shape == (12, 1)
        assert out.repr.shape == (12, 8)


def test_mlp_dropout_active_only_in_train(batch) -> None:
    net = ResidualMlpNet(batch, hidden=8, dropout=0.5, seed=3)
    out_eval1 = net.forward(batch, train=False)
    out_eval2 = net.forward(batch, train=False)
    assert np.array_equal(out_eval1.repr, out_eval2.repr)
    out_tr = net.forward(batch, train=True)
    assert not np.array_equal(out_eval1.repr, out_tr.repr)


# ------------------------------------------------------- gradient check
def _total_loss_fn(net, batch, mask, priors):
    """Scalar loss with pinned priors (matches stop-gradient backward)."""
    out = net.forward(batch, train=False, priors=priors)
    total, _, _, d_pred_num, d_logits = compute_losses(out, batch, mask)
    d_target = None
    if out.target_pred is not None:
        # include the target head in the checked objective
        pred = out.target_pred.reshape(-1)
        d_target = (2.0 * pred / len(pred)).reshape(-1, 1)
        total += float((pred**2).mean())
    net.zero_grad()
    net.backward(out, d_pred_num, d_logits, d_target)
    return total


@pytest.mark.parametrize("net_kind", ["attention", "mlp"])
def test_numerical_gradient_check(batch, net_kind) -> None:
    rng = np.random.default_rng(11)
    if net_kind == "attention":
        net = AttentiveTabularNet(batch, hidden=6, steps=2, emb_dim=3, with_target=True, seed=4)
    else:
        net = ResidualMlpNet(batch, hidden=6, emb_dim=3, dropout=0.0, with_target=True, seed=4)
    mask = make_mask(batch, 0.4, rng)
    priors = [rng.random((batch.n_rows, net.d_in)) + 0.5 for _ in range(net.steps)] \
        if net_kind == "attention" else None
    # pinned priors equal relaxed ones from an unperturbed forward
    if net_kind == "attention":
        out0 = net.forward(batch, train=False)
        priors = [p.copy() for p in out0.cache.get("priors", [])] or priors

    _total_loss_fn(net, batch, mask, priors)
    analytic = {k: g.copy() for k, g in net.G.items()}

    eps = 1e-6
    checked = 0
    for name in sorted(net.P):
        arr = net.P[name]
        flat = arr.reshape(-1)
        ga = analytic[name].reshape(-1)
        # check a deterministic sample of entries per tensor
        n_check = min(6, flat.size)
        idxs = rng.choice(flat.size, size=n_check, replace=False)
        for i in idxs:
            orig = flat[i]
            flat[i] = orig + eps
            lp = _total_loss_fn(net, batch, mask, priors)
            flat[i] = orig - eps
            lm = _total_loss_fn(net, batch, mask, priors)
            flat[i] = orig
            num = (lp - lm) / (2 * eps)
            denom = max(abs(num), abs(ga[i]), 1e-8)
            rel = abs(num - ga[i]) / denom
            assert rel < 1e-4, (
                f"gradient mismatch {name}[{i}]: numeric={num:.3e} "
                f"analytic={ga[i]:.3e} rel={rel:.2e}"
            )
            checked += 1
    assert checked >= 20


# ------------------------------------------------------------- training
def test_denoising_training_reduces_loss(batch) -> None:
    net = AttentiveTabularNet(batch, hidden=8, steps=2, emb_dim=4, seed=5)
    rep = train_denoising(
        net, batch, epochs=40, batch_size=6, corruption_rate=0.3, seed=5
    )
    assert rep.history[0] > rep.final_loss
    assert rep.final_loss > 0.0


def test_make_mask_forces_one_cell_per_row() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"a": [1.0, 2.0, 3.0, 4.0], "c": ["p", "q", "r", "s"]})
    batch = FeatureTokenizer(numeric=["a"], categorical=["c"]).fit_transform(df)
    mask = make_mask(batch, 0.0, rng)  # zero rate -> only forced cells
    assert (mask.num.any(axis=1) | mask.cat.any(axis=1)).all()
    assert mask.num.sum() + mask.cat.sum() == 4  # exactly one per row
