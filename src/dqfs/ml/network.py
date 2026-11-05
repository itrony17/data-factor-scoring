"""Deep tabular networks in pure NumPy.

Two branches share one public interface so the denoising trainer can treat
them interchangeably:

* :class:`AttentiveTabularNet` - TabNet-style branch: a shared trunk feeds an
  attentive transformer per decision step; prior-relaxed sparse attention
  selects features multiplicatively (``z = mask * x``) before each step's
  transformation, and step outputs accumulate into the representation.
* :class:`ResidualMlpNet` - residual MLP with categorical embeddings,
  pre-norm blocks and inverted dropout.

Both nets reconstruct masked cells (numeric + categorical) and optionally
carry a supervised target head. Backpropagation is hand-derived and verified
by finite-difference gradient checks in ``tests/test_network.py``.

Design note: the TabNet prior scale ``P`` is treated as a stop-gradient
constant during backward (a documented relaxation used by several TabNet
implementations). ``forward(..., priors=...)`` lets callers pin all step
priors, so gradient checks evaluate exactly the function the backward pass
differentiates.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from dqfs.ml.tokenizer import TokenizedBatch

logger = logging.getLogger(__name__)

_GELU_C = float(np.sqrt(2.0 / np.pi))


def gelu(x: np.ndarray) -> np.ndarray:
    """GELU activation (tanh approximation)."""
    return 0.5 * x * (1.0 + np.tanh(_GELU_C * (x + 0.044715 * x**3)))


def gelu_grad(x: np.ndarray) -> np.ndarray:
    """Derivative of :func:`gelu` with respect to its input ``x``."""
    inner = _GELU_C * (x + 0.044715 * x**3)
    t = np.tanh(inner)
    return 0.5 * (1.0 + t) + 0.5 * x * (1.0 - t * t) * _GELU_C * (1.0 + 3 * 0.044715 * x**2)


def softmax(x: np.ndarray, axis: int = -1) -> np.ndarray:
    """Numerically stable softmax."""
    shifted = x - x.max(axis=axis, keepdims=True)
    e = np.exp(shifted)
    return e / e.sum(axis=axis, keepdims=True)


def _he(fan_in: int, fan_out: int, rng: np.random.Generator) -> np.ndarray:
    """He-normal weight matrix."""
    return rng.normal(0.0, np.sqrt(2.0 / max(fan_in, 1)), size=(fan_in, fan_out))


class Adam:
    """Adam optimizer over a flat ``{name: ndarray}`` parameter dict."""

    def __init__(
        self,
        params: dict[str, np.ndarray],
        lr: float = 3e-3,
        beta1: float = 0.9,
        beta2: float = 0.999,
        eps: float = 1e-8,
    ) -> None:
        self.lr = lr
        self.beta1, self.beta2, self.eps = beta1, beta2, eps
        self.m = {k: np.zeros_like(v) for k, v in params.items()}
        self.v = {k: np.zeros_like(v) for k, v in params.items()}
        self.t = 0

    def step(self, params: dict[str, np.ndarray], grads: dict[str, np.ndarray]) -> None:
        """Apply one Adam update in place."""
        self.t += 1
        b1c = 1.0 - self.beta1**self.t
        b2c = 1.0 - self.beta2**self.t
        for k, g in grads.items():
            if k not in params:
                continue
            self.m[k] = self.beta1 * self.m[k] + (1.0 - self.beta1) * g
            self.v[k] = self.beta2 * self.v[k] + (1.0 - self.beta2) * g * g
            m_hat = self.m[k] / b1c
            v_hat = self.v[k] / b2c
            params[k] -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps)


@dataclass
class ForwardOutput:
    """Everything a branch's forward pass produces (and backward consumes)."""

    repr: np.ndarray                       # (n, hidden) aggregated representation
    pred_num: np.ndarray                   # (n, p) numeric reconstruction
    logits_cat: list[np.ndarray]           # per cat col: (n, V_j+1)
    target_pred: np.ndarray | None         # (n, 1) optional supervised head
    cache: dict[str, Any] = field(default_factory=dict)


class BaseTabularNet:
    """Shared plumbing: flat parameter/grad dicts, linear + layernorm ops."""

    emb_dim: int = 8

    def __init__(self, seed: int = 0) -> None:
        self.rng = np.random.default_rng(seed)
        self.P: dict[str, np.ndarray] = {}
        self.G: dict[str, np.ndarray] = {}

    # ------------------------------------------------------------ params
    def _add_param(self, name: str, arr: np.ndarray) -> None:
        self.P[name] = arr
        self.G[name] = np.zeros_like(arr)

    def _init_linear(self, name: str, fan_in: int, fan_out: int) -> None:
        self._add_param(f"{name}.W", _he(fan_in, fan_out, self.rng))
        self._add_param(f"{name}.b", np.zeros(fan_out))

    def zero_grad(self) -> None:
        """Reset all gradients to zero."""
        for g in self.G.values():
            g.fill(0.0)

    # ---------------------------------------------------------- core ops
    def _lin_fwd(self, name: str, x: np.ndarray) -> np.ndarray:
        return x @ self.P[f"{name}.W"] + self.P[f"{name}.b"]

    def _lin_bwd(self, name: str, x: np.ndarray, dout: np.ndarray) -> np.ndarray:
        self.G[f"{name}.W"] += x.T @ dout
        self.G[f"{name}.b"] += dout.sum(axis=0)
        return dout @ self.P[f"{name}.W"].T

    @staticmethod
    def _ln_fwd(x: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Axis-1 LayerNorm (no affine) -> (out, xhat, inv_std)."""
        mu = x.mean(axis=1, keepdims=True)
        var = x.var(axis=1, keepdims=True)
        inv = 1.0 / np.sqrt(var + 1e-5)
        xhat = (x - mu) * inv
        return xhat, xhat, inv

    @staticmethod
    def _ln_bwd(xhat: np.ndarray, inv: np.ndarray, dy: np.ndarray) -> np.ndarray:
        """Backward through axis-1 LayerNorm (no affine)."""
        n_feat = xhat.shape[1]
        sum_dy = dy.sum(axis=1, keepdims=True)
        sum_dy_xhat = (dy * xhat).sum(axis=1, keepdims=True)
        return (inv / n_feat) * (n_feat * dy - sum_dy - xhat * sum_dy_xhat)

    def _emb_fwd(self, j: int, codes: np.ndarray) -> np.ndarray:
        return self.P[f"emb{j}"][codes]

    def _emb_bwd(self, j: int, codes: np.ndarray, dout: np.ndarray) -> None:
        np.add.at(self.G[f"emb{j}"], codes, dout)

    def _stack_inputs(self, batch: TokenizedBatch) -> np.ndarray:
        """Concatenate categorical embeddings and numeric features."""
        parts = [self._emb_fwd(j, batch.X_cat[:, j]) for j in range(batch.n_cat)]
        parts.append(batch.X_num)
        return np.concatenate(parts, axis=1)

    def _head_forward(
        self, y: np.ndarray, batch: TokenizedBatch, with_target: bool
    ) -> tuple[np.ndarray, list[np.ndarray], np.ndarray | None]:
        pred_num = self._lin_fwd("head_num", y)
        logits_cat = [self._lin_fwd(f"head_cat{j}", y) for j in range(batch.n_cat)]
        target_pred = self._lin_fwd("head_target", y) if with_target else None
        return pred_num, logits_cat, target_pred

    def _head_backward(
        self,
        y: np.ndarray,
        d_pred_num: np.ndarray,
        d_logits_cat: list[np.ndarray],
        d_target: np.ndarray | None,
    ) -> np.ndarray:
        """Accumulate head grads; return gradient wrt representation ``y``."""
        dy = self._lin_bwd("head_num", y, d_pred_num)
        for j, dl in enumerate(d_logits_cat):
            dy = dy + self._lin_bwd(f"head_cat{j}", y, dl)
        if d_target is not None:
            dy = dy + self._lin_bwd("head_target", y, d_target)
        return dy


class AttentiveTabularNet(BaseTabularNet):
    """TabNet-style branch with prior-relaxed sparse feature attention.

    Args:
        batch: Tokenized inputs (used for dimensions).
        hidden: Trunk and step width.
        steps: Number of decision steps.
        emb_dim: Categorical embedding width.
        gamma: Prior relaxation coefficient (>1 encourages feature reuse).
        with_target: Attach a supervised target head.
        seed: Parameter initialization seed.
    """

    def __init__(
        self,
        batch: TokenizedBatch,
        hidden: int = 64,
        steps: int = 3,
        emb_dim: int = 8,
        gamma: float = 1.5,
        with_target: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.hidden = hidden
        self.steps = steps
        self.emb_dim = emb_dim
        self.gamma = gamma
        self.with_target = with_target
        p, q = batch.n_num, batch.n_cat
        self.d_in = p + q * emb_dim
        for j in range(q):
            self._add_param(
                f"emb{j}",
                self.rng.normal(0.0, 0.05, size=(batch.vocab_sizes[j], emb_dim)),
            )
        self._init_linear("trunk", self.d_in, hidden)
        for t in range(steps):
            self._init_linear(f"att{t}", hidden, self.d_in)
            self._init_linear(f"step{t}", self.d_in, hidden)
            self._init_linear(f"out{t}", hidden, hidden)
        self._init_linear("head_num", hidden, p)
        for j, v in enumerate(batch.vocab_sizes):
            self._init_linear(f"head_cat{j}", hidden, v)
        if with_target:
            self._init_linear("head_target", hidden, 1)

    def forward(
        self,
        batch: TokenizedBatch,
        train: bool = False,
        priors: list[np.ndarray] | None = None,
    ) -> ForwardOutput:
        """Run the attentive branch.

        Args:
            batch: Tokenized inputs.
            train: Unused (this branch has no dropout); kept for the shared
                interface.
            priors: Optional pinned prior scales, one ``(n, d_in)`` array per
                step. When given, relaxation is disabled and the supplied
                priors are treated as constants (gradient-check mode).
        """
        del train
        x = self._stack_inputs(batch)
        n = x.shape[0]
        pre_trunk = self._lin_fwd("trunk", x)
        xhat, _, inv = self._ln_fwd(pre_trunk)
        h0 = gelu(xhat)

        pinned = priors is not None
        if pinned:
            prior: list[np.ndarray] = [p.copy() for p in priors]  # type: ignore[union-attr]
        else:
            prior = [np.ones((n, self.d_in)) for _ in range(self.steps)]

        h = h0
        y = np.zeros((n, self.hidden))
        step_caches: list[dict[str, np.ndarray]] = []
        for t in range(self.steps):
            a = self._lin_fwd(f"att{t}", h)
            logits = a * prior[t]           # prior: stop-gradient constant
            m = softmax(logits, axis=1)
            z = m * x
            u_pre = self._lin_fwd(f"step{t}", z)
            u_norm, _, uinv = self._ln_fwd(u_pre)
            d = gelu(u_norm)
            pre_o = self._lin_fwd(f"out{t}", d)
            o = np.maximum(pre_o, 0.0)
            y = y + o
            step_caches.append({
                "h": h, "x": x, "m": m, "z": z, "u_norm": u_norm,
                "uinv": uinv, "d": d, "pre_o": pre_o, "prior": prior[t],
            })
            if not pinned and t + 1 < self.steps:
                prior[t + 1] = prior[t] * (self.gamma - m)
            h = h0 + d

        pred_num, logits_cat, target_pred = self._head_forward(y, batch, self.with_target)
        return ForwardOutput(
            repr=y, pred_num=pred_num, logits_cat=logits_cat,
            target_pred=target_pred,
            cache={"x": x, "h0": h0, "trunk_xhat": xhat, "trunk_inv": inv,
                   "codes": batch.X_cat.copy(), "steps": step_caches},
        )

    def backward(
        self,
        out: ForwardOutput,
        d_pred_num: np.ndarray,
        d_logits_cat: list[np.ndarray],
        d_target: np.ndarray | None = None,
    ) -> None:
        """Backpropagate head gradients through steps, trunk and embeddings."""
        dy = self._head_backward(out.repr, d_pred_num, d_logits_cat, d_target)
        cache = out.cache
        dh0 = np.zeros_like(cache["h0"])
        d_carry = np.zeros_like(dy)   # gradient routed to d_{t-1} via h_t
        dx = np.zeros_like(cache["x"])
        for t in range(self.steps - 1, -1, -1):
            sc = cache["steps"][t]
            # path 1: y += ReLU(pre_o) - ReLU' applies ONLY to this path
            do = dy * (sc["pre_o"] > 0.0)
            dd = self._lin_bwd(f"out{t}", sc["d"], do)
            # path 2: h_{t+1} = h0 + d_t is LINEAR - gradient added after ReLU
            dd = dd + d_carry
            # d = gelu(u_norm), u_norm = LayerNorm(u_pre)
            du = dd * gelu_grad(sc["u_norm"])
            du_pre = self._ln_bwd(sc["u_norm"], sc["uinv"], du)
            dz = self._lin_bwd(f"step{t}", sc["z"], du_pre)
            # z = m * x  (elementwise)
            dx += dz * sc["m"]
            dm = dz * sc["x"]
            # softmax backward; the direct prior scaling factor is exact,
            # while prior's dependence on earlier masks is stop-gradient
            dm_sum = (dm * sc["m"]).sum(axis=1, keepdims=True)
            da = sc["prior"] * sc["m"] * (dm - dm_sum)
            dh_t = self._lin_bwd(f"att{t}", sc["h"], da)
            dh0 += dh_t
            d_carry = dh_t if t > 0 else np.zeros_like(dh_t)
        # trunk: h0 = gelu(LN(trunk(x)))
        dh0_pre = dh0 * gelu_grad(cache["trunk_xhat"])
        dtrunk_pre = self._ln_bwd(cache["trunk_xhat"], cache["trunk_inv"], dh0_pre)
        dx += self._lin_bwd("trunk", cache["x"], dtrunk_pre)
        # route input gradients into the embedding tables
        codes = cache["codes"]
        offset = 0
        for j in range(codes.shape[1]):
            dim = self.P[f"emb{j}"].shape[1]
            self._emb_bwd(j, codes[:, j], dx[:, offset:offset + dim])
            offset += dim


class ResidualMlpNet(BaseTabularNet):
    """Residual MLP branch with embeddings, pre-norm blocks and dropout.

    Args:
        batch: Tokenized inputs (used for dimensions).
        hidden: Width of the input projection and residual blocks.
        blocks: Number of pre-norm residual blocks.
        emb_dim: Categorical embedding width.
        dropout: Dropout rate applied to block activations during training.
        with_target: Attach a supervised target head.
        seed: Parameter initialization seed.
    """

    def __init__(
        self,
        batch: TokenizedBatch,
        hidden: int = 64,
        blocks: int = 2,
        emb_dim: int = 8,
        dropout: float = 0.15,
        with_target: bool = False,
        seed: int = 0,
    ) -> None:
        super().__init__(seed=seed)
        self.hidden = hidden
        self.blocks = blocks
        self.emb_dim = emb_dim
        self.dropout = float(dropout)
        self.with_target = with_target
        q = batch.n_cat
        d_in = batch.n_num + q * emb_dim
        for j in range(q):
            self._add_param(
                f"emb{j}",
                self.rng.normal(0.0, 0.05, size=(batch.vocab_sizes[j], emb_dim)),
            )
        self._init_linear("in", d_in, hidden)
        for b in range(blocks):
            self._init_linear(f"blk{b}.a", hidden, hidden)
            self._init_linear(f"blk{b}.b", hidden, hidden)
        self._init_linear("head_num", hidden, batch.n_num)
        for j, v in enumerate(batch.vocab_sizes):
            self._init_linear(f"head_cat{j}", hidden, v)
        if with_target:
            self._init_linear("head_target", hidden, 1)

    def forward(
        self,
        batch: TokenizedBatch,
        train: bool = False,
        priors: list[np.ndarray] | None = None,
    ) -> ForwardOutput:
        """Run the residual MLP branch (``priors`` unused, shared API)."""
        del priors
        x = self._stack_inputs(batch)
        pre_in = self._lin_fwd("in", x)
        in_norm, _, in_inv = self._ln_fwd(pre_in)
        h = gelu(in_norm)
        block_caches: list[dict[str, np.ndarray]] = []
        for b in range(self.blocks):
            h_in = h
            a_pre = self._lin_fwd(f"blk{b}.a", h_in)
            a_norm, _, a_inv = self._ln_fwd(a_pre)
            act = gelu(a_norm)
            if train and self.dropout > 0:
                mask = (self.rng.random(act.shape) >= self.dropout) / (1.0 - self.dropout)
            else:
                mask = np.ones_like(act)
            dropped = act * mask
            b_pre = self._lin_fwd(f"blk{b}.b", dropped)
            h = h_in + b_pre
            block_caches.append({
                "h_in": h_in, "a_norm": a_norm, "a_inv": a_inv,
                "mask": mask, "dropped": dropped,
            })
        pred_num, logits_cat, target_pred = self._head_forward(h, batch, self.with_target)
        return ForwardOutput(
            repr=h, pred_num=pred_num, logits_cat=logits_cat,
            target_pred=target_pred,
            cache={"x": x, "in_norm": in_norm, "in_inv": in_inv,
                   "codes": batch.X_cat.copy(), "blocks": block_caches},
        )

    def backward(
        self,
        out: ForwardOutput,
        d_pred_num: np.ndarray,
        d_logits_cat: list[np.ndarray],
        d_target: np.ndarray | None = None,
    ) -> None:
        """Backpropagate head gradients through blocks, input LN, embeddings."""
        dh = self._head_backward(out.repr, d_pred_num, d_logits_cat, d_target)
        cache = out.cache
        for b in range(self.blocks - 1, -1, -1):
            bc = cache["blocks"][b]
            db_pre = dh.copy()
            ddropped = self._lin_bwd(f"blk{b}.b", bc["dropped"], db_pre)
            dh_in = dh
            dact = ddropped * bc["mask"]
            dax = dact * gelu_grad(bc["a_norm"])
            da_pre = self._ln_bwd(bc["a_norm"], bc["a_inv"], dax)
            dh_blk = self._lin_bwd(f"blk{b}.a", bc["h_in"], da_pre)
            dh = dh_in + dh_blk
        # input: h_first = gelu(LN(in(x)))
        dh_first = dh * gelu_grad(cache["in_norm"])
        din_pre = self._ln_bwd(cache["in_norm"], cache["in_inv"], dh_first)
        dx = self._lin_bwd("in", cache["x"], din_pre)
        # route gradient into the embedding tables
        offset = 0
        codes = cache["codes"]
        for j in range(codes.shape[1]):
            dim = self.P[f"emb{j}"].shape[1]
            self._emb_bwd(j, codes[:, j], dx[:, offset:offset + dim])
            offset += dim
