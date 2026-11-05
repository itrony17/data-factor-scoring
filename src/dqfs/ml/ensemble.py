"""Stacking orchestrator: fuse deep branches into row badness + importance.

The :class:`DeepQualityEnsemble` runs the full deep pipeline:

1. tokenize the dataset (robust numeric scaling, categorical vocabularies);
2. pretrain two branches with masked-cell denoising:
   an :class:`~dqfs.ml.network.AttentiveTabularNet` (TabNet-style) and a
   :class:`~dqfs.ml.network.ResidualMlpNet`;
3. extract one row-badness signal per branch (mean z-normalized masked
   reconstruction error, rank-normalized);
4. optionally (``target`` column provided) train supervised target heads
   out-of-fold and a ridge base learner; their absolute residuals become
   two more signals;
5. blend the signals (equal-weight ensemble) and, when a rule-based factor
   matrix is available, calibrate through a ridge meta-learner that maps
   deep signals to the statistical-rule badness (the "stacking blender");
6. compute per-column permutation importance for ML-driven factor weights.
"""

from __future__ import annotations

import logging
import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from dqfs.config import DqfsConfig
from dqfs.io_loader import ColumnSpecs
from dqfs.ml.denoise import (
    permutation_importance,
    rank_normalize,
    row_error_signals,
    train_denoising,
    train_target_head,
    zscore,
)
from dqfs.ml.network import AttentiveTabularNet, ResidualMlpNet
from dqfs.ml.tokenizer import FeatureTokenizer

logger = logging.getLogger(__name__)

_MIN_ROWS = 16
_MAX_TRAIN_ROWS = 4000
_RIDGE_LAMBDA = 1e-3


@dataclass
class EnsembleOutput:
    """Result of a full ensemble fit."""

    row_badness: np.ndarray                 # (n,) in [0, 1]
    column_importance: dict[str, float] = field(default_factory=dict)
    model_card: dict[str, Any] = field(default_factory=dict)
    enabled: bool = True


def _ridge_fit_predict(
    features: np.ndarray, target: np.ndarray, lam: float = _RIDGE_LAMBDA
) -> tuple[np.ndarray, np.ndarray]:
    """Closed-form ridge; returns (predictions, coefficients incl. intercept)."""
    design = np.column_stack([np.ones(len(features)), features])
    gram = design.T @ design
    reg = lam * np.eye(gram.shape[0])
    reg[0, 0] = 0.0  # do not regularize the intercept
    coef = np.linalg.solve(gram + reg, design.T @ target)
    return design @ coef, coef


def _r_squared(pred: np.ndarray, actual: np.ndarray) -> float:
    ss_res = float(((actual - pred) ** 2).sum())
    ss_tot = float(((actual - actual.mean()) ** 2).sum())
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


class DeepQualityEnsemble:
    """Train the deep ensemble and produce row badness + column importance."""

    def __init__(self, cfg: DqfsConfig) -> None:
        self.ml = cfg.ml
        self.cfg = cfg

    # ------------------------------------------------------------------ fit
    def fit(
        self,
        df: pd.DataFrame,
        specs: ColumnSpecs,
        target: str | None = None,
        rule_badness: np.ndarray | None = None,
    ) -> EnsembleOutput:
        """Run the full deep pipeline.

        Args:
            df: Full dataset (inference runs on every row).
            specs: Column classification.
            target: Optional supervised target column name.
            rule_badness: Optional per-row rule-based badness in [0, 1]
                (mean of available factor row contributions). Used as the
                meta-learner calibration target.

        Returns:
            :class:`EnsembleOutput` aligned with ``df`` rows.
        """
        started = time.perf_counter()
        ml_cols = [c for c in specs.numeric + specs.categorical if c in df.columns]
        card: dict[str, Any] = {"enabled": True, "seed": self.ml.seed}
        if target is not None and target not in df.columns:
            logger.warning("target column %r not found - running unsupervised", target)
            target = None

        if not self.ml.enabled or len(df) < _MIN_ROWS or not ml_cols:
            reason = (
                "disabled by config" if not self.ml.enabled
                else f"too few rows ({len(df)} < {_MIN_ROWS})" if len(df) < _MIN_ROWS
                else "no numeric/categorical features"
            )
            card.update({"enabled": False, "reason": reason})
            return EnsembleOutput(
                row_badness=np.zeros(len(df)), model_card=card, enabled=False
            )

        rng = np.random.default_rng(self.ml.seed)
        # training sample keeps runtime bounded; inference covers all rows
        if len(df) > _MAX_TRAIN_ROWS:
            train_idx = np.sort(rng.choice(len(df), _MAX_TRAIN_ROWS, replace=False))
        else:
            train_idx = np.arange(len(df))
        df_train = df.iloc[train_idx]

        tokenizer = FeatureTokenizer(
            numeric=specs.numeric,
            categorical=specs.categorical,
            vocab_size=512,
        )
        batch_train = tokenizer.fit_transform(df_train)
        batch_full = tokenizer.transform(df)
        card["training_rows"] = int(len(df_train))
        card["scored_rows"] = int(len(df))

        # --- branch A: TabNet-style attention -----------------------------
        net_a = AttentiveTabularNet(
            batch_train,
            hidden=self.ml.hidden_dim,
            steps=self.ml.attention_steps,
            emb_dim=self.ml.embedding_dim,
            with_target=target is not None,
            seed=self.ml.seed,
        )
        rep_a = train_denoising(
            net_a, batch_train, self.ml.epochs, self.ml.batch_size,
            self.ml.corruption_rate, seed=self.ml.seed,
        )
        # --- branch B: residual MLP ---------------------------------------
        net_b = ResidualMlpNet(
            batch_train,
            hidden=self.ml.hidden_dim,
            emb_dim=self.ml.embedding_dim,
            dropout=self.ml.dropout,
            with_target=target is not None,
            seed=self.ml.seed + 1,
        )
        rep_b = train_denoising(
            net_b, batch_train, self.ml.epochs, self.ml.batch_size,
            self.ml.corruption_rate, seed=self.ml.seed + 1,
        )

        # --- unsupervised signals ------------------------------------------
        signals: dict[str, np.ndarray] = {
            "attention_branch": row_error_signals(
                net_a, batch_full, self.ml.corruption_rate, seed=self.ml.seed
            ),
            "mlp_branch": row_error_signals(
                net_b, batch_full, self.ml.corruption_rate, seed=self.ml.seed + 1
            ),
        }

        # --- supervised signals (optional) ----------------------------------
        supervised = False
        if target is not None:
            y_values = pd.to_numeric(df[target], errors="coerce").to_numpy(float)
            keep = ~np.isnan(y_values)
            y_train = y_values[keep]
            binary = len(np.unique(y_train[np.isfinite(y_train)])) <= 2
            oof = self._target_oof_predictions(
                net_a, net_b, batch_full, batch_train, y_values, keep, binary, rng
            )
            signals.update(oof)
            supervised = bool(oof)
            card["target"] = target
            card["target_mode"] = "binary" if binary else "regression"

        # --- blend signals -> deep badness ----------------------------------
        sig_names = list(signals)
        sig_matrix = np.column_stack([rank_normalize(signals[k]) for k in sig_names]) \
            if sig_names else np.zeros((len(df), 0))
        deep_core = sig_matrix.mean(axis=1) if sig_names else np.zeros(len(df))

        # --- stacked meta-learner (deep signals -> rule badness) ------------
        row_badness = deep_core
        if rule_badness is not None and sig_names:
            z_feats = np.column_stack([zscore(signals[k]) for k in sig_names])
            fitted, coefs = _ridge_fit_predict(z_feats, rank_normalize(rule_badness))
            row_badness = np.clip(
                0.5 * rank_normalize(fitted) + 0.5 * deep_core, 0.0, 1.0
            )
            card["meta_ridge"] = {
                "signals": sig_names,
                "coefficients": [round(float(c), 4) for c in coefs],
                "r_squared": round(_r_squared(fitted, rank_normalize(rule_badness)), 4),
            }
        row_badness = np.clip(rank_normalize(row_badness), 0.0, 1.0)

        # --- per-column deep importance --------------------------------------
        imp_a = permutation_importance(
            net_a, tokenizer, df_train, batch_train,
            self.ml.corruption_rate, seed=self.ml.seed,
        )
        imp_b = permutation_importance(
            net_b, tokenizer, df_train, batch_train,
            self.ml.corruption_rate, seed=self.ml.seed + 1,
        )
        all_cols = sorted(set(imp_a) | set(imp_b))
        column_importance = {
            c: float(0.5 * imp_a.get(c, 0.0) + 0.5 * imp_b.get(c, 0.0))
            for c in all_cols
        }
        top_imp = max(column_importance.values(), default=0.0)
        if top_imp > 0:
            column_importance = {c: v / top_imp for c, v in column_importance.items()}

        card.update({
            "architecture": {
                "attention_branch": {
                    "type": "TabNet-style attentive transformer",
                    "steps": self.ml.attention_steps,
                    "hidden": self.ml.hidden_dim,
                    "parameters": int(sum(v.size for v in net_a.P.values())),
                },
                "mlp_branch": {
                    "type": "residual MLP with embeddings",
                    "blocks": 2,
                    "hidden": self.ml.hidden_dim,
                    "parameters": int(sum(v.size for v in net_b.P.values())),
                },
            },
            "denoising": {
                "epochs": self.ml.epochs,
                "corruption_rate": self.ml.corruption_rate,
                "final_loss_attention": round(rep_a.final_loss, 4),
                "final_loss_mlp": round(rep_b.final_loss, 4),
            },
            "signals": sig_names,
            "supervised": supervised,
            "runtime_seconds": round(time.perf_counter() - started, 2),
        })
        return EnsembleOutput(
            row_badness=row_badness,
            column_importance=column_importance,
            model_card=card,
            enabled=True,
        )

    # ------------------------------------------------------- supervised OOF
    def _target_oof_predictions(
        self,
        net_a: AttentiveTabularNet,
        net_b: ResidualMlpNet,
        batch_full,
        batch_train,
        y_values: np.ndarray,
        keep: np.ndarray,
        binary: bool,
        rng: np.random.Generator,
    ) -> dict[str, np.ndarray]:
        """Out-of-fold absolute residual signals from target heads + ridge.

        The two deep branches each get an out-of-fold target head (trunk
        frozen, head trained per fold). A ridge base learner over tokenized
        features provides a fourth signal. Folds are only defined over rows
        with a valid target value; invalid-target rows get a zero signal.
        """
        valid_rows = np.where(keep)[0]
        folds = min(self.ml.oof_folds, max(2, len(valid_rows) // 8))
        if len(valid_rows) < folds * 2:
            return {}
        fold_of = {r: i % folds for i, r in enumerate(valid_rows)}
        head_epochs = max(20, self.ml.epochs // 2)
        oof_pred_deep = {"attention_target": np.full(len(y_values), np.nan),
                         "mlp_target": np.full(len(y_values), np.nan)}
        for name, net in (("attention_target", net_a), ("mlp_target", net_b)):
            oof = oof_pred_deep[name]
            for k in range(folds):
                tr = np.array([r for r in valid_rows if fold_of[r] != k])
                va = np.array([r for r in valid_rows if fold_of[r] == k])
                if len(tr) < 4 or len(va) == 0:
                    continue
                net_copy = train_target_head_on_copy(net, batch_full, y_values, tr,
                                                     head_epochs, binary, rng)
                out = net_copy.forward(batch_full, train=False)
                assert out.target_pred is not None
                oof[va] = out.target_pred.reshape(-1)[va]
            logger.debug("OOF head %s done over %d folds", name, folds)

        # ridge base learner over tokenized features (numeric + cat codes)
        feats = np.column_stack([
            batch_full.X_num,
            batch_full.X_cat.astype(float),
        ])
        feats = (feats - feats.mean(axis=0)) / (feats.std(axis=0) + 1e-9)
        oof_ridge = np.full(len(y_values), np.nan)
        for k in range(folds):
            tr = np.array([r for r in valid_rows if fold_of[r] != k])
            va = np.array([r for r in valid_rows if fold_of[r] == k])
            if len(tr) < 4 or len(va) == 0:
                continue
            pred, _ = _ridge_fit_predict(feats[tr], y_values[tr])
            # predict on va: refit design with va rows
            design_va = np.column_stack([np.ones(len(va)), feats[va]])
            design_tr = np.column_stack([np.ones(len(tr)), feats[tr]])
            gram = design_tr.T @ design_tr
            reg = _RIDGE_LAMBDA * np.eye(gram.shape[0])
            reg[0, 0] = 0.0
            coef = np.linalg.solve(gram + reg, design_tr.T @ y_values[tr])
            oof_ridge[va] = design_va @ coef

        signals: dict[str, np.ndarray] = {}
        for name, oof in (*oof_pred_deep.items(), ("ridge_base", oof_ridge)):
            resid = np.abs(y_values - oof)
            resid[~np.isfinite(resid)] = 0.0
            sig = zscore(np.nan_to_num(resid))
            sig[~keep] = 0.0
            signals[name] = np.clip(sig, -3.0, 3.0)
        return {k: v for k, v in signals.items()}


def train_target_head_on_copy(
    net, batch, y_values, train_idx, epochs, binary, rng
):
    """Train the target head on a deep copy so OOF folds stay independent."""
    from dqfs.ml.denoise import copy_net

    clone = copy_net(net)
    train_target_head(
        clone, batch, y_values, train_idx, epochs,
        lr=3e-3, binary=binary, seed=int(rng.integers(1 << 31)),
    )
    return clone
