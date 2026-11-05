"""ML subpackage: deep tabular ensemble for data quality scoring.

The engine is implemented in pure NumPy (no torch/tensorflow dependency) so
it runs deterministically in CI and small Docker images:

* :class:`~dqfs.ml.tokenizer.FeatureTokenizer` - numeric robust scaling and
  categorical index encoding with embedding vocabularies;
* :class:`~dqfs.ml.network.AttentiveTabularNet` - TabNet-style branch with
  prior-relaxed sparse feature attention;
* :class:`~dqfs.ml.network.ResidualMlpNet` - deep residual MLP branch with
  categorical embeddings and dropout;
* :mod:`~dqfs.ml.denoise` - masked-cell denoising (TabMAE-style) training,
  per-row anomaly signals and per-column permutation importance;
* :class:`~dqfs.ml.ensemble.DeepQualityEnsemble` - stacking orchestrator
  that fuses the two deep branches (plus optional supervised signals) into
  a single row-badness score and a model card.
"""

from dqfs.ml.tokenizer import FeatureTokenizer, TokenizedBatch

__all__ = ["FeatureTokenizer", "TokenizedBatch"]
