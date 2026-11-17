"""Sparse occupancy (``occ_sparse``): labels, shards, sampler, net, tracks, viz adapter.

Spec: ``docs/specs/occ_sparse.md``. Runtime modules (``grid``, ``classes``,
``model``, ``tracks``, ``polylines``, ``viz_adapter``, ``runtime``) never import
``labels`` and never read depth, semantics or boxes. Those are training-only.
"""
