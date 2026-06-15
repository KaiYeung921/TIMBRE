"""
Pure math helpers for TIMBRE analysis.
No keras, no data loading required — import freely.
"""
import numpy as np


def sim(a, b):
    """Rotation-invariant similarity in [0,1] between two complex filters."""
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    return 0.0 if na == 0 or nb == 0 else np.abs(np.sum(np.conj(a) * b)) / (na * nb)


def align(a, b):
    """Rotate a about the origin onto b (removes global-phase freedom)."""
    return a * np.exp(1j * np.angle(np.sum(np.conj(a) * b)))


def orthobasis(vectors):
    """Orthonormal basis for the span of a list of complex filters via QR."""
    Q, _ = np.linalg.qr(np.column_stack(vectors))
    return Q


def principal_cos(QA, QB):
    """Cosines of principal angles between two subspaces (1=aligned, 0=orthogonal)."""
    return np.clip(np.linalg.svd(np.conj(QA.T) @ QB, compute_uv=False), 0, 1)


def participation_ratio(sv):
    """Effective number of dimensions from a singular-value spectrum."""
    lam = sv ** 2
    return lam.sum() ** 2 / (np.sum(lam ** 2) + 1e-30)


def extra_subspace(arm_node_vecs, m3_vec):
    """Orthonormal basis for the arm-subspace component orthogonal to M3."""
    Q = orthobasis(arm_node_vecs)
    m = m3_vec / np.linalg.norm(m3_vec)
    Qp = Q - np.outer(m, np.conj(m) @ Q)
    U, s, _ = np.linalg.svd(Qp, full_matrices=False)
    return U[:, s > 1e-6 * s.max()]


def partition_ok(arm_nodes, n_nodes):
    """Return True if each arm has a clean disjoint node assignment."""
    flat = [n for arm in arm_nodes for n in arm]
    return len(set(flat)) == (n_nodes // 3) * 3


def eta_sq(act, bins):
    """Proportion of variance in activations explained by group membership (eta-squared)."""
    valid = ~bins.isna()
    a, b = act[valid].values, bins[valid].values
    gm = a.mean()
    ss_total = ((a - gm) ** 2).sum()
    ss_between = sum(np.sum(b == k) * (a[b == k].mean() - gm) ** 2 for k in np.unique(b))
    return ss_between / (ss_total + 1e-10)
