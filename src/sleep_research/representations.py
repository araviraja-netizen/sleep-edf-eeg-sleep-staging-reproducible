"""PCA and bounded exact t-SNE from saved features, independent of the training loop."""

import importlib.util
import numpy as np
from scipy.spatial.distance import pdist, squareform


def pca(features, components=2):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or len(features) < 2 or not np.isfinite(features).all():
        raise ValueError("PCA requires at least two finite feature rows")
    if not 1 <= components <= min(features.shape):
        raise ValueError("Invalid PCA component count")
    mean = features.mean(axis=0)
    centered = features - mean
    u, singular, vt = np.linalg.svd(centered, full_matrices=False)
    # Choose a fixed orientation so SVD sign ambiguity does not flip visualizations.
    signs = np.sign(vt[np.arange(len(vt)), np.argmax(np.abs(vt), axis=1)])
    signs[signs == 0] = 1
    vt *= signs[:, None]
    u *= signs[None, :]
    variance = singular ** 2 / (len(features) - 1)
    total = variance.sum()
    ratios = variance[:components] / total if total else np.zeros(components)
    return u[:, :components] * singular[:components], {"mean": mean.tolist(), "components": vt[:components].tolist(),
           "explained_variance_ratio": ratios.tolist(), "centered": True, "whitened": False}


def joint_probabilities(features, perplexity=30, tolerance=1e-5):
    """Exact Gaussian affinities with entropy-matched bandwidths (t-SNE Algorithm 1)."""
    features = np.asarray(features, dtype=np.float64)
    n = len(features)
    if not 1 < perplexity < n:
        raise ValueError("t-SNE perplexity must exceed one and be smaller than sample count")
    distances = squareform(pdist(features, metric="sqeuclidean"))
    conditional = np.zeros((n, n), dtype=float)
    for i in range(n):
        mask = np.arange(n) != i
        row = distances[i, mask]
        beta, lower, upper = 1.0, -np.inf, np.inf
        for _ in range(60):
            energy = -row * beta
            energy -= energy.max()
            probabilities = np.exp(energy)
            probabilities /= probabilities.sum()
            entropy = -np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300)))
            difference = entropy - np.log(perplexity)
            if abs(difference) < tolerance:
                break
            if difference > 0:
                lower = beta
                beta = beta * 2 if np.isinf(upper) else (beta + upper) / 2
            else:
                upper = beta
                beta = beta / 2 if np.isinf(lower) else (beta + lower) / 2
        conditional[i, mask] = probabilities
    probabilities = (conditional + conditional.T) / (2 * n)
    np.fill_diagonal(probabilities, 0)
    probabilities /= probabilities.sum()
    return probabilities


def tsne_objective_gradient(positions, probabilities):
    distances = squareform(pdist(positions, metric="sqeuclidean"))
    numerator = 1 / (1 + distances)
    np.fill_diagonal(numerator, 0)
    q = numerator / numerator.sum()
    weights = (probabilities - q) * numerator
    gradient = 4 * (weights.sum(axis=1)[:, None] * positions - weights @ positions)
    positive = probabilities > 0
    kl = float(np.sum(probabilities[positive] * np.log(probabilities[positive] / np.maximum(q[positive], 1e-300))))
    return kl, gradient


def exact_tsne(features, perplexity=30, iterations=500, seed=42, max_samples=1000):
    features = np.asarray(features, dtype=np.float64)
    if features.ndim != 2 or not np.isfinite(features).all() or not 4 <= len(features) <= max_samples:
        raise ValueError(f"Exact t-SNE requires 4..{max_samples} finite feature rows; use a recorded subsample")
    if iterations < 300:
        raise ValueError("Use at least 300 iterations to include post-exaggeration optimization")
    if features.shape[1] > 50:
        features = pca(features, min(50, len(features), features.shape[1]))[0]
    scale = features.std()
    if scale <= 0:
        raise ValueError("t-SNE requires nonconstant features")
    features = (features - features.mean(axis=0)) / scale
    probabilities = joint_probabilities(features, perplexity)
    positions = np.random.default_rng(seed).normal(0, 1e-4, size=(len(features), 2))
    velocity = np.zeros_like(positions)
    gains = np.ones_like(positions)
    learning_rate = max(len(features) / 12 / 4, 50.0)
    for step in range(iterations):
        exaggerated = probabilities * (12 if step < 250 else 1)
        _, gradient = tsne_objective_gradient(positions, exaggerated)
        gains = np.where(np.sign(gradient) != np.sign(velocity), gains + 0.2, gains * 0.8)
        gains = np.maximum(gains, 0.01)
        momentum = 0.5 if step < 250 else 0.8
        velocity = momentum * velocity - learning_rate * gains * gradient
        positions += velocity
        positions -= positions.mean(axis=0)
    kl, _ = tsne_objective_gradient(positions, probabilities)
    if not np.isfinite(positions).all():
        raise ValueError("t-SNE optimization diverged")
    return positions, {"method": "exact NumPy/SciPy t-SNE", "perplexity": perplexity, "iterations": iterations,
                       "seed": seed, "early_exaggeration": 12, "early_exaggeration_iterations": 250,
                       "learning_rate": learning_rate, "initialization": "seeded Gaussian std=1e-4",
                       "final_kl_divergence": kl, "sample_count": len(features), "quadratic_memory": True}


def optional_umap(features, seed=42):
    if importlib.util.find_spec("umap") is None:
        raise RuntimeError("UMAP is not installed; optional workflow omitted without installation")
    import umap
    return umap.UMAP(random_state=seed).fit_transform(features)
