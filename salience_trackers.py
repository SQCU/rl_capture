# salience_trackers.py 

import numpy as np
from sklearn.decomposition import IncrementalPCA
from scipy.spatial.distance import mahalanobis
from collections import deque

# --- NEW: A factory dictionary to map config strings to classes ---
TRACKER_STRATEGIES = {
    "naive_cos_dissimilarity": NaiveZScoreTracker,
    "pca_mahalanobis": PCAMahanalobisTracker,
}

# We still need OnlineStats for our trackers
class OnlineStats:
    """Implements Welford's algorithm for stable online variance calculation."""
    def __init__(self):
        self.n = 0
        self.mean = 0.0
        self.M2 = 0.0

    def update(self, new_value: float):
        self.n += 1
        delta = new_value - self.mean
        self.mean += delta / self.n
        delta2 = new_value - self.mean
        self.M2 += delta * delta2

    @property
    def variance(self) -> float:
        return self.M2 / self.n if self.n > 1 else 0.0

    @property
    def std_dev(self) -> float:
        return self.variance**0.5

class BaseSalienceTracker:
    """
    Abstract Base Class for all salience tracking strategies.
    Defines the interface that the SalienceWorker will use.
    """
    def __init__(self, config: dict):
        self.config = config

    def update(self, latents_batch: np.ndarray):
        """Update the internal state of the tracker with a new batch of latents."""
        raise NotImplementedError

    def get_novelty_scores(self, latents_batch: np.ndarray) -> np.ndarray:
        """Calculate a novelty score for each latent in a batch."""
        raise NotImplementedError

    def is_novel(self, scores: np.ndarray) -> np.ndarray:
        """Given a batch of scores, return a boolean mask of which are novel."""
        raise NotImplementedError


class NaiveZScoreTracker(BaseSalienceTracker):
    """
    The original strategy: uses cosine distance between sequential frames
    and flags an event based on a simple Z-score.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.distance_stats = OnlineStats()
        self.last_latent = None

    def update(self, latents_batch: np.ndarray):
        # We update the stats based on the distances within this batch
        if self.last_latent is None and len(latents_batch) > 0:
            self.last_latent = latents_batch[0]
        
        if len(latents_batch) > 1:
            # Prepend the last latent from the previous batch to calculate the first distance
            full_sequence = np.vstack([self.last_latent, latents_batch])
            # Cosine similarity is 1 - distance
            sim = np.einsum('ij,ij->i', full_sequence[:-1], full_sequence[1:]) / (np.linalg.norm(full_sequence[:-1], axis=1) * np.linalg.norm(full_sequence[1:], axis=1))
            distances = 1 - sim
            for dist in distances:
                self.distance_stats.update(dist)
        
        if len(latents_batch) > 0:
            self.last_latent = latents_batch[-1]

    def get_novelty_scores(self, latents_batch: np.ndarray) -> np.ndarray:
        # The "score" is simply the cosine distance to the previous frame.
        if self.last_latent is None or len(latents_batch) == 0:
            return np.zeros(len(latents_batch))

        full_sequence = np.vstack([self.last_latent, latents_batch])
        sim = np.einsum('ij,ij->i', full_sequence[:-1], full_sequence[1:]) / (np.linalg.norm(full_sequence[:-1], axis=1) * np.linalg.norm(full_sequence[1:], axis=1))
        return 1 - sim
    
    def is_novel(self, scores: np.ndarray) -> np.ndarray:
        mean, std_dev = self.distance_stats.mean, self.distance_stats.std_dev
        if std_dev == 0:
            return np.zeros_like(scores, dtype=bool)
        
        threshold = mean + (std_dev * self.config['z_score_threshold'])
        return scores > threshold

class PCAMahanalobisTracker(BaseSalienceTracker):
    """
    The advanced strategy: uses IncrementalPCA to model the latent space
    and a dynamic Z-score on the Mahalanobis distance to detect novelty.
    This is a CPU-based implementation.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.latent_dim = 1024 # Assuming SigLIP base
        self.pca = IncrementalPCA(n_components=self.config['pca_n_components'])
        self.novelty_score_stats = OnlineStats()
        self.n_samples_seen = 0

    def update(self, latents_batch: np.ndarray):
        self.pca.partial_fit(latents_batch)
        self.n_samples_seen += len(latents_batch)

    def get_novelty_scores(self, latents_batch: np.ndarray) -> np.ndarray:
        if self.n_samples_seen <= self.config['pca_n_components']:
            return np.zeros(len(latents_batch))

        # Determine dynamic number of components to use based on variance threshold
        cumulative_variance = np.cumsum(self.pca.explained_variance_ratio_)
        k_dynamic = np.searchsorted(cumulative_variance, self.config['pca_variance_threshold']) + 1
        
        # Get the required components for the calculation
        mean = self.pca.mean_
        components = self.pca.components_[:k_dynamic]
        variance = self.pca.explained_variance_[:k_dynamic]

        # Project, calculate squared Mahalanobis distance, and take sqrt
        transformed = (latents_batch - mean) @ components.T
        sq_mahalanobis = np.sum((transformed ** 2) / variance, axis=1)
        
        return np.sqrt(sq_mahalanobis)

    def is_novel(self, scores: np.ndarray) -> np.ndarray:
        # Update our running stats on what a "normal" score is
        for score in scores:
            self.novelty_score_stats.update(score)
            
        mean, std_dev = self.novelty_score_stats.mean, self.novelty_score_stats.std_dev
        if std_dev == 0:
            return np.zeros_like(scores, dtype=bool)

        # A score is novel if it's an outlier compared to *other recent scores*
        threshold = mean + (std_dev * self.config['novelty_z_score_threshold'])
        return scores > threshold