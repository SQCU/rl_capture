# salience_trackers.py 

import os
import numpy as np
import torch
import torch.nn.functional as F
from sklearn.decomposition import IncrementalPCA
from scipy.spatial.distance import mahalanobis
from collections import deque

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
        self.device = config['device']

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
    A Torch-native implementation of the Z-score strategy.
    All core calculations are done on the GPU.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        self.distance_stats = OnlineStats()
        self.last_latent = None

    def _stable_cosine_distance(self, sequence: torch.Tensor) -> torch.Tensor:
        """ Calculates cosine distance robustly on the specified device. """
        # F.normalize is the numerically stable way to do this in PyTorch
        normalized_sequence = F.normalize(sequence, p=2, dim=1)
        
        # Calculate similarity between adjacent vectors
        similarity = torch.einsum('ij,ij->i', normalized_sequence[:-1], normalized_sequence[1:])
        
        # Clamp values to handle potential floating point inaccuracies
        similarity = torch.clamp(similarity, -1.0, 1.0)
        
        return 1.0 - similarity

    def update(self, latents_batch: torch.Tensor):
        if self.last_latent is None and len(latents_batch) > 0:
            self.last_latent = latents_batch[0].unsqueeze(0)
        
        if len(latents_batch) > 1:
            full_sequence = torch.cat([self.last_latent, latents_batch], dim=0)
            distances = self._stable_cosine_distance(full_sequence)
            
            # --- MINIMAL CPU TRANSFER ---
            # Only the final scalar distances are moved to the CPU to update the stats.
            cpu_distances = distances.cpu().float().numpy()
            for dist in cpu_distances:
                self.distance_stats.update(dist)
        
        if len(latents_batch) > 0:
            self.last_latent = latents_batch[-1].unsqueeze(0)

    def get_novelty_scores(self, latents_batch: torch.Tensor) -> torch.Tensor:
        if self.last_latent is None or len(latents_batch) == 0:
            return torch.zeros(len(latents_batch), device=self.device)

        full_sequence = torch.cat([self.last_latent, latents_batch], dim=0)
        return self._stable_cosine_distance(full_sequence)
    
    def is_novel(self, scores: torch.Tensor) -> torch.Tensor:
        mean, std_dev = self.distance_stats.mean, self.distance_stats.std_dev
        if std_dev == 0:
            return torch.zeros_like(scores, dtype=torch.bool)
        
        threshold = mean + (std_dev * self.config['z_score_threshold'])
        return scores > threshold


class PCAMahanalobisTracker(BaseSalienceTracker):
    """
    A hybrid Torch/Sklearn implementation of the PCA Mahalanobis strategy.
    - Model fitting (update) is done on the CPU using sklearn for robustness.
    - Novelty scoring is done on the GPU using pure PyTorch for speed.
    """
    def __init__(self, config: dict):
        super().__init__(config)
        # The PCA model itself lives on the CPU.
        self.pca = IncrementalPCA(n_components=self.config['pca_n_components'])
        
        # We will store Torch copies of the PCA parameters on the target device.
        self.mean_ = None
        self.components_ = None
        self.explained_variance_ = None
        
        # The final novelty score statistics are still scalar and live on the CPU.
        self.novelty_score_stats = OnlineStats()
        self.n_samples_seen = 0

        # --- NEW: An internal buffer to accumulate latents before fitting ---
        self.latent_buffer = []
        # We'll fit the model when the buffer reaches a reasonable size, e.g., twice the number of components.
        self.fit_threshold = self.config['pca_n_components'] * 2

    def update(self, latents_batch: torch.Tensor):
        """
        Accumulates new latents and periodically updates the PCA model
        once enough new samples have been collected.
        """
        # --- MODIFIED: The update logic is now buffered ---

        # 1. Add new latents to our internal buffer.
        #    This is an infrequent operation, so the CPU transfer is acceptable.
        self.latent_buffer.extend(latents_batch.cpu().float().numpy())
        
        if len(self.latent_buffer) >= self.fit_threshold:
            fit_data = np.array(self.latent_buffer)
            
            # Check for the first-fit condition explicitly
            if self.n_samples_seen == 0 and len(fit_data) < self.pca.n_components:
                # This case should no longer happen with the buffer, but it's a safe guard.
                return 

            print(f"[{os.getpid()}/PCATracker] Fitting PCA with {len(fit_data)} new samples...")
            self.pca.partial_fit(fit_data)
            self.n_samples_seen += len(fit_data)

            # Clear the buffer now that the data is in the model.
            self.latent_buffer = []

            # After fitting, transfer the new model parameters to the GPU.
            self.mean_ = torch.from_numpy(self.pca.mean_).to(self.device, dtype=torch.float32)
            self.components_ = torch.from_numpy(self.pca.components_).to(self.device, dtype=torch.float32)
            self.explained_variance_ = torch.from_numpy(self.pca.explained_variance_).to(self.device, dtype=torch.float32)
        else:
            print(f"len(self.latent_buffer):{len(self.latent_buffer)}<=self.fit_threshold:{self.fit_threshold}, pooling...")

    def get_novelty_scores(self, latents_batch: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Mahalanobis distance for a batch of latents
        entirely on the GPU using the cached PCA parameters.
        """
        # If the model hasn't been fitted yet, there's no novelty.
        if self.mean_ is None or self.n_samples_seen <= self.config['pca_n_components']:
            return torch.zeros(len(latents_batch), device=self.device)

        # Use p% variance threshold to find the dynamic number of components to use.
        cumulative_variance = torch.cumsum(self.explained_variance_ / torch.sum(self.explained_variance_), dim=0)
        k_dynamic = torch.searchsorted(cumulative_variance, self.config['pca_variance_threshold']).item() + 1
        
        # Slice the components and variance to the dynamic size for denoising.
        components = self.components_[:k_dynamic]
        variance = self.explained_variance_[:k_dynamic]

        # --- All of the following operations are pure PyTorch on the GPU ---
        
        # Project the centered data onto the principal components.
        transformed = (latents_batch - self.mean_) @ components.T
        
        # Calculate the squared Mahalanobis distance. Add epsilon for numerical stability.
        epsilon = 1e-8
        sq_mahalanobis = torch.sum((transformed ** 2) / (variance + epsilon), dim=1)
        
        return torch.sqrt(sq_mahalanobis)

    def is_novel(self, scores: torch.Tensor) -> torch.Tensor:
        # Transfer the final scalar scores to the CPU to update the OnlineStats.
        # This is a minimal data transfer.
        cpu_scores = scores.detach().cpu().float().numpy()
        for score in cpu_scores:
            if not np.isnan(score):
                self.novelty_score_stats.update(score)
            
            mean, std_dev = self.novelty_score_stats.mean, self.novelty_score_stats.std_dev
        if std_dev == 0:
            return torch.zeros_like(scores, dtype=torch.bool)

        threshold = mean + (std_dev * self.config['novelty_z_score_threshold'])
        is_novel_mask = scores > threshold

        # --- NEW: Detailed Logging for Tuning ---
        # Use .any() to check if there are any spikes without a CPU transfer
        if is_novel_mask.any():
            # Only transfer data to CPU for printing if a spike was detected
            cpu_scores = scores.detach().cpu().numpy()
            cpu_mask = is_novel_mask.cpu().numpy()
            for i in np.where(cpu_mask)[0]:
                print(
                    f"[PCATracker] KEYFRAME DETECTED! "
                    f"Score: {cpu_scores[i]:.4f} > Threshold: {threshold:.4f} "
                    f"(Mean: {mean:.4f}, StdDev: {std_dev:.4f}, Z-Score: {(cpu_scores[i]-mean)/std_dev:.2f})"
                )
        
        return is_novel_mask

# --- NEW: A factory dictionary to map config strings to classes ---
TRACKER_STRATEGIES = {
    "naive_cos_dissimilarity": NaiveZScoreTracker,
    "pca_mahalanobis": PCAMahanalobisTracker,
}
