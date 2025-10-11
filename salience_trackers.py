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

class TorchOnlineStats:
    """A Torch-native implementation of Welford's algorithm for stable online variance."""
    def __init__(self, device):
        self.device = device
        self.n = torch.tensor(0, dtype=torch.int64, device=self.device)
        self.mean = torch.tensor(0.0, dtype=torch.float32, device=self.device)
        self.M2 = torch.tensor(0.0, dtype=torch.float32, device=self.device)

    def update(self, new_values_batch: torch.Tensor):
        """Updates stats with a whole batch of new values on the GPU."""
        if new_values_batch.numel() == 0:
            return
            
        batch_n = new_values_batch.numel()
        batch_mean = new_values_batch.mean()
        batch_M2 = torch.sum((new_values_batch - batch_mean) ** 2)

        if self.n == 0:
            self.n = batch_n
            self.mean = batch_mean
            self.M2 = batch_M2
            return

        new_n = self.n + batch_n
        delta = batch_mean - self.mean
        
        self.M2 = self.M2 + batch_M2 + (delta ** 2) * self.n * batch_n / new_n
        self.mean = self.mean + delta * batch_n / new_n
        self.n = new_n

    @property
    def variance(self) -> torch.Tensor:
        return self.M2 / self.n if self.n > 1 else torch.tensor(0.0, device=self.device)

    @property
    def std_dev(self) -> torch.Tensor:
        return torch.sqrt(self.variance)

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
        # --- MODIFIED: The PCA model now lives on the GPU. ---
        # self.pca = IncrementalPCA(n_components=self.config['pca_n_components']) # --- REMOVED ---
        self.pca = TorchIncrementalPCA(n_components=self.config['pca_n_components'], device=self.device) # +++ ADDED +++
        
        # --- REMOVED: We no longer need to manually cache parameters on the GPU. ---
        # self.mean_ = None
        # self.components_ = None
        # self.explained_variance_ = None
        
        # --- MODIFIED: Use the new GPU-native stats tracker. ---
        # self.novelty_score_stats = OnlineStats() # --- REMOVED ---
        self.novelty_score_stats = TorchOnlineStats(device=self.device) # +++ ADDED +++
        self.n_samples_seen = 0

        # --- MODIFIED: The buffer can now hold GPU tensors directly. ---
        self.latent_buffer = []
        self.fit_threshold = self.config['pca_n_components'] * 2

    def update(self, latents_batch: torch.Tensor):
        """
        Accumulates new latents and periodically updates the PCA model
        once enough new samples have been collected.
        """
        # --- MODIFIED: Accumulate GPU tensors directly, no CPU transfer. ---
        # self.latent_buffer.extend(latents_batch.cpu().float().numpy()) # --- REMOVED ---
        self.latent_buffer.append(latents_batch) # +++ ADDED +++
        
        # --- MODIFIED: Fit with a concatenated GPU tensor. ---
        if sum(t.shape[0] for t in self.latent_buffer) >= self.fit_threshold:
            # fit_data = np.array(self.latent_buffer) # --- REMOVED ---
            fit_data = torch.cat(self.latent_buffer, dim=0) # +++ ADDED +++
            
            # This guard is still good practice.
            if self.n_samples_seen == 0 and len(fit_data) < self.pca.n_components:
                return 

            print(f"[{os.getpid()}/PCATracker] Fitting PCA with {len(fit_data)} new samples... (ON GPU)")
            # --- THIS IS THE KEY: This call is now non-blocking and runs on the GPU. ---
            self.pca.partial_fit(fit_data)
            self.n_samples_seen += len(fit_data)
            self.latent_buffer = []

            # --- REMOVED: No longer need to manually transfer parameters back to the GPU. ---
            # self.mean_ = torch.from_numpy(self.pca.mean_).to(self.device, dtype=torch.float32)
            # self.components_ = torch.from_numpy(self.pca.components_).to(self.device, dtype=torch.float32)
            # self.explained_variance_ = torch.from_numpy(self.pca.explained_variance_).to(self.device, dtype=torch.float32)
        else:
            # Minor change for accurate logging
            print(f"len(self.latent_buffer):{sum(t.shape[0] for t in self.latent_buffer)}<=self.fit_threshold:{self.fit_threshold}, pooling...")


    def get_novelty_scores(self, latents_batch: torch.Tensor) -> torch.Tensor:
        """
        Calculates the Mahalanobis distance for a batch of latents
        entirely on the GPU using the cached PCA parameters.
        """
        # --- MODIFIED: Access parameters directly from the GPU model. ---
        # if self.mean_ is None or self.n_samples_seen <= self.config['pca_n_components']: # --- REMOVED ---
        if self.pca.mean_ is None or self.n_samples_seen <= self.config['pca_n_components']: # +++ ADDED +++
            return torch.zeros(len(latents_batch), device=self.device)

        # Access all parameters directly from the torch-native pca object
        explained_variance_ = self.pca.explained_variance_
        cumulative_variance = torch.cumsum(explained_variance_ / torch.sum(explained_variance_), dim=0)
        k_dynamic = torch.searchsorted(cumulative_variance, self.config['pca_variance_threshold']).item() + 1
        
        components = self.pca.components_[:k_dynamic]
        variance = explained_variance_[:k_dynamic]
        
        transformed = (latents_batch - self.pca.mean_) @ components.T
        
        epsilon = 1e-8
        sq_mahalanobis = torch.sum((transformed ** 2) / (variance + epsilon), dim=1)
        
        return torch.sqrt(sq_mahalanobis)

    def is_novel(self, scores: torch.Tensor) -> torch.Tensor:
        # --- MODIFIED: Update stats in a single, non-blocking GPU call. ---
        # cpu_scores = scores.detach().cpu().float().numpy() # --- REMOVED ---
        # for score in cpu_scores: # --- REMOVED ---
        #     if not np.isnan(score): # --- REMOVED ---
        #         self.novelty_score_stats.update(score) # --- REMOVED ---
        self.novelty_score_stats.update(scores.detach()[~torch.isnan(scores)]) # +++ ADDED +++

        mean, std_dev = self.novelty_score_stats.mean, self.novelty_score_stats.std_dev
        if std_dev == 0:
            return torch.zeros_like(scores, dtype=torch.bool)

        threshold = mean + (std_dev * self.config['novelty_z_score_threshold'])
        is_novel_mask = scores > threshold

        # The logging part remains largely the same, as printing requires CPU transfer anyway.
        if is_novel_mask.any():
            cpu_scores = scores.detach().cpu().float().numpy()
            cpu_mask = is_novel_mask.cpu().float().numpy()
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
