"""
Dynamic Temperature Sampler and Multi-Dataset Batch Sampler.

Implements the temperature-based dataset balancing strategy from the paper:
  P(i) ∝ (n_i)^T
  
Where:
  - P(i) = probability of sampling from dataset i
  - n_i = number of samples in dataset i
  - T = temperature parameter (0 < T ≤ 1)
  
Temperature effects:
  - T = 1.0: Proportional sampling (larger datasets sampled more)
  - T = 0.5: Square root sampling (balances small/large datasets)
  - T → 0.0: Uniform sampling (all datasets sampled equally)

This module is fully offline-compatible (no network dependencies).
"""

import random
import logging
from typing import Dict, List
from torch.utils.data import Sampler

logger = logging.getLogger("TAMER.Sampler")


class TemperatureSampler(Sampler):
    """
    Samples dataset indices with temperature-based weighting.
    
    Each dataset is assigned a sampling probability based on its size
    raised to the power of temperature, then normalized.
    """
    def __init__(
        self,
        dataset_ranges: Dict[str, tuple],  # {name: (start_idx, end_idx)}
        temperature: float = 0.8,
        shuffle_within: bool = True,
        seed: int = 42
    ):
        self.dataset_ranges = dataset_ranges
        self.temperature = temperature
        self.shuffle_within = shuffle_within
        self.rng = random.Random(seed)

        # Calculate dataset sizes
        self.dataset_sizes = {}
        for name, (start, end) in dataset_ranges.items():
            self.dataset_sizes[name] = max(0, end - start)

        self._compute_weights()
        
        logger.info(
            f"TemperatureSampler initialized with T={temperature:.2f} | "
            f"Datasets: {len(self.dataset_sizes)} | "
            f"Total samples: {sum(self.dataset_sizes.values())}"
        )

    def _compute_weights(self):
        """
        Compute dataset selection probabilities based on temperature.
        Formula: P(i) = (n_i)^T / Σ(n_j)^T
        """
        total = sum(self.dataset_sizes.values())
        if total == 0:
            self.dataset_probs = {}
            return

        # Apply temperature: raw_weight = (size)^temperature
        raw_weights = {}
        for name, size in self.dataset_sizes.items():
            raw_weights[name] = (size ** self.temperature) if size > 0 else 0

        # Normalize to get probabilities
        weight_sum = sum(raw_weights.values())
        if weight_sum == 0:
            self.dataset_probs = {name: 0.0 for name in self.dataset_sizes}
        else:
            self.dataset_probs = {name: w / weight_sum for name, w in raw_weights.items()}

    def set_temperature(self, temperature: float):
        """Update the temperature and recompute weights."""
        self.temperature = temperature
        self._compute_weights()
        logger.info(f"Temperature updated to {temperature:.3f}")

    def __iter__(self):
        """
        Generate indices for one epoch according to temperature-weighted probabilities.
        """
        indices = []
        names = list(self.dataset_ranges.keys())
        probs = [self.dataset_probs[n] for n in names]

        if not names or sum(probs) == 0:
            return iter([])

        total_samples = sum(self.dataset_sizes.values())

        # Sample from each dataset according to computed probabilities
        for name, prob in zip(names, probs):
            start, end = self.dataset_ranges[name]
            size = self.dataset_sizes[name]
            if size == 0:
                continue

            # Determine how many samples to draw from this dataset
            n_samples = max(1, int(total_samples * prob))
            dataset_indices = list(range(start, end))
            
            if n_samples > size:
                # Oversample with replacement
                sampled = self.rng.choices(dataset_indices, k=n_samples)
            else:
                # Undersample
                if self.shuffle_within:
                    self.rng.shuffle(dataset_indices)
                sampled = dataset_indices[:n_samples]

            indices.extend(sampled)

        # Final shuffle for randomness
        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return sum(self.dataset_sizes.values())


class MultiDatasetBatchSampler:
    """
    Creates batches where each batch comes from a SINGLE dataset.
    
    This design choice improves training stability by ensuring that:
    1. All images in a batch have similar characteristics
    2. Gradient updates are more coherent
    3. BatchNorm statistics are computed over similar distributions
    
    [FIX] Properly handles epoch boundaries - no data is dropped when
    a dataset pool is exhausted mid-batch.
    """
    def __init__(
        self,
        dataset_ranges: Dict[str, tuple],
        batch_size: int,
        temperature: float = 0.8,
        drop_last: bool = False,
        seed: int = 42
    ):
        self.dataset_ranges = dataset_ranges
        self.batch_size = batch_size
        self.temperature = temperature
        self.drop_last = drop_last
        self.rng = random.Random(seed)

        # Compute dataset sizes and validate
        self.dataset_sizes = {}
        for name, (start, end) in dataset_ranges.items():
            size = end - start
            if size < batch_size and size > 0:
                logger.warning(
                    f"Dataset '{name}' has {size} samples, smaller than "
                    f"batch_size={batch_size}. It may be under-sampled."
                )
            self.dataset_sizes[name] = max(0, size)

        self._compute_probs()

        # Pre-generate per-dataset index pools
        self._index_pools = {}
        for name, (start, end) in dataset_ranges.items():
            self._index_pools[name] = list(range(start, end))

        logger.info(
            f"MultiDatasetBatchSampler: batch_size={batch_size} | "
            f"T={temperature:.2f} | {len(self.dataset_sizes)} datasets"
        )

    def _compute_probs(self):
        """Compute dataset selection probabilities using temperature formula."""
        raw_weights = {}
        for name, size in self.dataset_sizes.items():
            raw_weights[name] = (size ** self.temperature) if size > 0 else 0

        weight_sum = sum(raw_weights.values())
        if weight_sum == 0:
            self.dataset_probs = {name: 0.0 for name in self.dataset_sizes}
        else:
            self.dataset_probs = {name: w / weight_sum for name, w in raw_weights.items()}

    def set_temperature(self, temperature: float):
        """Update temperature and recompute probabilities."""
        self.temperature = temperature
        self._compute_probs()

    def __iter__(self):
        total_samples = sum(self.dataset_sizes.values())
        total_batches = total_samples // self.batch_size

        names = list(self.dataset_ranges.keys())
        probs = [self.dataset_probs[n] for n in names]

        if not names or sum(probs) == 0:
            return

        # Initialize shuffled pools for each dataset
        shuffled_pools = {}
        for name in names:
            pool = self._index_pools[name][:]
            self.rng.shuffle(pool)
            shuffled_pools[name] = pool

        pool_offsets = {name: 0 for name in names}

        for _ in range(total_batches):
            # Weighted selection of dataset for this batch
            chosen = self.rng.choices(names, weights=probs, k=1)[0]

            # Safety check: ensure chosen dataset can fill a batch
            if self.dataset_sizes[chosen] < self.batch_size:
                valid_names = [n for n in names if self.dataset_sizes[n] >= self.batch_size]
                if not valid_names:
                    break
                chosen = self.rng.choice(valid_names)

            start_offset = pool_offsets[chosen]
            end_offset = start_offset + self.batch_size

            # [FIX: Epoch Boundary Data Loss]
            # If we don't have enough samples left in the current pool,
            # we carry over what we have and top up from a freshly shuffled pool
            batch_indices = shuffled_pools[chosen][start_offset:end_offset]

            if len(batch_indices) < self.batch_size:
                # We've exhausted the current pool
                # Save the partial batch
                leftover = batch_indices
                
                # Reshuffle the pool
                pool = self._index_pools[chosen][:]
                self.rng.shuffle(pool)
                shuffled_pools[chosen] = pool
                
                # Top up to complete the batch
                needed = self.batch_size - len(leftover)
                batch_indices = leftover + shuffled_pools[chosen][:needed]
                pool_offsets[chosen] = needed
            else:
                # Normal case: we had enough samples
                pool_offsets[chosen] = end_offset

            yield batch_indices

    def __len__(self):
        total_samples = sum(self.dataset_sizes.values())
        return total_samples // self.batch_size


def get_temperature_for_step(
    current_step: int,
    total_steps: int,
    temp_start: float = 0.8,
    temp_end: float = 0.4,
) -> float:
    """
    Linearly decay temperature over training.
    
    Rationale from the paper:
    - Early training (high T ≈ 0.8): Sample proportionally to dataset size
    - Late training (low T ≈ 0.4): More uniform sampling across datasets
    
    This helps the model:
    1. Learn from large datasets first (efficient early learning)
    2. Refine on small datasets later (better generalization)
    
    Args:
        current_step: Current training step
        total_steps: Total training steps
        temp_start: Starting temperature (default 0.8)
        temp_end: Ending temperature (default 0.4)
    
    Returns:
        Interpolated temperature value
    """
    progress = min(1.0, current_step / max(total_steps, 1))
    return temp_start + (temp_end - temp_start) * progress