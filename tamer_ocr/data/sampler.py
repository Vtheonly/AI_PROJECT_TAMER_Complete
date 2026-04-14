"""
Dynamic Temperature Sampler and Multi-Dataset Batch Sampler.

Implements the temperature-based dataset balancing strategy:
  P(i) ∝ (n_i)^T

Where n_i is the number of samples in dataset i, and T is the temperature.
- T < 1: Upweights small datasets (CROHME) — good for early training
- T = 1: Uniform sampling proportional to dataset size
- T > 1: Upweights large datasets (Im2LaTeX)

The temperature decays from temp_start to temp_end over training.
"""

import random
import math
import logging
from typing import List, Dict, Optional
from torch.utils.data import Sampler, BatchSampler

logger = logging.getLogger("TAMER.Sampler")


class TemperatureSampler(Sampler):
    """
    Samples dataset indices with temperature-based weighting.
    
    Usage:
        sampler = TemperatureSampler(dataset_counts, total_length, temp=0.8)
        # Returns a flat list of global indices, weighted by temperature
    """
    def __init__(
        self,
        dataset_ranges: Dict[str, tuple],  # {name: (start_idx, end_idx)}
        temperature: float = 0.8,
        shuffle_within: bool = True,
    ):
        """
        Args:
            dataset_ranges: Dict mapping dataset name to (start_idx, end_idx) range
            temperature: Temperature parameter T for P(i) ∝ (n_i)^T
            shuffle_within: Whether to shuffle indices within each dataset
        """
        self.dataset_ranges = dataset_ranges
        self.temperature = temperature
        self.shuffle_within = shuffle_within

        # Calculate dataset sizes
        self.dataset_sizes = {}
        for name, (start, end) in dataset_ranges.items():
            self.dataset_sizes[name] = end - start

        # Calculate sampling weights
        self._compute_weights()

    def _compute_weights(self):
        """Compute dataset selection probabilities based on temperature."""
        total = sum(self.dataset_sizes.values())
        if total == 0:
            self.dataset_probs = {}
            return

        # P(i) ∝ (n_i)^T, then normalize
        raw_weights = {}
        for name, size in self.dataset_sizes.items():
            raw_weights[name] = (size ** self.temperature) if size > 0 else 0

        weight_sum = sum(raw_weights.values())
        if weight_sum == 0:
            self.dataset_probs = {name: 0.0 for name in self.dataset_sizes}
        else:
            self.dataset_probs = {name: w / weight_sum for name, w in raw_weights.items()}

        logger.info(f"Temperature={self.temperature:.2f} → Probs: {self.dataset_probs}")

    def set_temperature(self, temperature: float):
        """Update the temperature and recompute weights."""
        self.temperature = temperature
        self._compute_weights()

    def __iter__(self):
        # Build a flat index list weighted by temperature
        indices = []
        names = list(self.dataset_ranges.keys())
        probs = [self.dataset_probs[n] for n in names]

        if not names or sum(probs) == 0:
            return iter([])

        # Total number of samples to produce
        total_samples = sum(self.dataset_sizes.values())

        # Allocate samples per dataset based on temperature probabilities
        for name, prob in zip(names, probs):
            start, end = self.dataset_ranges[name]
            size = end - start
            if size == 0:
                continue

            # How many samples from this dataset
            n_samples = max(1, int(total_samples * prob))
            # Oversample/undersample to match temperature weight
            if n_samples > size:
                # Oversample: repeat with replacement
                dataset_indices = list(range(start, end))
                sampled = random.choices(dataset_indices, k=n_samples)
            else:
                # Undersample: take a subset
                dataset_indices = list(range(start, end))
                if self.shuffle_within:
                    random.shuffle(dataset_indices)
                sampled = dataset_indices[:n_samples]

            indices.extend(sampled)

        # Shuffle the combined list
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return sum(self.dataset_sizes.values())


class MultiDatasetBatchSampler:
    """
    Creates batches where each batch comes from a SINGLE dataset.
    
    This improves training stability compared to mixed-dataset batches.
    Dataset selection follows the temperature-weighted probabilities.
    
    Usage:
        batch_sampler = MultiDatasetBatchSampler(
            dataset_ranges={'im2latex': (0, 100000), 'crohme': (100000, 110000)},
            batch_size=8,
            temperature=0.8,
            drop_last=False
        )
        for batch_indices in batch_sampler:
            # batch_indices all belong to the same dataset
    """
    def __init__(
        self,
        dataset_ranges: Dict[str, tuple],
        batch_size: int,
        temperature: float = 0.8,
        drop_last: bool = False,
    ):
        self.dataset_ranges = dataset_ranges
        self.batch_size = batch_size
        self.temperature = temperature
        self.drop_last = drop_last

        # Pre-compute dataset sizes
        self.dataset_sizes = {}
        for name, (start, end) in dataset_ranges.items():
            self.dataset_sizes[name] = end - start

        # Compute temperature probabilities
        self._compute_probs()

        # Pre-generate per-dataset index pools
        self._index_pools = {}
        for name, (start, end) in dataset_ranges.items():
            self._index_pools[name] = list(range(start, end))

    def _compute_probs(self):
        """Compute dataset selection probabilities."""
        raw_weights = {}
        for name, size in self.dataset_sizes.items():
            raw_weights[name] = (size ** self.temperature) if size > 0 else 0

        weight_sum = sum(raw_weights.values())
        if weight_sum == 0:
            self.dataset_probs = {name: 0.0 for name in self.dataset_sizes}
        else:
            self.dataset_probs = {name: w / weight_sum for name, w in raw_weights.items()}

    def set_temperature(self, temperature: float):
        """Update the temperature."""
        self.temperature = temperature
        self._compute_probs()

    def __iter__(self):
        # Calculate total number of batches
        total_samples = sum(self.dataset_sizes.values())
        total_batches = total_samples // self.batch_size

        names = list(self.dataset_ranges.keys())
        probs = [self.dataset_probs[n] for n in names]

        if not names or sum(probs) == 0:
            return

        # Shuffle each dataset's index pool
        shuffled_pools = {}
        for name in names:
            pool = self._index_pools[name][:]
            random.shuffle(pool)
            shuffled_pools[name] = pool

        pool_offsets = {name: 0 for name in names}

        for _ in range(total_batches):
            # Select which dataset this batch comes from
            chosen = random.choices(names, weights=probs, k=1)[0]

            start_offset = pool_offsets[chosen]
            end_offset = start_offset + self.batch_size

            # If we've exhausted this dataset's pool, reshuffle
            if end_offset > len(shuffled_pools[chosen]):
                pool = self._index_pools[chosen][:]
                random.shuffle(pool)
                shuffled_pools[chosen] = pool
                start_offset = 0
                end_offset = self.batch_size

            if end_offset > len(shuffled_pools[chosen]):
                # Dataset is smaller than batch size, skip
                continue

            batch_indices = shuffled_pools[chosen][start_offset:end_offset]
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
    Linearly decay temperature from temp_start to temp_end over training.
    
    Early training: T=0.8 → upweights small datasets (CROHME, MathWriting)
    Late training: T=0.4 → more uniform, prevents forgetting printed data
    """
    progress = min(1.0, current_step / max(total_steps, 1))
    return temp_start + (temp_end - temp_start) * progress
