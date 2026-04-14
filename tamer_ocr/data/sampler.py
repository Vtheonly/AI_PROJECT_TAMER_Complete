"""
Dynamic Temperature Sampler and Multi-Dataset Batch Sampler.

Implements the temperature-based dataset balancing strategy:
  P(i) ∝ (n_i)^T
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

    def set_temperature(self, temperature: float):
        """Update the temperature and recompute weights."""
        self.temperature = temperature
        self._compute_weights()

    def __iter__(self):
        indices = []
        names = list(self.dataset_ranges.keys())
        probs = [self.dataset_probs[n] for n in names]

        if not names or sum(probs) == 0:
            return iter([])

        total_samples = sum(self.dataset_sizes.values())

        for name, prob in zip(names, probs):
            start, end = self.dataset_ranges[name]
            size = self.dataset_sizes[name]
            if size == 0:
                continue

            n_samples = max(1, int(total_samples * prob))
            dataset_indices = list(range(start, end))
            
            if n_samples > size:
                # Oversample
                sampled = self.rng.choices(dataset_indices, k=n_samples)
            else:
                # Undersample
                if self.shuffle_within:
                    self.rng.shuffle(dataset_indices)
                sampled = dataset_indices[:n_samples]

            indices.extend(sampled)

        self.rng.shuffle(indices)
        return iter(indices)

    def __len__(self):
        return sum(self.dataset_sizes.values())


class MultiDatasetBatchSampler:
    """
    Creates batches where each batch comes from a SINGLE dataset.
    This improves training stability.
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

        self.dataset_sizes = {}
        for name, (start, end) in dataset_ranges.items():
            size = end - start
            # Safety check for batch size
            if size < batch_size and size > 0:
                logger.warning(f"Dataset {name} size ({size}) is smaller than batch_size ({batch_size}).")
            self.dataset_sizes[name] = max(0, size)

        self._compute_probs()

        # Pre-generate per-dataset index pools
        self._index_pools = {}
        for name, (start, end) in dataset_ranges.items():
            self._index_pools[name] = list(range(start, end))

    def _compute_probs(self):
        raw_weights = {}
        for name, size in self.dataset_sizes.items():
            raw_weights[name] = (size ** self.temperature) if size > 0 else 0

        weight_sum = sum(raw_weights.values())
        if weight_sum == 0:
            self.dataset_probs = {name: 0.0 for name in self.dataset_sizes}
        else:
            self.dataset_probs = {name: w / weight_sum for name, w in raw_weights.items()}

    def set_temperature(self, temperature: float):
        self.temperature = temperature
        self._compute_probs()

    def __iter__(self):
        total_samples = sum(self.dataset_sizes.values())
        total_batches = total_samples // self.batch_size

        names = list(self.dataset_ranges.keys())
        probs = [self.dataset_probs[n] for n in names]

        if not names or sum(probs) == 0:
            return

        # Prepare shuffled pools
        shuffled_pools = {}
        for name in names:
            pool = self._index_pools[name][:]
            self.rng.shuffle(pool)
            shuffled_pools[name] = pool

        pool_offsets = {name: 0 for name in names}

        for _ in range(total_batches):
            # Weighted selection of dataset
            chosen = self.rng.choices(names, weights=probs, k=1)[0]

            # Ensure we don't pick an empty dataset
            if self.dataset_sizes[chosen] < self.batch_size:
                # Attempt to pick another dataset if possible
                valid_names = [n for n in names if self.dataset_sizes[n] >= self.batch_size]
                if not valid_names: break
                chosen = self.rng.choice(valid_names)

            start_offset = pool_offsets[chosen]
            end_offset = start_offset + self.batch_size

            # Reshuffle if pool exhausted
            if end_offset > len(shuffled_pools[chosen]):
                pool = self._index_pools[chosen][:]
                self.rng.shuffle(pool)
                shuffled_pools[chosen] = pool
                start_offset = 0
                end_offset = self.batch_size

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
    """Linearly decay temperature."""
    progress = min(1.0, current_step / max(total_steps, 1))
    return temp_start + (temp_end - temp_start) * progress