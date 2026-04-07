import random
from torch.utils.data import Sampler

class CurriculumSampler(Sampler):
    """Sorts data by structural complexity, gradually unfreezing harder examples."""
    def __init__(self, dataset, warmup_epochs: int):
        self.dataset = dataset
        self.warmup_epochs = warmup_epochs
        self.epoch = 0
        self.n = len(dataset)
        self.sorted_indices = sorted(range(self.n), key=lambda i: self.dataset.complexities[i])

    def set_epoch(self, epoch: int):
        self.epoch = epoch

    def __iter__(self):
        if self.epoch < self.warmup_epochs:
            fraction = min(1.0, 0.3 + 0.7 * (self.epoch / max(self.warmup_epochs, 1)))
            subset_size = max(1, int(self.n * fraction))
            indices = list(self.sorted_indices[:subset_size])
            random.shuffle(indices)
            return iter(indices)
            
        indices = list(range(self.n))
        random.shuffle(indices)
        return iter(indices)

    def __len__(self):
        if self.epoch < self.warmup_epochs:
            return max(1, int(self.n * min(1.0, 0.3 + 0.7 * (self.epoch / max(self.warmup_epochs, 1)))))
        return self.n