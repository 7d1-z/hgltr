import random
import torch
import numpy as np
from torch.utils.data import Sampler


class RandomCycleIter:
    def __init__(self, data, test_mode=False):
        self.data_list = list(data)
        self.length = len(self.data_list)
        self.index = self.length - 1
        self.test_mode = test_mode

    def __iter__(self):
        return self

    def __next__(self):
        self.index += 1
        if self.index == self.length:
            self.index = 0
            if not self.test_mode:
                random.shuffle(self.data_list)
        return self.data_list[self.index]


def class_aware_sample_generator(class_iter, class_sample_iters, total_samples, samples_per_class=1):
    sample_count = 0
    intra_class_idx = 0
    current_class_batch = None

    while sample_count < total_samples:
        if intra_class_idx == 0:
            class_id = next(class_iter)
            current_class_batch = [
                next(class_sample_iters[class_id]) for _ in range(samples_per_class)
            ]
        yield current_class_batch[intra_class_idx]

        sample_count += 1
        intra_class_idx = (intra_class_idx + 1) % samples_per_class


class ClassAwareSampler(Sampler):
    def __init__(self, targets, samples_per_class=1, test_mode=False):
        """
        Args:
            targets: list or array, label for each sample
            samples_per_class: int, the number of samples per class
            test_mode: bool, if True, not shuffle
        """
        super().__init__()
        self.num_classes = len(np.unique(targets))
        self.class_iter = RandomCycleIter(range(self.num_classes), test_mode=test_mode)

        class_to_indices = [[] for _ in range(self.num_classes)]
        for idx, label in enumerate(targets):
            class_to_indices[label].append(idx)

        self.class_sample_iters = [
            RandomCycleIter(indices, test_mode=test_mode) for indices in class_to_indices
        ]

        max_class_size = max(len(indices) for indices in class_to_indices)
        self.total_samples = max_class_size * self.num_classes
        self.samples_per_class = samples_per_class

    def __iter__(self):
        return class_aware_sample_generator(
            self.class_iter,
            self.class_sample_iters,
            self.total_samples,
            self.samples_per_class
        )

    def __len__(self):
        return self.total_samples



class BalancedDatasetSampler(Sampler):

    def __init__(self, target, indices=None, num_samples=None):
                
        # if indices is not provided, 
        # all elements in the dataset will be considered
        self.indices = list(range(len(target))) \
            if indices is None else indices
            
        # if num_samples is not provided, 
        # draw `len(indices)` samples in each iteration
        self.num_samples = len(self.indices) \
            if num_samples is None else num_samples

        # distribution of classes in the dataset
        label_to_count = [0] * len(np.unique(target))
        for idx in self.indices:
            label = self._get_label(target, idx)
            label_to_count[label] += 1
    
        per_cls_weights = 1 / np.array(label_to_count)

        # weight for each sample
        weights = [per_cls_weights[self._get_label(target, idx)]
                   for idx in self.indices]
        
        
        self.weights = torch.DoubleTensor(weights)

    def _get_label(self, target, idx):
        return target[idx]

    def __iter__(self):
        return iter(torch.multinomial(self.weights, self.num_samples, replacement=True).tolist())

    def __len__(self):
        return self.num_samples