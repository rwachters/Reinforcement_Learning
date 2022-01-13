from utilities import convert_to_tensor
import numpy as np
import torch
from collections import deque
import random
from typing import Tuple, Deque, Union


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, size: int):
        """Initialize a ReplayBuffer object.

            :param size: maximum size of buffer
        """
        self.deque: Deque[Tuple[np.ndarray, ...]] = deque(maxlen=size)

    def add(self, sample: Tuple[np.ndarray, ...]):
        """Add a new sample to the buffer."""
        self.deque.append(sample)

    def sample(self, batch_size: int) -> Tuple[torch.Tensor, ...]:
        """Randomly sample a batch of samples from the buffer."""
        samples = random.sample(self.deque, k=batch_size)
        samples_transposed = tuple(zip(*samples))
        return tuple(convert_to_tensor(np.stack(np_array_list)) for np_array_list in samples_transposed)

    def __len__(self):
        """Return the current size of the buffer."""
        return len(self.deque)
