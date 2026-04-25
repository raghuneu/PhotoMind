"""Experience replay buffer — adapted from LunarLander DQN notebook."""

import random
import pickle
import numpy as np
import torch
from collections import deque, namedtuple

Experience = namedtuple(
    "Experience",
    field_names=["state", "action", "reward", "next_state", "done"],
)


class ReplayBuffer:
    """Fixed-size buffer to store experience tuples."""

    def __init__(self, buffer_size: int, batch_size: int):
        self.memory = deque(maxlen=buffer_size)
        self.batch_size = batch_size

    def add(self, state: np.ndarray, action: int, reward: float,
            next_state: np.ndarray, done: bool) -> None:
        """Add a new experience to memory."""
        self.memory.append(Experience(state, action, reward, next_state, done))

    def sample(self):
        """Randomly sample a batch of experiences from memory."""
        experiences = random.sample(self.memory, k=self.batch_size)

        states = torch.FloatTensor(
            np.vstack([e.state for e in experiences])
        )
        actions = torch.LongTensor(
            np.vstack([e.action for e in experiences])
        )
        rewards = torch.FloatTensor(
            np.vstack([e.reward for e in experiences])
        )
        next_states = torch.FloatTensor(
            np.vstack([e.next_state for e in experiences])
        )
        dones = torch.FloatTensor(
            np.vstack([e.done for e in experiences])
        )

        return states, actions, rewards, next_states, dones

    def __len__(self) -> int:
        return len(self.memory)

    def save(self, path: str) -> None:
        """Persist buffer to disk."""
        with open(path, "wb") as f:
            pickle.dump(list(self.memory), f)

    @classmethod
    def load(cls, path: str, buffer_size: int, batch_size: int) -> "ReplayBuffer":
        """Load buffer from disk."""
        buf = cls(buffer_size, batch_size)
        with open(path, "rb") as f:
            data = pickle.load(f)
        for exp in data:
            buf.memory.append(exp)
        return buf
