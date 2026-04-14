"""Contextual bandit implementations for query routing (RL Approach 1).

Three bandit algorithms that learn which search strategy (factual/semantic/
behavioral) works best for each query context:
  - ThompsonSamplingBandit: Beta posteriors, provably optimal exploration
  - UCBBandit: UCB1 upper confidence bound
  - EpsilonGreedyBandit: baseline comparison
"""

import os
import pickle
import numpy as np
from abc import ABC, abstractmethod
from sklearn.cluster import KMeans

from src.rl.rl_config import (
    N_ARMS, N_CONTEXT_CLUSTERS, UCB_EXPLORATION_CONSTANT, EPSILON_GREEDY_EPSILON,
)


class ContextualBandit(ABC):
    """Base class for contextual bandit algorithms."""

    def __init__(self, n_arms: int = N_ARMS, n_clusters: int = N_CONTEXT_CLUSTERS):
        self.n_arms = n_arms
        self.n_clusters = n_clusters
        self.kmeans = None

    def fit_clusters(self, all_features: np.ndarray) -> None:
        """Fit KMeans on the query feature space to define context clusters."""
        n_samples = all_features.shape[0]
        k = min(self.n_clusters, n_samples)
        self.kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
        self.kmeans.fit(all_features)
        self.n_clusters = k

    def _get_cluster(self, features: np.ndarray) -> int:
        """Map a feature vector to its nearest cluster."""
        if self.kmeans is None:
            return 0
        return int(self.kmeans.predict(features.reshape(1, -1))[0])

    @abstractmethod
    def select_arm(self, features: np.ndarray) -> int:
        """Choose an arm given the context features."""

    @abstractmethod
    def update(self, features: np.ndarray, arm: int, reward: float) -> None:
        """Update internal state after observing a reward."""

    def save(self, path: str) -> None:
        """Persist bandit state to disk."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @classmethod
    def load(cls, path: str) -> "ContextualBandit":
        """Load bandit state from disk."""
        with open(path, "rb") as f:
            return pickle.load(f)


class ThompsonSamplingBandit(ContextualBandit):
    """Thompson Sampling with Beta posteriors per context cluster."""

    def __init__(self, n_arms: int = N_ARMS, n_clusters: int = N_CONTEXT_CLUSTERS):
        super().__init__(n_arms, n_clusters)
        self.alpha = np.ones((n_clusters, n_arms))  # success counts
        self.beta_param = np.ones((n_clusters, n_arms))  # failure counts

    def select_arm(self, features: np.ndarray) -> int:
        c = self._get_cluster(features)
        # Sample from Beta posterior for each arm
        samples = np.array([
            np.random.beta(self.alpha[c, a], self.beta_param[c, a])
            for a in range(self.n_arms)
        ])
        return int(np.argmax(samples))

    def update(self, features: np.ndarray, arm: int, reward: float) -> None:
        c = self._get_cluster(features)
        # Convert continuous reward to binary for Beta update
        success = reward > 0.5
        if success:
            self.alpha[c, arm] += 1.0
        else:
            self.beta_param[c, arm] += 1.0

    def get_posteriors(self) -> dict:
        """Return posterior parameters for visualization."""
        return {
            "alpha": self.alpha.copy(),
            "beta": self.beta_param.copy(),
        }


class UCBBandit(ContextualBandit):
    """UCB1 algorithm per context cluster."""

    def __init__(self, n_arms: int = N_ARMS, n_clusters: int = N_CONTEXT_CLUSTERS,
                 c: float = UCB_EXPLORATION_CONSTANT):
        super().__init__(n_arms, n_clusters)
        self.c = c
        self.Q = np.zeros((n_clusters, n_arms))       # mean rewards
        self.N = np.zeros((n_clusters, n_arms))        # pull counts
        self.total_N = np.zeros(n_clusters)            # total pulls per cluster

    def select_arm(self, features: np.ndarray) -> int:
        cluster = self._get_cluster(features)

        # Force exploration of untried arms before incrementing counts.
        # total_N must reflect only pulls that feed the UCB formula, not
        # forced-exploration selections where N[cluster, a] was still 0.
        for a in range(self.n_arms):
            if self.N[cluster, a] == 0:
                return a

        self.total_N[cluster] += 1
        ucb_values = self.Q[cluster] + self.c * np.sqrt(
            np.log(self.total_N[cluster]) / self.N[cluster]
        )
        return int(np.argmax(ucb_values))

    def update(self, features: np.ndarray, arm: int, reward: float) -> None:
        cluster = self._get_cluster(features)
        self.N[cluster, arm] += 1
        # Incremental mean update
        self.Q[cluster, arm] += (reward - self.Q[cluster, arm]) / self.N[cluster, arm]


class EpsilonGreedyBandit(ContextualBandit):
    """Epsilon-greedy baseline per context cluster."""

    def __init__(self, n_arms: int = N_ARMS, n_clusters: int = N_CONTEXT_CLUSTERS,
                 epsilon: float = EPSILON_GREEDY_EPSILON):
        super().__init__(n_arms, n_clusters)
        self.epsilon = epsilon
        self.Q = np.zeros((n_clusters, n_arms))
        self.N = np.zeros((n_clusters, n_arms))

    def select_arm(self, features: np.ndarray) -> int:
        cluster = self._get_cluster(features)
        if np.random.random() < self.epsilon:
            return np.random.randint(self.n_arms)
        return int(np.argmax(self.Q[cluster]))

    def update(self, features: np.ndarray, arm: int, reward: float) -> None:
        cluster = self._get_cluster(features)
        self.N[cluster, arm] += 1
        self.Q[cluster, arm] += (reward - self.Q[cluster, arm]) / self.N[cluster, arm]


def load_trained_bandit(path: str) -> ContextualBandit | None:
    """Load a trained bandit model, returning None if not found."""
    if not os.path.exists(path):
        return None
    try:
        return ContextualBandit.load(path)
    except Exception:
        return None
