"""DQN confidence calibrator (RL Approach 2).

Learns when to accept (high/moderate confidence), hedge, or decline
retrieval results. Architecture adapted from LunarLander DQN notebook.
"""

import os
import random
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from src.rl.replay_buffer import ReplayBuffer
from src.rl.rl_config import (
    DQN_STATE_DIM, DQN_ACTION_DIM, DQN_HIDDEN_DIM,
    DQN_LR, DQN_GAMMA, DQN_EPSILON_START, DQN_EPSILON_MIN,
    DQN_EPSILON_DECAY, DQN_BUFFER_SIZE, DQN_BATCH_SIZE,
    DQN_TAU, DQN_UPDATE_EVERY, ACTION_NAMES,
)


class ConfidenceState:
    """Build an 8-dim state vector from retrieval results."""

    @staticmethod
    def from_retrieval(
        results: list,
        strategy_idx: int,
        query_features: np.ndarray,
    ) -> np.ndarray:
        """Convert retrieval results into an 8-dimensional state vector."""
        state = np.zeros(8, dtype=np.float32)

        scores = [r.get("relevance_score", 0.0) for r in results]
        scores_sorted = sorted(scores, reverse=True)

        # 0: top score
        state[0] = scores_sorted[0] if scores_sorted else 0.0
        # 1: score gap (top - second)
        state[1] = (scores_sorted[0] - scores_sorted[1]) if len(scores_sorted) >= 2 else 0.0
        # 2: number of results (normalized)
        state[2] = min(len(results) / 10.0, 1.0)
        # 3: strategy index (normalized to [0, 1])
        state[3] = strategy_idx / 2.0
        # 4: query length (from features)
        state[4] = query_features[0] if len(query_features) > 0 else 0.0
        # 5: has exact entity match
        state[5] = 1.0 if any("entity" in r.get("evidence", "").lower() or
                              "vendor" in r.get("evidence", "").lower() or
                              "amount" in r.get("evidence", "").lower()
                              for r in results) else 0.0
        # 6: type matches strategy expectation
        strategy_type_map = {0: "receipt", 1: "food", 2: "food"}
        expected_type = strategy_type_map.get(strategy_idx, "")
        if results and results[0].get("image_type", "") == expected_type:
            state[6] = 1.0
        # 7: average score
        state[7] = np.mean(scores) if scores else 0.0

        return state


class ConfidenceDQN(nn.Module):
    """Q-value network — identical architecture to LunarLander's DeepQNetwork.

    FC(8→64) → ReLU → FC(64→64) → ReLU → FC(64→4)
    """

    def __init__(self, state_size: int = DQN_STATE_DIM,
                 action_size: int = DQN_ACTION_DIM,
                 hidden_size: int = DQN_HIDDEN_DIM):
        super().__init__()
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, action_size)

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return self.fc3(x)


class ConfidenceDQNAgent:
    """DQN agent for confidence calibration — adapted from LunarLander's DQNAgent."""

    def __init__(
        self,
        state_size: int = DQN_STATE_DIM,
        action_size: int = DQN_ACTION_DIM,
        learning_rate: float = DQN_LR,
        gamma: float = DQN_GAMMA,
        epsilon: float = DQN_EPSILON_START,
        epsilon_min: float = DQN_EPSILON_MIN,
        epsilon_decay: float = DQN_EPSILON_DECAY,
        buffer_size: int = DQN_BUFFER_SIZE,
        batch_size: int = DQN_BATCH_SIZE,
        update_every: int = DQN_UPDATE_EVERY,
        tau: float = DQN_TAU,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.tau = tau
        self.update_every = update_every
        self.batch_size = batch_size

        # Networks
        self.online_network = ConfidenceDQN(state_size, action_size)
        self.target_network = ConfidenceDQN(state_size, action_size)
        self.target_network.load_state_dict(self.online_network.state_dict())

        self.optimizer = optim.Adam(self.online_network.parameters(), lr=learning_rate)
        self.memory = ReplayBuffer(buffer_size, batch_size)
        self.step_count = 0

    def select_action(self, state: np.ndarray) -> int:
        """Epsilon-greedy action selection."""
        if random.random() < self.epsilon:
            return random.randint(0, self.action_size - 1)

        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        self.online_network.eval()
        with torch.no_grad():
            q_values = self.online_network(state_tensor).cpu().numpy().flatten()
        self.online_network.train()
        return int(np.argmax(q_values))

    def step(self, state: np.ndarray, action: int, reward: float,
             next_state: np.ndarray, done: bool) -> float | None:
        """Store experience and learn every update_every steps. Returns loss if learned."""
        self.memory.add(state, action, reward, next_state, done)
        self.step_count += 1

        if (self.step_count % self.update_every == 0
                and len(self.memory) >= self.batch_size):
            return self._learn()
        return None

    def _learn(self) -> float:
        """Update online network using a mini-batch from replay buffer."""
        states, actions, rewards, next_states, dones = self.memory.sample()

        # Compute target Q-values
        with torch.no_grad():
            q_targets_next = self.target_network(next_states).detach().max(1)[0].unsqueeze(1)
            q_targets = rewards + (self.gamma * q_targets_next * (1 - dones))

        # Compute current Q-values
        q_expected = self.online_network(states).gather(1, actions)

        # Loss and backprop
        loss = F.mse_loss(q_expected, q_targets)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        self._soft_update()
        return loss.item()

    def _soft_update(self) -> None:
        """θ_target ← τ·θ_online + (1-τ)·θ_target"""
        for target_param, online_param in zip(
            self.target_network.parameters(),
            self.online_network.parameters(),
        ):
            target_param.data.copy_(
                self.tau * online_param.data + (1.0 - self.tau) * target_param.data
            )

    def decay_epsilon(self) -> None:
        """Decay epsilon after each episode."""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, path: str) -> None:
        """Save online network weights."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        torch.save(self.online_network.state_dict(), path)

    def load(self, path: str) -> None:
        """Load online network weights and sync target network."""
        self.online_network.load_state_dict(
            torch.load(path, map_location="cpu", weights_only=True)
        )
        self.target_network.load_state_dict(self.online_network.state_dict())


def action_to_grade(action: int, score: float = 0.5) -> str:
    """Map DQN action index to confidence grade letter."""
    if action == 0:  # accept_high
        return "A" if score >= 0.5 else "B"
    elif action == 1:  # accept_moderate
        return "C"
    elif action == 2:  # hedge
        return "D"
    else:  # decline
        return "F"


def load_trained_dqn(path: str) -> ConfidenceDQNAgent | None:
    """Load a trained DQN agent, returning None if not found."""
    if not os.path.exists(path):
        return None
    try:
        agent = ConfidenceDQNAgent()
        agent.load(path)
        agent.epsilon = 0.0  # Pure exploitation at inference
        return agent
    except Exception:
        return None
