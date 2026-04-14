"""Centralized hyperparameters for PhotoMind RL components."""

import os

# ── Bandit Config ────────────────────────────────────────────────────
N_ARMS = 3
ARM_NAMES = ["factual", "semantic", "behavioral"]
N_CONTEXT_CLUSTERS = 4
UCB_EXPLORATION_CONSTANT = 2.0
EPSILON_GREEDY_EPSILON = 0.1

# ── DQN Config (adapted from LunarLander) ───────────────────────────
DQN_STATE_DIM = 8
DQN_ACTION_DIM = 4
DQN_HIDDEN_DIM = 64
DQN_LR = 5e-4
DQN_GAMMA = 0.99
DQN_EPSILON_START = 1.0
DQN_EPSILON_MIN = 0.01
DQN_EPSILON_DECAY = 0.995
DQN_BUFFER_SIZE = 10_000
DQN_BATCH_SIZE = 32
DQN_TAU = 1e-3
DQN_UPDATE_EVERY = 4

ACTION_NAMES = ["accept_high", "accept_moderate", "hedge", "decline"]

# ── Training Config ──────────────────────────────────────────────────
N_TRAINING_EPISODES = 2000
N_SEEDS = 5
SEEDS = [42, 123, 456, 789, 1024]
AUGMENTATION_FACTOR = 10

# ── Reward Matrix ────────────────────────────────────────────────────
# Maps (action_idx, retrieval_correct, should_decline) → reward
REWARD_MATRIX = {
    # accept_high (action 0)
    (0, True, False): +1.0,    # correct retrieval, high confidence — perfect
    (0, False, False): -1.0,   # wrong retrieval, high confidence — SILENT FAILURE
    (0, True, True): -1.0,     # should have declined — overconfident
    (0, False, True): -1.0,    # should have declined — overconfident and wrong
    # accept_moderate (action 1)
    (1, True, False): +0.7,    # correct, moderate confidence — good
    (1, False, False): -0.5,   # wrong, moderate confidence — bad
    (1, True, True): -0.7,     # should have declined
    (1, False, True): -0.7,    # should have declined
    # hedge (action 2)
    (2, True, False): +0.3,    # correct but hedged — too conservative
    (2, False, False): +0.2,   # wrong but hedged — appropriate caution
    (2, True, True): +0.3,     # should decline, hedged — close
    (2, False, True): +0.3,    # should decline, hedged — close
    # decline (action 3)
    (3, True, False): -0.3,    # correct retrieval but declined — unnecessary
    (3, False, False): +0.5,   # wrong retrieval, declined — appropriate
    (3, True, True): +1.0,     # should decline, did decline — perfect
    (3, False, True): +1.0,    # should decline, did decline — perfect
}

# ── Model Paths ──────────────────────────────────────────────────────
_BASE = os.path.join(os.path.dirname(__file__), "..", "..", "knowledge_base", "rl_models")
BANDIT_MODEL_PATH = os.path.join(_BASE, "bandit_thompson.pkl")
DQN_MODEL_PATH = os.path.join(_BASE, "dqn_confidence.pth")
BANDIT_CLUSTERS_PATH = os.path.join(_BASE, "bandit_clusters.pkl")
