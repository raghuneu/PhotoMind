# PhotoMind RL: Mathematical Formulations

LaTeX-ready formulations for the technical report and presentation slides.

---

## 1. MDP Definition (DQN Confidence Calibrator)

The confidence calibration problem is formulated as a single-step Markov Decision Process:

- **State space** S ⊆ ℝ^8: retrieval context vectors
  - s = [top_score, score_gap, num_results_norm, strategy_idx_norm, query_length_norm, entity_match, type_match, avg_score]
- **Action space** A = {accept_high, accept_moderate, hedge, decline} = {0, 1, 2, 3}
- **Reward function** R: S × A → ℝ (see reward matrix in rl_config.py)
- **Discount factor** γ = 0.99 (structurally irrelevant for single-step episodes; retained for architectural consistency with LunarLander baseline)
- **Episodes**: single-step (done = True after every action)

Since episodes are single-step, the Bellman optimality equation reduces to:

    Q*(s, a) = R(s, a)    for all s ∈ S, a ∈ A

The optimal policy is: π*(s) = argmax_a R(s, a)

---

## 2. DQN Q-Value Function

The Q-network approximates Q*(s, a) using a fully connected neural network:

    Q(s, a; θ) = FC_3(ReLU(FC_2(ReLU(FC_1(s)))))

where FC_i are linear layers with:
- FC_1: ℝ^8 → ℝ^64
- FC_2: ℝ^64 → ℝ^64
- FC_3: ℝ^64 → ℝ^4

**TD target** (with terminal next state):

    y = R(s, a)    [since next_state is terminal, (1 - done) = 0]

**Loss function:**

    L(θ) = E_{(s,a,r) ~ D}[(y - Q(s, a; θ))^2]

where D is the replay buffer.

**Soft update:**

    θ_target ← τ · θ_online + (1 − τ) · θ_target,    τ = 0.001

---

## 3. Contextual Bandit: Thompson Sampling

**Context clustering:** c = argmin_{k=1..K} ||φ(q) − μ_k||²

where φ: Q → ℝ^12 is the feature extractor and μ_k are KMeans centroids (K=4).

**Posterior:** For each context cluster c and arm a, maintain Beta posteriors:

    P(θ_{c,a}) = Beta(α_{c,a}, β_{c,a})

initialized with α_{c,a} = β_{c,a} = 1 (uniform prior).

**Arm selection:**

    ã_{c,a} ~ Beta(α_{c,a}, β_{c,a})    for each a ∈ {0, 1, 2}
    a* = argmax_a ã_{c,a}

**Posterior update** (binary reward, threshold = 0.5):

    if r > 0.5:  α_{c,a*} ← α_{c,a*} + 1
    else:        β_{c,a*} ← β_{c,a*} + 1

---

## 4. UCB1 Algorithm

**UCB score per arm:**

    UCB(c, a) = Q̄(c, a) + C · sqrt(ln(N_c) / N_{c,a})

where:
- Q̄(c, a) = running mean reward for arm a in cluster c
- N_c = total pulls in cluster c (N_c = sum_a N_{c,a}; incremented only after all arms in the cluster have been explored at least once, ensuring the formula is never evaluated with N_{c,a} = 0)
- N_{c,a} = pulls of arm a in cluster c
- C = 2.0 (exploration constant)

**Incremental mean update:**

    Q̄(c, a) ← Q̄(c, a) + (r − Q̄(c, a)) / N_{c,a}

---

## 5. Bandit Reward Function

    R_bandit(a, y, expected_type) =
        1.0    if ARM_NAMES[a] == expected_type AND expected_photo in results
        0.5    if expected_photo in results but ARM_NAMES[a] != expected_type
        0.3    if expected_photo is None (aggregate query) AND ARM_NAMES[a] == expected_type
        0.0    otherwise

---

## 6. Feature Extractor

Query feature vector φ(q) ∈ ℝ^12:

    φ(q) = [
        len(q) / 100.0,                    # query length (normalized)
        |words(q)| / 20.0,                 # word count (normalized)
        1[has_amount_keyword(q)],           # price/spend/cost signals
        1[has_date_keyword(q)],             # temporal signals
        1[has_vendor_keyword(q)],           # receipt/bill/store signals
        1[has_behavioral_keyword(q)],       # frequency/pattern signals
        1[has_semantic_keyword(q)],         # show/find/look-like signals
        1[has_negation(q)],                 # not/never/no signals
        1[is_wh_question(q)],              # what/where/when/how
        1[is_yes_no_question(q)],          # is/are/do/does
        1[is_imperative(q)],               # show/find/get/list
        1[has_known_vendor_match(q)],       # vendor name in KB
    ]

---

## 7. Confidence Interval (t-distribution)

For n observations x_1, ..., x_n with sample mean x̄ and standard error SE = s/√n:

    CI_{95%} = x̄ ± t_{α/2, n-1} · SE

where t_{α/2, n-1} is the critical value from the t-distribution with (n-1) degrees of freedom.

For n = 5 seeds: t_{0.025, 4} ≈ 2.776.

---

## 8. Cohen's d (Paired)

Effect size for paired comparison:

    d = mean(RL - Baseline) / std(RL - Baseline, ddof=1)

Interpretation: |d| < 0.2 small, 0.2–0.5 medium, 0.5–0.8 large, > 0.8 very large.

**Special case (deterministic effect):** When std(RL - Baseline) = 0 but mean(RL - Baseline) ≠ 0,
Cohen's d is mathematically undefined (infinite). This occurs for silent failure rate with Full RL:
all 5 seeds produce identical 0.0% vs 1.8% differences. The effect is perfectly consistent and
its size is unbounded — reported as "inf*" in tables.
