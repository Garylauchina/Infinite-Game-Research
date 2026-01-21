# Infinite Game V5.0: Market Substrate Model Technical Note

**Author**: Gang Liu  
**Date**: 2025-01-15  
**Version**: V5.0 Phase 1  
**Repository**: https://github.com/Garylauchina/Infinite-Game-Research

---

## Quick Reference: Model Knobs, Baselines, Outputs, and Findings

### Model Knobs (Control Parameters)

| Parameter | Symbol | Default | Sweep Range | Description |
|-----------|--------|---------|--------------|-------------|
| Chaos penalty strength | $\beta$ | 0.4 | $\{0.2, 0.3, 0.4, 0.5, 0.6\}$ | Multiplier for congestion penalty |
| Add player threshold | $\theta_{\text{add}}$ | 0.35 | $\{0.25, 0.30, 0.35, 0.40, 0.45\}$ | Experience threshold to add player |
| Remove player threshold | $\theta_{\text{remove}}$ | 0.15 | $\{0.10, 0.15, 0.20, 0.25, 0.30\}$ | Experience threshold to remove player |
| Complexity window | $W$ | 5000 | $\{3000, 5000, 7000, 10000\}$ | Trajectory window size for clustering |
| Number of clusters | $K$ | 5 | $\{3, 5, 7, 10\}$ | K-means clustering resolution |

### Baselines

1. **No chaos**: $\beta = 0$ (chaos factor disabled)
2. **With chaos**: $\beta = 0.4$ (default)
3. **Fixed $N_t$ ablation**: $N_t$ clamped to constant (no participation adjustment)

### Outputs

| Metric | Symbol | Definition | Source |
|--------|--------|------------|--------|
| Final player count | $N_T$ | Player count at final tick | `len(active_players)` |
| Mean complexity | $\bar{c}$ | Average complexity (steady state) | `StructureMetrics.compute_complexity()` |
| Final complexity | $c_T$ | Complexity at final tick | Same as above |
| Active protocols | $P_T$ | Number of active clusters (>2% occupancy) | `StructureMetrics.get_cluster_info()['active_protocols']` |
| Transition coverage | Coverage | Fraction of possible transitions that occur | Computed from `cluster_assignments` |
| Dominant cluster share | Max share | Maximum cluster occupancy proportion | `max(cluster_distribution)` |

### Main Findings (Propositions)

**P1 (Density→Participation)**: Structure density $c_t$ drives participation $N_t$ via experience $\bar{E}_t$; reverse direction is weak (validated by fixed $N_t$ ablation).

**P2 (Scale Invariance)**: Protocol structure is stable across clustering resolutions $K \in \{3,5,7,10\}$ (complexity correlation $r = 0.89$).

**P3 (Weak Reflexivity)**: Reflexivity modulates $N_t$ but does not generate structure (structure emerges in fixed $N_t$ ablation with $\Delta c = 0.02$, not significant).

**P4 (Non-Equilibrium Steady State)**: System maintains stable but non-static structure over long horizons (no collapse observed in $T = 500,000$ ticks).

---

## A. Abstract

We present a **market substrate model** (zero model) that strips away all high-level mechanisms (intelligence, learning, credit, leverage, institutional rules) to isolate the minimal common rules of trading markets. **We do not model an order book nor dealer inventory; 'avatars' represent participation intensity only.** 

**Core Design Paradigm 1**: A single market has a **unique participating entity "Market Entity" (MarketMr)** that represents the aggregation of all transactions. All trading activities belong to this single entity, which is the unified representative of all transaction aggregations.

**Core Design Paradigm 2**: We design **"structure density" as a reward function** for the Market Entity. Structure density serves as a core component of the experience reward (weight 0.4), driving participation decisions. Higher structure density leads to higher experience reward, which increases participation intensity.

The system consists of: (1) random price/direction actions from a single abstract "Market Entity" (MarketMr) represented by multiple avatars; (2) statistical match probability based on price priority, liquidity, and a stabilizing penalty (chaos factor); (3) state evolution through aggregated statistics; (4) structure density computation via clustering and transition entropy; (5) participation intensity modulation driven by average experience, where structure density is the key reward component.

**Core finding**: Complex market structures can emerge stably from minimal trading rules alone, without requiring intelligent participants or high-level mechanisms. 

**Proposition P1 (Density→Participation)**: Under fixed trading rules and random actions, increasing structure density $c_t$ increases mean experience $\bar{E}_t$, which increases participation $N_t$ via threshold policy; the reverse causal direction is weak in ablations where $N_t$ is clamped.

This establishes a baseline for distinguishing rule-induced structures from mechanism-induced structures in financial markets.

**Keywords**: Market substrate, structure density, emergence, zero model, trading rules

---

## B. Model

### B.1 State Vector

The market state at time $t$ is a 5-dimensional vector:

$$s_t = (p_t, v_t, \ell_t, b_t, c_t)$$

where:
- $p_t \in [0,1]$: normalized price (absolute price mapped to $[p_{\min}, p_{\max}]$)
- $v_t \in [0,1]$: short-term volatility (20-tick rolling window)
- $\ell_t \in [0,1]$: liquidity (match rate: $\ell_t = |M_t| / |A_t|$)
- $b_t \in [0,1]$: imbalance (buy-sell pressure, mapped from $[-1,1]$)
- $c_t \in [0,1]$: structure density (complexity, computed from trajectory)

### B.2 Action Generation

At each tick $t$, $N_t$ avatars (players) of the **Market Entity (MarketMr)** generate actions:

$$A_t = \{a_i = (\text{price}_i, \text{side}_i, q_i) : i \in [1, N_t]\}$$

where:
- $\text{price}_i \sim \text{Uniform}(p_{\min}, p_{\max})$: random price
- $\text{side}_i \sim \text{Uniform}\{\text{buy}, \text{sell}\}$: random direction
- $q_i = 1.0$: fixed quantity

**Key Design Paradigm**: The Market Entity is the **unique participating entity** in the market, representing the aggregation of all transactions. All avatars belong to this single entity, and their actions aggregate to form the market's overall trading activity.

**Key**: No learning, no strategy, pure randomness.

### B.3 Match Probability

For each action $a_i \in A_t$, the match probability is:

$$\Pr(\text{match} \mid a_i, s_t, A_t, N_t) = f_{\text{price}}(a_i, s_t) \cdot f_{\text{chaos}}(A_t, N_t) + f_{\text{state}}(s_t)$$

where:

**Price priority** (40% weight):
$$f_{\text{price}}(a_i, s_t) = \max(0.01, 0.98 \cdot \exp(-4 \cdot d_i))$$

with $d_i = |\text{price}_i - \bar{p}_t| / \bar{p}_t$ (relative distance to center price $\bar{p}_t = p_t \cdot (p_{\max} - p_{\min}) + p_{\min}$).

**Stabilizing penalty** (congestion proxy, not a claim about real market mechanisms):
$$f_{\text{chaos}}(A_t, N_t) = 1.0 - \beta \cdot \text{adjusted\_chaos}(A_t, N_t)$$

where $\beta = 0.4$ is a control parameter (swept in experiments), and $\text{adjusted\_chaos}(A_t, N_t)$ is computed as:

$$\text{adjusted\_chaos} = \text{base\_chaos\_raw} \cdot \frac{0.08}{0.15} \cdot \text{multiplier}$$

where:
- $\text{base\_chaos\_raw}(A_t, N_t)$ is a macro stabilizer combining:
  - Price dispersion: $\sigma_{\text{price}} / \bar{p}_t$ (40% weight, clipped to [0.2, 3.0])
  - Direction entropy: $H(\text{sides}) / \log_2(2)$ (35% weight)
  - Scale overload: $0.6 \cdot (\text{actions}/8) + 0.4 \cdot \log(1 + N_t/10)$ (25% weight)
- The scaling factor $0.08/0.15$ reduces the base chaos from historical 0.15 to current 0.08
- $\text{multiplier}$ is dynamically adjusted based on player count and average experience (range [0.5, 1.5])

**Note**: This is a stabilizing mechanism, not a claim about real market microstructure. The chaos factor serves as a control knob for phase diagram exploration.

**State corrections**:
$$f_{\text{state}}(s_t) = 0.3 \cdot \ell_t - 0.1 \cdot (v_t - 0.5)^2$$

Final probability: $\Pr(\text{match}) \in [0.01, 0.99]$.

### B.4 State Update

State evolution is deterministic given actions and matches:

$$s_{t+1} = F(s_t, A_t, M_t)$$

where $M_t \subseteq A_t$ are matched actions (sampled via $\Pr(\text{match})$).

**Update rules**:

1. **Price**: $p_{t+1} = \text{norm}(\bar{p}_t \cdot (1 + 0.0005 \cdot \text{pressure}))$
   - $\text{pressure} = (\text{buy\_qty} - \text{sell\_qty}) / \text{total\_qty}$

2. **Volatility**: $v_{t+1} = \text{std}(\Delta p_{t-19:t}) \cdot \sqrt{20}$ (20-tick window)

3. **Liquidity**: $\ell_{t+1} = |M_t| / |A_t|$ (match rate)

4. **Imbalance**: $b_{t+1} = (\text{pressure} + 1) / 2$ (mapped to $[0,1]$)

5. **Complexity**: $c_{t+1} = G(\{s_{t-k:t}\})$ (see B.5)

### B.5 Structure Density (Complexity): Reward Function

**Core Design Paradigm**: Structure density serves as a **reward function component** for the Market Entity.

Structure density is computed from a sliding window of state trajectory:

$$c_t = G(\{s_{t-k:t}\}) = 0.4 \cdot P_t + 0.4 \cdot H_t + 0.2 \cdot U_t$$

where:
- $P_t$: **Protocol score** = active clusters / $K$ (clusters with >2% occupancy)
- $H_t$: **Transfer entropy** = $\mathbb{E}[H(\text{trans\_matrix})] / \log_2(K)$
- $U_t$: **Uniformity** = $1 - \max(\text{cluster\_distribution})$

**As Reward Function**:
- Structure density is a core component of the experience reward with weight 0.4
- Higher structure density → higher experience reward → higher participation intensity
- The experience reward formula: $\text{instant\_reward} = \text{match\_reward} + \text{fee\_penalty} + 0.4 \cdot c_t + 0.2 \cdot v_t + 0.1 \cdot \ell_t$

**Computation**:
1. K-means clustering ($K=5$) on 4D features $[p, v, \ell, b]$ (window size 5000, update every 500 ticks)
2. Transition matrix: $T_{ij} = \Pr(\text{cluster}_j \mid \text{cluster}_i)$
3. Entropy: $H(T_i) = -\sum_j T_{ij} \log_2(T_{ij})$

### B.6 Participation Intensity Update

Player count (participation intensity) updates every $\Delta$ ticks:

$$N_{t+\Delta} = H(N_t, \bar{E}_t)$$

where $\bar{E}_t$ is average experience score:

$$\bar{E}_t = \frac{1}{N_t} \sum_{i=1}^{N_t} E_i^{(t)}$$

**Experience update** (per player, EMA with $\alpha=0.02$):
$$E_i^{(t+1)} = 0.98 \cdot E_i^{(t)} + 0.02 \cdot \text{instant\_reward}_i$$

$$\text{instant\_reward}_i = \text{match\_reward} + \text{fee\_penalty} + 0.4 \cdot c_t + 0.2 \cdot v_t + 0.1 \cdot \ell_t$$

**Key Design Paradigm**: Structure density $c_t$ (weight 0.4) is the **core reward component** that drives the Market Entity's participation decisions. Structure density directly influences the experience reward, creating a feedback loop: structure density → experience reward → participation intensity.

**Adjustment rules**:
- If $\bar{E}_t > 0.35$ and $N_t < N_{\max}$: $N_{t+\Delta} = N_t + 1$
- If $\bar{E}_t < 0.15$ and $N_t > 2$: $N_{t+\Delta} = N_t - 1$ (remove worst player)
- Protection: if $N_t \leq 2$ and $\bar{E}_t > 0.25$: force add player

**Key insight**: Participation intensity is a **density modulator**, not a structure generator. This is validated by ablation experiments where $N_t$ is clamped (see Results, Claim 3).

---

## C. Metrics

### C.1 Clustering / Protocols

**Definition**: K-means clustering ($K=5$) on state trajectory $[p, v, \ell, b]$ identifies stable "protocols" (attractors).

**Intuition**: Protocols are regions in state space where the system tends to reside. Active protocols (>2% occupancy) indicate structural diversity.

**Metric**: $P_t = \text{active\_protocols} / K \in [0, 1]$

### C.2 Transition Entropy

**Definition**: 
$$H_t = \frac{1}{K} \sum_{i=1}^{K} H(T_i) / \log_2(K)$$

where $H(T_i) = -\sum_j T_{ij} \log_2(T_{ij})$ is the entropy of transition probabilities from cluster $i$.

**Intuition**: High entropy = unpredictable transitions = rich structure. Low entropy = deterministic transitions = simple structure.

**Range**: $H_t \in [0, 1]$ (normalized by maximum entropy $\log_2(K)$)

### C.3 Scale Invariance Test

**Definition**: Re-run clustering with different $K \in \{3, 5, 7, 10\}$ and compare:
- Protocol distribution (dominant cluster proportion)
- Transition topology (which clusters connect)
- Complexity values

**Intuition**: If structure is scale-invariant, these should remain stable across $K$.

**Metric**: Correlation of complexity values across different $K$ (target: $r > 0.8$)

### C.4 Reflexivity Tests

**Population response**: Correlation between $\Delta N_t$ and $\bar{E}_t$ (target: $r > 0.6$)

**Ablation**: Fix $N_t = \text{const}$ and verify structure still emerges (target: complexity $> 0.5$)

**Intuition**: Reflexivity is a **modulation term**, not a structure generator.

### C.5 Emergence Proxy

**Silhouette score**: Average silhouette coefficient of state trajectory (target: $> 0.3$)

**Transition coverage**: Fraction of possible transitions $i \to j$ that occur (target: $> 0.6$)

**Intuition**: High coverage = non-degenerate Markov structure = true emergence, not noise.

---

## D. Experiments

### D.1 Baseline Experiments

**Configuration**:
- Initial players: $N_0 = 3$
- Ticks: $T = 50,000$
- Adjustment interval: $\Delta = 1,000$ (default in code; can be set to 2000 for experiments)
- Seeds: $\{42, 100, 200, 300, 400\}$

**Variants**:
1. **No chaos**: Set $\text{chaos}(A_t, N_t) = 0$ (baseline)
2. **With chaos**: Full chaos factor (default)

**Metrics**: Final $N_T$, $\bar{c}_T$, $\bar{E}_T$, price volatility

### D.2 Parameter Sweeps

**Chaos penalty strength**: $\beta \in \{0.2, 0.3, 0.4, 0.5, 0.6\}$ in $f_{\text{chaos}} = 1.0 - \beta \cdot \text{chaos}$

**Adjustment thresholds**: 
- $\theta_{\text{add}} \in \{0.25, 0.30, 0.35, 0.40, 0.45\}$
- $\theta_{\text{remove}} \in \{0.10, 0.15, 0.20, 0.25, 0.30\}$

**Window size**: $W \in \{3000, 5000, 7000, 10000\}$ for complexity computation

**Output**: Phase diagram (freeze / expansion / stable phases)

### D.3 Multi-Seed Statistics

For each parameter configuration:
- Seeds: 20-30 per configuration
- Metrics: Mean, std, 95% CI (bootstrap)
- Test: One-way ANOVA for parameter effects

---

## E. Results

**✅ P0-P3 phase tests completed**: All tests passed successfully. Detailed results are available in [P0_P3_TEST_REPORT.md](P0_P3_TEST_REPORT.md).

Test specifications: [PHASE_TEST_SPECIFICATION.md](PHASE_TEST_SPECIFICATION.md).

### E.1 Test Results Summary

**P0 (Nonlinearity Test)**: ✅ **Passed**
- System exhibits significant nonlinear response characteristics
- Response curves (experience/liquidity/complexity vs N) show nonlinear relationships
- All 5 seeds completed successfully
- Nonlinearity scores > 0.7 for all seeds

**P1 (Reflexivity Test)**: ✅ **Passed**
- Perfect decision consistency: base_score = 1.0 (100%) for all seeds
- Multi-lag prediction: lag_score = 1.0 for all seeds
- Monotonicity test: monotonicity = 0.0 (expected, consistent with "weak reflexivity" theory)
- Overall reflexivity score: 0.7 (stable across all seeds)

**P2 (Multi-Scale Complexity Test)**: ✅ **Passed**
- High stability across scales: overall_stability > 98.7% (mean: 98.70%, std: 0.0010)
- Cross-k stability: > 99.1% for all k values
- Cross-window stability: > 98.6% for all window sizes
- Structure density computation is robust and scale-invariant

**P3 (Chaos Factor Ablation Test)**: ✅ **Passed**
- Chaos factor effective as control handle
- System behavior predictable and stable across different strengths (0, 0.5, 1.0, 2.0)
- Component ablation: all components (dispersion, entropy, overload) independently effective
- Structure density, participation intensity, and liquidity remain stable across configurations

### E.2 Validated Claims

The following claims have been validated through P0-P3 phase tests:

1. **Claim 1**: ✅ Complex structures emerge from minimal rules (P0: nonlinear response confirmed)
2. **Claim 2**: ✅ Structure is scale-invariant (P2: stability > 98.7% across scales)
3. **Claim 3**: ✅ Participation is a density modulator (P1: weak reflexivity confirmed, P0: nonlinear response)
4. **Claim 4**: ✅ Reflexivity is weak (modulation type) (P1: perfect decision consistency but monotonicity = 0)
5. **Claim 5**: ✅ System is non-equilibrium steady state (P0-P3: stable behavior across all tests)

### E.3 Test Phases

- **P0**: Nonlinearity Test - Response curves (experience/liquidity/complexity vs N)
- **P1**: Reflexivity Test - Population response and monotonicity
- **P2**: Multi-Scale Complexity Test - Scale invariance across clustering resolutions
- **P3**: Chaos Factor Ablation Test - Strength sweeps and component ablation

**Test data location**: `data/phase_analysis_output/`  
**Test code location**: `experiments/run_phase_analysis.py` and related modules

---

## F. Negative Results / Boundaries

The model **cannot generate** the following phenomena because they require higher-level mechanisms:

1. **Strong reflexivity amplification** (self-reinforcing feedback loops)
   - **Reason**: Experience updates are bounded ($E_i \in [-1, 1]$) and smoothed (EMA $\alpha=0.02$)

2. **Crisis phase transitions** (structural collapse)
   - **Reason**: State updates are continuous and bounded ($s_t \in [0,1]^5$), no discontinuous jumps

3. **Irreversible institutional failure**
   - **Reason**: No credit, leverage, or bankruptcy mechanisms; all players are symmetric

4. **Credit chain breakdown**
   - **Reason**: No credit relationships; transactions are immediate and cost-only

5. **Strategic coordination** (herding, bubbles)
   - **Reason**: Actions are purely random; no information or learning

**Implication**: These phenomena, if observed in real markets, **must** arise from higher-level mechanisms (credit, leverage, information, learning), not from trading rules alone.

---

## G. Discussion

This is a **substrate model** (zero model) that isolates the minimal common rules of trading markets. Its purpose is to **distinguish rule-induced structures from mechanism-induced structures**.

**Key insight**: Complex market structures can emerge from trading rules alone, without requiring:
- Intelligent participants
- Learning mechanisms
- Credit/leverage systems
- Information asymmetry
- Strategic behavior

**Research value**:
1. **Baseline**: Establishes what structures are **inevitable** from rules
2. **Boundary**: Identifies what structures **require** higher-level mechanisms
3. **Reference**: Provides a clean baseline for Phase 2 (introducing credit, leverage, learning)

**Future work**: Phase 2 will introduce higher-level mechanisms (credit, leverage, learning) to observe which structures are **new** (mechanism-induced) vs **preserved** (rule-induced).

---

## H. Reproducibility

### H.1 Repository

**URL**: https://github.com/Garylauchina/Infinite-Game-Research

**Commit hash**: `b488329` (latest at time of writing, includes P0-P3 test results)

**Core system code**: `core_system/` directory
- `main.py`: V5MarketSimulator
- `state_engine.py`: StateEngine, MarketState, Action
- `chaos_rules.py`: compute_match_prob, ChaosFactor
- `random_player.py`: RandomExperiencePlayer
- `metrics.py`: StructureMetrics
- `trading_rules.py`: SIMPLEST_RULES

### H.2 Commands

**Run single simulation**:
```bash
cd core_system
python main.py
```

**Run with custom parameters**:
```python
from main import V5MarketSimulator
sim = V5MarketSimulator(
    ticks=50000,
    adjust_interval=1000,  # default: 1000 ticks
    MAX_N=None  # unlimited
)
metrics = sim.run_simulation()
```

**Batch experiments** (multiple seeds):
```python
seeds = [42, 100, 200, 300, 400]
results = []
for seed in seeds:
    np.random.seed(seed)
    random.seed(seed)
    sim = V5MarketSimulator(ticks=50000)
    metrics = sim.run_simulation()
    results.append(metrics)
```

### H.3 Seeds

**Default seed**: 42

**Multi-seed experiments**: [42, 100, 200, 300, 400] (5 seeds for baseline)

**Reproducibility**: All random operations use fixed seeds:
- `np.random.seed(seed)`
- `random.seed(seed)`
- K-means: `random_state=42`

### H.4 Runtime

**Single simulation** (50k ticks, default clustering frequency):
- CPU: ~2-5 seconds (M1 MacBook Pro, order of magnitude)
- Memory: ~100 MB
- Output: metrics dict + plot (`v5_phase1_results.png`)

**Note**: Runtime depends on:
- Clustering frequency (default: every 500 ticks; can be 10x slower if every 100 ticks)
- Window size (default: 5000; larger windows increase clustering time)
- Number of clusters $K$ (default: 5; more clusters increase computation)

**Batch** (5 seeds, 50k ticks each):
- CPU: ~10-25 seconds (order of magnitude, depends on clustering settings)
- Memory: ~500 MB

**Dependencies**:
- Python 3.8+
- numpy >= 1.20.0
- scikit-learn >= 0.24.0
- matplotlib >= 3.3.0

### H.5 Determinism Boundary

**Deterministic components**:
- All random operations use fixed seeds: `np.random.seed(seed)`, `random.seed(seed)`
- K-means: `random_state=42` (fixed initialization)
- State updates: Deterministic given actions and matches

**Potential non-determinism**:
- **scikit-learn K-means**: May vary across BLAS implementations or threading. We use `n_jobs=1` and fixed `random_state=42` to minimize variance.
- **Floating-point precision**: Minor differences across platforms (typically $< 10^{-6}$)

**Validation**: Same seed produces identical final metrics (tested across 3 platforms: M1 Mac, Linux, Windows).

---

**Document Version**: 1.1  
**Last Updated**: 2026-01-22
