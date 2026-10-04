# Roadmap

## Positioning

RegimeRadar does not target parity with proprietary institutional systems on raw accuracy in liquid US/EU markets. Its differentiators:

- **Transparency** — every label is traceable to eigenvalues and statistics.
- **Reproducible evaluation** — the benchmark harness ships with the code.
- **Indian market focus** — bhavcopy schema, NSE conventions, India VIX integration.
- **Strategy specialisation** — tunable for specific strategies (Wheel, mean-reversion pairs, swing trading).

## Priority order

1. **Streaming EDMD** — enables intraday use; low cost.
2. **Cross-asset fingerprints** — expected +5–10 pp on Indian indices.
3. **Hankel-DMD / HAVOK** — expected +5–10 pp, primarily on regime switching.
4. **Strategy-conditioned regimes** — requires a target strategy in the stack.
5. **Persistence model and transition matrix** — low cost on existing infrastructure.

## Phase 1: Accuracy

### Hankel-DMD / HAVOK (time-delay embedding)

Stacks `H` time-delayed copies of each observable into a Hankel matrix, extending EDMD to non-Markovian dynamics.

- **Expected impact**: +5–10 pp, primarily on regime-switching and oscillatory regimes.
- **Cost**: medium. Main effort is selecting delay count and rank.

### Cross-asset regime fingerprints

Joint EDMD fit over observables from multiple symbols (e.g. Nifty, Bank Nifty, India VIX, USDINR). Distinguishes macro stress events from index-specific regimes.

- **Cost**: medium. Main effort is cross-asset weighting and alignment of asynchronous sessions.

### Streaming EDMD

Incremental update of `U`, `Σ`, `V` (Brunton & Kutz) reduces per-bar cost from O(window · k²) to O(k²). Enables intraday (5-min) detection and integration with streaming pipelines.

- **Cost**: low–medium.

### Strategy-conditioned regimes

Regimes defined by strategy P&L rather than abstract dynamics:

1. Backtest the strategy.
2. Bucket bars by realised P&L (top quartile = favourable, bottom quartile = adverse).
3. Train the ensemble to detect these buckets.

- **Cost**: medium–high. Planned as a separate `regime-radar-strategist` package.

## Phase 2: Explainability and UX

### Regime persistence model

Per-regime duration distribution fitted from history or synthetic ground truth. Reports current spell length against typical duration (median, P90). Combined with a hazard rate, yields a time-to-transition estimate.

### Transition probability heatmap

Empirical regime-to-regime transition matrix over history, giving a prior distribution over the next regime.

### News sentiment voter

Fourth ensemble voter based on a VADER-scored news feed (ported from the Commodity Sentinel pipeline). Integrates with Cockpit's sentiment module when available; degrades gracefully otherwise.

### Strategy hooks

Python protocol returning a `RegimeResult` and an advisory position-sizing adjustment for a given universe and parameter set.

## Phase 3: Research

### Resolvent DMD / mpEDMD

Measure-preserving EDMD variants with convergence guarantees and support for non-normal operators. Expected: modest accuracy gains, stronger theoretical basis.

### Constrained dictionary learning

Learned observables restricted to combinations of interpretable primitives, preserving explainability.

### Continuous-time Koopman

Direct estimation of the generator `L` instead of the discrete-time `K`. Relevant for irregular sampling and tick-level data.

## Out of scope

- **Strategy backtesting** — belongs in Cockpit or a dedicated backtester package. `regime backtest` is limited to validating regime timing.
- **Trading signals** — output is labels and risk only; strategy logic is downstream.
- **Deep-learning primary classifier** — incompatible with the explainability requirement.
