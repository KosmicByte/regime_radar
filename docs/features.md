# Roadmap & competitive features

How RegimeRadar compares to the proprietary regime-detection systems used by professional desks, and what's planned to close the gap.

## The honest framing

Big players (Citadel, Two Sigma, Renaissance, bulge-bracket research desks) have advantages that no open-source library can erase:

1. **Data**: tick-level, multi-venue, alternative-data overlays (satellite, credit-card flows, news APIs with full historical access).
2. **Execution**: tight feedback loops between regime detection and execution mean their definition of "regime" is *whatever P&L responds to*, not an abstract mathematical category.
3. **Compute**: nightly retraining on billions of bars across asset classes.
4. **People**: full-time research teams refining a single regime-detection system over years.

These are structural moats. RegimeRadar is not going to beat them on raw accuracy on liquid US equities or major currency pairs.

What RegimeRadar *can* compete on:

- **Transparency** — every label is explained by its eigenvalues and statistics. You can audit a call. With proprietary systems, you cannot.
- **Honesty** — the evaluation harness ships with the code. The 70% claim is reproducible. Proprietary systems quote whatever accuracy supports the marketing.
- **Indian markets focus** — bhavcopy schema, NSE conventions, India VIX integration. The big players' systems are tuned for US/EU markets and may underperform on Nifty / Bank Nifty.
- **Specialisation** — for *your specific* strategy (Wheel, mean-reversion pairs, swing trading), a strategy-conditioned regime detector that you can tune will beat a generic high-end system.

These are not "as good as" features — they are *different* features. Compete where the competition can't follow.

## Phase 1: bigger accuracy wins

These are features that should push the headline accuracy materially higher.

### Hankel-DMD / HAVOK (time-delay embedding)

The current EDMD uses observables at time `t` only. Brunton's HAVOK extends this by stacking `H` time-delayed copies of each observable, creating a Hankel matrix. Mathematically equivalent to a higher-dimensional Koopman approximation; practically, it captures **non-Markovian** dynamics that single-bar observables miss.

Expected impact: 5-10 percentage points on accuracy, especially on regime-switching and oscillatory regimes.

Implementation cost: medium. The Hankel construction is straightforward; the harder part is choosing the delay count and tuning rank.

### Cross-asset regime fingerprints

A regime in Nifty looks different depending on what Bank Nifty, India VIX, and USDINR are doing simultaneously. Stack observables across symbols into one joint EDMD fit; the cross-asset correlations become part of the dictionary.

Why this matters: "high-vol chop in Nifty + high-vol chop in Bank Nifty + spiking VIX + INR weakness" is a *different* regime from "high-vol chop in Nifty alone" — the former is a macro stress event, the latter is index-specific.

Implementation cost: medium. The mechanics are easy; the harder part is choosing the cross-asset weighting and handling asynchronous bars (US data + India data have different sessions).

### Online / streaming EDMD

Re-fitting from scratch on every new bar is wasteful. Brunton & Kutz' streaming DMD updates `U`, `Σ`, `V` incrementally, so the detector can produce a fresh regime label in O(k²) instead of O(window · k²) per bar.

Why this matters: enables intraday detection (5-min bars) at acceptable latency and lets the detector run inside a streaming pipeline (Kafka → detector → trading system).

Implementation cost: low-medium. The streaming-SVD code is well-documented in the academic literature.

### Strategy-conditioned regimes

Instead of defining regimes by abstract math, define them by **where your strategy makes or loses money**. Concretely:

1. Backtest the strategy (e.g., Wheel options selling, or pair trading).
2. Bucket bars by realized P&L outcome (top quartile = "favourable regime", bottom quartile = "adverse regime").
3. Train the ensemble to detect those buckets.

This is the killer feature. Generic regime detectors classify markets into categories that *might* matter for strategy P&L. A strategy-conditioned detector classifies them into categories that **demonstrably** matter.

Implementation cost: medium-high. Requires a backtesting layer, which we deliberately kept out of this package. Best implemented as a separate `regime-radar-strategist` package consuming RegimeRadar and a backtest.

## Phase 2: explainability & UX

These don't move accuracy but make the package materially more useful day-to-day.

### Regime persistence model

For each detected regime, fit a per-regime duration distribution from history (or synthetic ground truth). Then the user sees "this trend has lasted 38 bars; typical trend in this market lasts 22 bars (P90 = 51)" — a calibrated sense of *how unusual the current spell is*.

Coupled with a hazard rate, this becomes a *time-to-transition* estimate.

### Transition probability heatmap

The empirical regime → regime transition matrix over history, displayed as a heatmap. "When we're in HIGH_VOL_CHOP, we transition to TRENDING_DOWN 35% of the time, to MEAN_REVERTING 28%, ..."

This is a one-call insight: given the current regime, what's the prior distribution over next regimes?

### News / sentiment overlay (4th voter)

Plug in a sentiment regime as a fourth ensemble voter, sourced from a VADER-scored news feed (port the Commodity Sentinel pipeline). Particularly useful for distinguishing "fundamental regime shift" from "technical noise".

Implementation: integrate with Cockpit's sentiment module when that's built; degrade gracefully if not available.

### Strategy hooks

A simple Python protocol that lets a strategy ask "what regime should I be playing right now, given my universe and parameters?" — and get a `RegimeResult` plus a recommended position sizing tweak. Soft, not prescriptive.

## Phase 3: harder math, narrower wins

Things that are research-grade and may or may not pay off.

### Resolvent DMD / mpEDMD (measure-preserving EDMD)

Recent (post-2022) extensions of EDMD that handle non-normal operators and provide convergence guarantees. Likely small accuracy improvements, but cleaner theory. Worth implementing for the academic story.

### Dictionary learning for observables

Instead of hand-picking observables, learn them via a small autoencoder. The danger: we lose explainability — the whole reason the package exists. The compromise: learn observables but constrain them to be combinations of interpretable primitives.

### Continuous-time Koopman

Direct estimation of the generator `L` (the Lie derivative) instead of the discrete-time `K`. Useful for irregular sampling and high-frequency data. Mostly relevant if we extend to tick-level analysis.

## What we will NOT do

- **Build a backtester here.** That's a separate package's job (Cockpit or a new `backtester` repo). Conflating detection with backtesting muddies the regime definitions.
- **Hard-code trading signals.** The system outputs *labels and risk*, not "buy/sell". Strategy logic lives downstream.
- **Add deep-learning regime classifiers as the primary path.** The whole identity of RegimeRadar is its explainability. A neural-net main path would destroy that.

## Prioritisation suggestion

If you have a finite amount of time, the highest-ROI ordering is:

1. **Streaming EDMD** — unlocks intraday use, low implementation cost.
2. **Cross-asset fingerprints** — likely 5-10 pp accuracy on Indian indices.
3. **Hankel/HAVOK** — likely another 5-10 pp, especially on regime switching.
4. **Strategy-conditioned regimes** — the genuine competitive differentiator. Build only after a target strategy exists in your stack.
5. **Persistence + transition matrix** — these are basically free given the existing infrastructure; do them whenever you have a Saturday afternoon.

Everything else is iceing.
