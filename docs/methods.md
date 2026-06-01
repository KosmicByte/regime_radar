# Methods

The mathematical content, in enough detail that you can audit (or replace) any piece.

## Koopman / Dynamic Mode Decomposition

### The Koopman view of nonlinear dynamics

For a discrete-time dynamical system `x_{t+1} = F(x_t)` on a state space, the Koopman operator `K` acts on *observables* `ψ: X → ℂ` (functions of the state) by

```
(K ψ)(x) = ψ(F(x))
```

`F` may be nonlinear, but `K` is **linear**, just on a (possibly infinite-dimensional) function space. This is why Koopman analysis is powerful: it linearises nonlinear dynamics, at the cost of moving to a richer space.

In practice we work with a finite dictionary of observables `{ψ_1, …, ψ_k}` and approximate `K` by a `k × k` matrix.

### Exact DMD (Tu et al. 2014)

Given snapshot pairs `X = [x_1, …, x_{T-1}]` and `Y = [x_2, …, x_T]` with the locally-linear assumption `Y ≈ A X`:

1. SVD: `X = U Σ V*`
2. Truncate to rank `r` (energy- or rank-based)
3. Projected operator: `Ã = U* Y V Σ⁻¹` (an `r × r` matrix)
4. Eigendecomposition: `Ã W = W Λ`
5. DMD modes in original space: `Φ = Y V Σ⁻¹ W`
6. Initial amplitudes: `b = Φ⁺ x_1`

The eigenvalues `λ_i` are the DMD eigenvalues. `|λ_i|` measures persistence (1 = stable mode, < 1 = decaying, > 1 = growing); `arg(λ_i)` measures oscillation frequency.

### Extended DMD (Williams, Kevrekidis, Rowley 2015)

Where DMD acts on raw states, EDMD acts on a dictionary of observables `ψ(x)`. Build snapshot matrices in the lifted space:

```
Ψ_X = [ψ(x_1), …, ψ(x_{T-1})]    (k × T-1)
Ψ_Y = [ψ(x_2), …, ψ(x_T)]
```

Then estimate the Koopman matrix `K ≈ Ψ_Y Ψ_X⁺` via truncated SVD. The eigendecomposition of `K` gives the Koopman eigenvalues and eigenfunctions (in observable space).

### Our observable dictionary

Default observables, built on the log return series `r_t = log(C_t / C_{t-1})`:

- `log_return` — `r_t`
- `squared_return` — `r_t²` (vol signal)
- `abs_return` — `|r_t|` (vol signal, less skewed)
- `momentum_{5,21,63}` — rolling mean of returns
- `vol_{5,21,63}` — rolling std of returns
- `drawdown` — `C_t / max(C_{1..t}) - 1`
- `price_zscore_{21,63}` — `(log C_t - rolling_mean) / rolling_std` (the critical observable for detecting price-level mean reversion)

The choice of observables is the most important hyperparameter in EDMD. Default values are exposed in `ObservableConfig` and can be customised.

## Regime mapping from the spectrum

Each Koopman eigenvalue carries interpretation. Our mapping (in `core/regime.py`):

| Spectrum signature | Regime |
|---|---|
| Rule classifier sees mean reversion + half-life < 15 bars | `MEAN_REVERTING` |
| `VR(5) < 0.70` and ann_vol < 40% | `MEAN_REVERTING` |
| ann_vol > 35% | `HIGH_VOL_CHOP` |
| `|λ₁| > 1.05` | `BREAKOUT` |
| `|λ₁| < 0.7` | `HIGH_VOL_CHOP` (fast decay = no persistence) |
| `arg(λ₁) > 0.6 rad` | `MEAN_REVERTING` (oscillation dominates) |
| spectral gap < 0.005 | `UNKNOWN` (regime in flux) |
| `0.95 ≤ |λ₁| ≤ 1.05` + low vol | `LOW_VOL_GRIND` |
| `0.95 ≤ |λ₁| ≤ 1.05` + moderate vol | `TRENDING_UP` (sign set by drift) |
| `0.7 < |λ₁| < 0.95` | `MEAN_REVERTING` (decay) |

Direction (up vs down) for trending labels comes from the annualised drift signal, not the spectrum (eigenvalue magnitude is sign-agnostic for real-valued price series).

## Hidden Markov Model

Standard `hmmlearn.GaussianHMM` on a 2-D feature stream:

```
X_t = (log_return_t, log_rolling_vol_t)
```

with `n_states` regimes (default 3). After EM fitting, each state has a mean `(μ_r, μ_v)` and we label it via the same taxonomy:

- High vol (above 1.3× median vol) + negative mean return → `TRENDING_DOWN`
- High vol + positive mean return → `BREAKOUT`
- High vol + flat → `HIGH_VOL_CHOP`
- Low vol (below 0.7× median) + near-zero return → `LOW_VOL_GRIND`
- Modest vol + clear sign → `TRENDING_{UP,DOWN}`
- Otherwise → `MEAN_REVERTING`

The HMM's posterior `γ_T(i)` at the latest bar serves as its confidence in the current label.

Robustness: we try `covariance_type='full'` first, falling back to `'diag'` then `'spherical'` on singular covariance — this happens routinely on very-low-vol synthetic series.

## Variance ratio (Lo-MacKinlay)

Given log returns `r_1, …, r_T`:

```
VR(k) = Var(r_t + r_{t+1} + … + r_{t+k-1}) / (k · Var(r_t))
```

For a random walk, `VR(k) = 1` for all `k`. Departures from 1 signal:

- `VR < 1` — mean reversion (variance grows slower than linearly with horizon)
- `VR > 1` — momentum (variance grows faster than linearly)

We use `k = 5` and require `VR(5) < 0.70` for a "strong reversion" call (about 3σ below 1.0 on a 126-bar window).

## AR(1) half-life of mean reversion

For an OU process `dx = θ(μ - x) dt + σ dW`, the discrete representation is `x_{t+1} = (1 - θΔt) x_t + θΔt μ + ε`. So a linear regression of `log C_{t+1}` on `log C_t` yields

```
slope φ = 1 - θΔt
half_life = ln(2) / (-ln(φ))    [bars]
```

`φ ≈ 1` means random walk (no reversion); `φ < 0.95` means meaningful reversion at the price level. Half-life is the expected time for a deviation from mean to shrink by 50%.

This is the **critical** test that catches OU-style regimes. Returns-based tests like VR cannot see this because OU reversion lives in the price level, not in return autocorrelation.

We additionally require `ann_vol ≥ 0.10` for the half-life signal to fire, because in very-low-vol regimes the AR(1) `φ` can appear low purely from noise without genuine reversion. True OU reversion produces meaningful price deviations from the mean.

## Transition-risk score

Three components combined via sigmoid + weighted average:

### Eigenvalue drift

We fit EDMD on `baseline_windows` (default 5) historical windows of length `window`, sorted by magnitude, and take the L2 distance between the current top-k eigenvalues and the baseline mean.

### Spectral-gap collapse

`max(0, baseline_gap_mean - current_gap)`. A collapsing dominant eigenvalue means the operator is losing structure — regime ahead.

### Vol acceleration

z-score of `(rolling_vol_21 - rolling_vol_63)` against its 60-bar history. Captures vol regime shifts at the shortest horizon we trust.

### Combination

```
s_eig = sigmoid(eig_drift - 0.5, scale=3.0)
s_gap = sigmoid(gap_collapse * 10 - 1.0, scale=2.0)
s_vol = sigmoid(|vol_accel| - 1.0, scale=1.5)
risk = clip(0.4·s_eig + 0.3·s_gap + 0.3·s_vol, 0, 1)
```

Default `transition_threshold = 0.6`; risk above this is flagged via `crossed_threshold=True`.

## What we deliberately did NOT use

- **Neural-network embeddings.** They would erase explainability — the whole point of this package is that you can read off the eigenvalues.
- **Forward-vol or PnL labels for training.** We never train *on* a target; the synthetic harness uses *generative* ground truth so the labels are causally pre-defined.
- **Online optimisation of weights.** The ensemble weights are fixed and documented. If you want to tune them, edit `DEFAULT_WEIGHTS` in `core/regime.py` and re-run the benchmark.
