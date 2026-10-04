# Methods

## Koopman / Dynamic Mode Decomposition

### Koopman operator

For a discrete-time system `x_{t+1} = F(x_t)`, the Koopman operator `K` acts on observables `ψ: X → ℂ`:

```
(K ψ)(x) = ψ(F(x))
```

`K` is linear on a (possibly infinite-dimensional) function space, even when `F` is nonlinear. In practice, `K` is approximated by a `k × k` matrix over a finite dictionary `{ψ_1, …, ψ_k}`.

### Exact DMD (Tu et al. 2014)

Given snapshots `X = [x_1, …, x_{T-1}]`, `Y = [x_2, …, x_T]`, with `Y ≈ A X`:

1. SVD: `X = U Σ V*`
2. Truncate to rank `r` (energy- or rank-based)
3. Projected operator: `Ã = U* Y V Σ⁻¹` (`r × r`)
4. Eigendecomposition: `Ã W = W Λ`
5. Modes: `Φ = Y V Σ⁻¹ W`
6. Amplitudes: `b = Φ⁺ x_1`

`|λ_i|` measures persistence (= 1 stable, < 1 decaying, > 1 growing); `arg(λ_i)` measures oscillation frequency.

### Extended DMD (Williams, Kevrekidis, Rowley 2015)

EDMD applies DMD to lifted observables:

```
Ψ_X = [ψ(x_1), …, ψ(x_{T-1})]    (k × T-1)
Ψ_Y = [ψ(x_2), …, ψ(x_T)]
```

`K ≈ Ψ_Y Ψ_X⁺` via truncated SVD. Eigendecomposition of `K` yields Koopman eigenvalues and eigenfunctions in observable space.

### Observable dictionary

Defined on log returns `r_t = log(C_t / C_{t-1})`:

| Observable | Definition |
|---|---|
| `log_return` | `r_t` |
| `squared_return` | `r_t²` |
| `abs_return` | `\|r_t\|` |
| `momentum_{5,21,63}` | Rolling mean of returns |
| `vol_{5,21,63}` | Rolling std of returns |
| `drawdown` | `C_t / max(C_{1..t}) − 1` |
| `price_zscore_{21,63}` | `(log C_t − rolling_mean) / rolling_std`; detects price-level mean reversion |

Observable choice is the primary EDMD hyperparameter. Defaults are configurable via `ObservableConfig`.

## Spectrum-to-regime mapping

Implemented in `core/regime.py`, evaluated in order:

| Condition | Regime |
|---|---|
| Rule classifier: mean reversion with half-life < 15 bars | `MEAN_REVERTING` |
| `VR(5) < 0.70` and ann_vol < 40% | `MEAN_REVERTING` |
| ann_vol > 35% | `HIGH_VOL_CHOP` |
| `\|λ₁\| > 1.05` | `BREAKOUT` |
| `\|λ₁\| < 0.7` | `HIGH_VOL_CHOP` |
| `arg(λ₁) > 0.6 rad` | `MEAN_REVERTING` |
| Spectral gap < 0.005 | `UNKNOWN` |
| `0.95 ≤ \|λ₁\| ≤ 1.05`, low vol | `LOW_VOL_GRIND` |
| `0.95 ≤ \|λ₁\| ≤ 1.05`, moderate vol | `TRENDING_UP` / `TRENDING_DOWN` |
| `0.7 < \|λ₁\| < 0.95` | `MEAN_REVERTING` |

Direction for trending labels is set by annualised drift; eigenvalue magnitude is sign-agnostic.

## Hidden Markov Model

`hmmlearn.GaussianHMM` on a 2-D feature stream:

```
X_t = (log_return_t, log_rolling_vol_t)
```

Default `n_states = 3`. Fitted state means `(μ_r, μ_v)` are labelled as follows:

| Vol | Mean return | Label |
|---|---|---|
| > 1.3× median | negative | `TRENDING_DOWN` |
| > 1.3× median | positive | `BREAKOUT` |
| > 1.3× median | flat | `HIGH_VOL_CHOP` |
| < 0.7× median | near zero | `LOW_VOL_GRIND` |
| moderate | clear sign | `TRENDING_UP` / `TRENDING_DOWN` |
| otherwise | — | `MEAN_REVERTING` |

Confidence is the posterior `γ_T(i)` at the latest bar.

Covariance fallback: `full` → `diag` → `spherical` on singular covariance (common on very-low-vol series).

## Variance ratio (Lo-MacKinlay)

```
VR(k) = Var(r_t + r_{t+1} + … + r_{t+k-1}) / (k · Var(r_t))
```

- `VR = 1`: random walk
- `VR < 1`: mean reversion
- `VR > 1`: momentum

Parameters: `k = 5`; strong reversion threshold `VR(5) < 0.70` (~3σ below 1.0 on a 126-bar window).

## AR(1) half-life

For an OU process `dx = θ(μ − x) dt + σ dW`, the discrete form is `x_{t+1} = (1 − θΔt) x_t + θΔt μ + ε`. Regressing `log C_{t+1}` on `log C_t`:

```
φ = 1 − θΔt
half_life = ln(2) / (−ln(φ))    [bars]
```

- `φ ≈ 1`: random walk
- `φ < 0.95`: meaningful price-level reversion

Detects OU-type reversion, which resides in the price level and is not visible to returns-based tests such as VR. Requires `ann_vol ≥ 0.10`; below this, low `φ` values arise from noise.

## Transition-risk score

| Component | Definition |
|---|---|
| Eigenvalue drift | L2 distance between current top-k eigenvalues (sorted by magnitude) and the mean over `baseline_windows` (default 5) historical fits |
| Spectral-gap collapse | `max(0, baseline_gap_mean − current_gap)` |
| Vol acceleration | z-score of `(rolling_vol_21 − rolling_vol_63)` against its 60-bar history |

```
s_eig = sigmoid(eig_drift − 0.5, scale=3.0)
s_gap = sigmoid(gap_collapse × 10 − 1.0, scale=2.0)
s_vol = sigmoid(|vol_accel| − 1.0, scale=1.5)
risk  = clip(0.4·s_eig + 0.3·s_gap + 0.3·s_vol, 0, 1)
```

`crossed_threshold = True` when `risk > transition_threshold` (default 0.6).

## Excluded methods

- **Neural-network embeddings** — incompatible with eigenvalue-level explainability.
- **Supervised training on forward-vol or P&L labels** — the synthetic harness uses generative ground truth only.
- **Online weight optimisation** — ensemble weights are fixed in `DEFAULT_WEIGHTS` (`core/regime.py`); changes require a benchmark re-run.
