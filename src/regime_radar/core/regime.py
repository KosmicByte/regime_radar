"""Regime classification from the Koopman spectrum + ensemble of three independent voters.

We combine three orthogonal views and aggregate via weighted soft voting:
    1. EDMD spectrum → maps eigenvalue structure to a label
    2. Gaussian HMM → independent probabilistic state
    3. Rule-based vol+trend → transparent baseline

Each voter contributes a label and a confidence; the final label is the argmax of the
weighted probability distribution, and `reasons` is populated so the user can audit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from regime_radar.config import get_settings
from regime_radar.core.contract import PointInTimeFrame
from regime_radar.core.edmd import EDMDResult, fit_edmd
from regime_radar.core.hmm import fit_hmm
from regime_radar.core.observables import ObservableConfig
from regime_radar.core.rules import classify_rule_based
from regime_radar.models import KoopmanModes, RegimeLabel, RegimeReason, RegimeResult
from regime_radar.provenance import InferenceRecord, write_record
from regime_radar.version import MODEL_VERSION, code_version

# Voter weights — tunable. EDMD slightly heavier when its spectral gap is wide.
DEFAULT_WEIGHTS = {"edmd": 0.40, "hmm": 0.30, "rule": 0.30}


def _label_from_spectrum(
    edmd: EDMDResult,
    rule_vol_pct: float,
    rule_ann_vol: float,
    rule_vr5: float = 1.0,
    rule_half_life: float = float("inf"),
    rule_label_hint: RegimeLabel | None = None,
) -> tuple[RegimeLabel, str]:
    """Map a Koopman spectrum to a regime label using interpretable thresholds.

    Heuristics (each backed by a one-liner so the reason is auditable):
        - rule classifier saw mean reversion + half-life < 15 → endorse it
        - VR(5) < 0.70 → mean-reverting (returns-space test)
        - |λ_1| ≈ 1 with small imag → trending (direction set by drift voter)
        - 0.7 < |λ_1| < 0.95 → mean-reverting (decay)
        - |λ_1| > 1.05 → breakout / unstable
        - Complex eigenvalues with sizeable imag part → cyclic / high-vol chop
        - Spectral gap collapse → unknown (regime in flux)
    """
    if len(edmd.eigenvalues) == 0:
        return RegimeLabel.UNKNOWN, "Empty spectrum."

    lam1 = edmd.eigenvalues[0]
    mag1 = float(np.abs(lam1))
    arg1 = float(np.abs(np.angle(lam1)))  # 0 = real positive
    gap = edmd.spectral_gap

    # HIGHEST PRIORITY: short AR(1) half-life + rule classifier endorses mean reversion.
    # The rule classifier has already gated on absolute drift and vol floor (>=10%),
    # so we trust its signal here.
    if (
        rule_half_life < 15.0
        and 0.10 <= rule_ann_vol < 0.40
        and rule_label_hint == RegimeLabel.MEAN_REVERTING
    ):
        return RegimeLabel.MEAN_REVERTING, (
            f"AR(1) half-life {rule_half_life:.1f} bars — strong price-level reversion."
        )

    # Strong variance-ratio evidence of mean reversion (return-space)
    if rule_vr5 < 0.70 and rule_ann_vol < 0.40:
        return RegimeLabel.MEAN_REVERTING, (
            f"VR(5)={rule_vr5:.2f}<0.70 — mean-reversion signature outweighs spectrum."
        )

    # High-vol regime overrides spectrum interpretation only when absolute vol is high.
    # Percentile-based detection alone causes false positives in moderate-vol trending
    # regimes (recent vol slightly elevated vs older history, but not actually high).
    if rule_ann_vol > 0.35:
        return RegimeLabel.HIGH_VOL_CHOP, (
            f"Realized vol high (ann {rule_ann_vol:.0%}) — vol regime dominates."
        )

    if mag1 > 1.05:
        return RegimeLabel.BREAKOUT, f"|λ₁|={mag1:.3f}>1.05 — operator is expanding."
    if mag1 < 0.7:
        return RegimeLabel.HIGH_VOL_CHOP, f"|λ₁|={mag1:.3f}<0.7 — fast decay, no persistence."
    if arg1 > 0.6:  # ~34° → meaningful oscillatory component
        return RegimeLabel.MEAN_REVERTING, (
            f"Complex λ₁ (arg={arg1:.2f} rad) — oscillation dominates over drift."
        )
    if gap < 0.005:
        return RegimeLabel.UNKNOWN, (
            f"Spectral gap {gap:.4f} too small — no clearly dominant mode (regime in flux)."
        )
    if 0.95 <= mag1 <= 1.05:
        # Persistent dominant mode. The label depends on the vol context:
        # - low vol + persistent mode → calm grind, not "trending"
        # - moderate vol + persistent mode → genuine trend (direction set by drift voter)
        if rule_ann_vol < 0.10:
            return RegimeLabel.LOW_VOL_GRIND, (
                f"|λ₁|={mag1:.3f}≈1 with low vol (ann {rule_ann_vol:.0%}) — calm random walk."
            )
        return RegimeLabel.TRENDING_UP, (
            f"|λ₁|={mag1:.3f}≈1 with real-dominant arg={arg1:.2f} — persistent mode."
        )
    return RegimeLabel.MEAN_REVERTING, f"|λ₁|={mag1:.3f} in (0.7, 0.95): decaying back to mean."


def _soft_vote(
    edmd_label: RegimeLabel,
    edmd_conf: float,
    hmm_label: RegimeLabel,
    hmm_conf: float,
    rule_label: RegimeLabel,
    rule_conf: float,
    weights: dict[str, float] | None = None,
) -> dict[RegimeLabel, float]:
    """Weighted soft vote across the three methods.

    Each voter contributes weight × confidence to its label. Remaining mass is spread
    uniformly across other labels (mild smoothing so we never zero out alternatives).
    """
    weights = weights or DEFAULT_WEIGHTS
    probs: dict[RegimeLabel, float] = {label: 0.0 for label in RegimeLabel}

    for label, conf, w in (
        (edmd_label, edmd_conf, weights["edmd"]),
        (hmm_label, hmm_conf, weights["hmm"]),
        (rule_label, rule_conf, weights["rule"]),
    ):
        probs[label] += w * conf
        spread = w * (1.0 - conf) / max(len(probs) - 1, 1)
        for other in probs:
            if other != label:
                probs[other] += spread

    total = sum(probs.values())
    if total > 0:
        probs = {k: v / total for k, v in probs.items()}
    return probs


def detect_regime(
    close: np.ndarray | None = None,
    timestamps: pd.Series | None = None,
    symbol: str = "?",
    interval: str = "1d",
    observable_config: ObservableConfig | None = None,
    edmd_rank: int = 10,
    hmm_n_states: int = 3,
    weights: dict[str, float] | None = None,
    *,
    frame: PointInTimeFrame | None = None,
) -> RegimeResult:
    """Run the full ensemble and produce a single auditable RegimeResult.

    Detection consumes a point-in-time contract. Either pass a ``frame`` directly, or pass
    raw ``close`` (+ optional ``timestamps``) and a frame is built internally — both paths
    get the same no-look-ahead truncation guarantee, so neither can ever see a bar dated
    after the evaluation instant.

    Args:
        close: 1-D price series, length T. Required unless ``frame`` is given.
        timestamps: optional matching timestamps; used to set ``as_of`` to the last bar.
        symbol, interval: metadata (ignored when ``frame`` carries its own).
        observable_config: dictionary configuration for EDMD.
        edmd_rank: SVD truncation rank for EDMD.
        hmm_n_states: HMM state count.
        weights: override default voter weights.
        frame: a pre-built :class:`PointInTimeFrame`. Preferred at call sites that already
            hold one (e.g. walk-forward evaluation), as it makes the no-leak guarantee
            explicit and skips re-truncation.
    """
    # 1) Resolve the point-in-time contract.
    if frame is None:
        if close is None:
            raise ValueError("detect_regime requires either `frame=` or `close=`.")
        frame = PointInTimeFrame.from_arrays(
            np.asarray(close, dtype=float),
            timestamps,
            symbol=symbol,
            interval=interval,
        )
    close = frame.close
    symbol = frame.symbol
    interval = frame.interval

    # 2) EDMD voter
    edmd = fit_edmd(close, config=observable_config, rank=edmd_rank)
    rule = classify_rule_based(close)

    # Use rule's vol percentile context inside the spectrum mapper.
    from regime_radar.core.observables import log_returns, rolling_std

    r = log_returns(close)
    hist_vol = rolling_std(r, 252)
    hist_valid = hist_vol[~np.isnan(hist_vol)]
    cur_vol = float(np.std(r[-63:]))
    vol_pct = float(np.mean(hist_valid < cur_vol)) if len(hist_valid) > 0 else 0.5

    edmd_label, edmd_note = _label_from_spectrum(
        edmd,
        vol_pct,
        rule.annualised_vol,
        rule.variance_ratio_5,
        rule.half_life_days,
        rule.label,
    )

    # If spectrum said "trending" but the rule says drift is negative, flip direction.
    if edmd_label == RegimeLabel.TRENDING_UP and rule.annualised_drift < -0.02:
        edmd_label = RegimeLabel.TRENDING_DOWN
        edmd_note += " Direction overridden by negative annual drift."

    edmd_conf = float(min(1.0, edmd.spectral_gap * 20 + 0.4))  # 0.4..1.0 over gap 0..0.03

    # 2) HMM voter
    hmm = fit_hmm(close, n_states=hmm_n_states)
    hmm_label = hmm.current_label
    hmm_conf = hmm.current_confidence

    # Cap raw HMM confidence at 0.85. Posteriors of 1.00 are an artefact of Gaussian
    # emission likelihoods, not real epistemic certainty — leaving a doubt floor of
    # 0.15 prevents one over-confident voter from drowning out the other two.
    hmm_conf = min(hmm_conf, 0.85)

    # HMM tends to overconfidently call directional labels in low-info regimes.
    # Dampening ladder (most severe first):
    is_directional = hmm_label in (
        RegimeLabel.TRENDING_UP,
        RegimeLabel.TRENDING_DOWN,
        RegimeLabel.BREAKOUT,
    )
    rule_says_reverting = rule.label == RegimeLabel.MEAN_REVERTING

    # CRITICAL: HMM points directionally but BOTH other voters disagree on direction.
    # Two-against-one across orthogonal methods is the strongest possible refutation.
    edmd_disagrees_directionally = (
        edmd_label in (RegimeLabel.TRENDING_UP, RegimeLabel.TRENDING_DOWN, RegimeLabel.BREAKOUT)
        and edmd_label != hmm_label
    )
    rule_disagrees_directionally = (
        rule.label in (RegimeLabel.TRENDING_UP, RegimeLabel.TRENDING_DOWN, RegimeLabel.BREAKOUT)
        and rule.label != hmm_label
    )
    if is_directional and edmd_disagrees_directionally and rule_disagrees_directionally:
        hmm_conf *= 0.25  # two-against-one — HMM is almost certainly wrong
    elif is_directional and rule_says_reverting and rule.half_life_days < 15.0:
        hmm_conf *= 0.3  # strong price-level reversion evidence
    elif is_directional and rule_says_reverting and rule.variance_ratio_5 < 0.70:
        hmm_conf *= 0.4  # strong return-space reversion evidence
    elif is_directional and abs(rule.trend_strength) < 0.05:
        hmm_conf *= 0.6  # weak Sharpe makes HMM's confidence unreliable

    # 3) Rule voter
    rule_label = rule.label
    rule_conf = rule.confidence

    # Ensemble
    probs = _soft_vote(edmd_label, edmd_conf, hmm_label, hmm_conf, rule_label, rule_conf, weights)
    final_label = max(probs, key=probs.get)
    final_conf = probs[final_label]

    # Reasons (ordered by contribution)
    reasons = [
        RegimeReason(
            factor="edmd_spectrum",
            value=f"|λ₁|={abs(edmd.eigenvalues[0]):.3f}",
            contribution=DEFAULT_WEIGHTS["edmd"] * edmd_conf,
            note=edmd_note,
        ),
        RegimeReason(
            factor="hmm_state",
            value=hmm_label.value,
            contribution=DEFAULT_WEIGHTS["hmm"] * hmm_conf,
            note=f"HMM Viterbi state with posterior {hmm_conf:.2f}.",
        ),
        RegimeReason(
            factor="rule_vol_trend",
            value=rule_label.value,
            contribution=DEFAULT_WEIGHTS["rule"] * rule_conf,
            note=rule.note,
        ),
    ]
    reasons.sort(key=lambda r: r.contribution, reverse=True)

    # as_of and the input hash come from the point-in-time contract — single source of truth.
    as_of = frame.as_of
    input_hash = frame.input_hash()

    modes = KoopmanModes(
        eigenvalues=[complex(z) for z in edmd.eigenvalues],
        growth_rates=edmd.growth_rates.tolist(),
        frequencies=edmd.frequencies.tolist(),
        mode_energies=edmd.mode_energies.tolist(),
        rank=edmd.rank,
        spectral_gap=edmd.spectral_gap,
    )

    result = RegimeResult(
        symbol=symbol,
        interval=interval,
        as_of=as_of,
        label=final_label,
        confidence=final_conf,
        probabilities={k: float(v) for k, v in probs.items()},
        reasons=reasons,
        modes=modes,
        method_votes={"edmd": edmd_label, "hmm": hmm_label, "rule": rule_label},
        realized_vol=rule.annualised_vol,
        trend_strength=rule.trend_strength,
        model_version=MODEL_VERSION,
        code_version=code_version(),
        input_hash=input_hash,
    )

    # Provenance: append an immutable record when enabled. Best-effort — never let logging
    # break a detection.
    settings = get_settings()
    if settings.provenance_enabled:
        try:
            write_record(
                InferenceRecord(
                    as_of=str(result.as_of),
                    symbol=result.symbol,
                    interval=result.interval,
                    model_version=result.model_version,
                    code_version=result.code_version,
                    input_hash=result.input_hash,
                    label=result.label.value,
                    confidence=result.confidence,
                    probabilities={k.value: float(v) for k, v in result.probabilities.items()},
                ),
                settings.provenance_dir / f"{result.symbol.replace('/', '_')}.jsonl",
            )
        except Exception:  # noqa: BLE001 — provenance must never break detection
            pass

    return result
