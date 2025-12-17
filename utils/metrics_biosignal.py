"""
생체신호(Biosignal) 평가용 고급 메트릭

이 모듈은 rPPG/PPG 뿐 아니라 ECG, RESP 등 시간 신호 전반에 적용 가능한
품질/생리학적 지표를 제공합니다.

포함 지표:
  - BWMD: Bi‑Wasserstein‑like Morphology Distance (DTW + peak timing + area)
  - SRE: Signal Reconstruction Error (RMSE + SNR 기반 가중)
  - WCR: Weighted Correlation (CCC + Spearman 조합)
  - EDD: Error Distribution Distance (JSD 기반)
  - Composite v2: 위 지표의 백분위 정규화 가중 합산 점수
  - HRV(peak 기반): mean_rr, median_rr, sdnn, rmssd, pnn50, mean_hr
"""
from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.signal import find_peaks
from scipy.stats import spearmanr, entropy


# ---------------------------------------------------------------------------
# Helper dataclasses
# ---------------------------------------------------------------------------


@dataclass
class BiosignalMetrics:
    """단일 신호(또는 subject) 수준의 고급 메트릭 결과."""

    bwmd: float
    sre: float
    wcr: float
    edd: float

    rmse: float
    snr_db: float
    cri: float
    si: float

    hrv_label: Dict[str, float]
    hrv_pred: Dict[str, float]
    hrv_diff: Dict[str, float]


@dataclass
class CompositeResult:
    """Composite v2 점수 결과."""

    score: float  # 0..100
    breakdown: Dict[str, float]


# ---------------------------------------------------------------------------
# Core distance/correlation utilities
# ---------------------------------------------------------------------------


def dtw_distance(a: np.ndarray, b: np.ndarray) -> float:
    """
    L1 cost 기반 DTW 거리(정규화).

    Args:
        a, b: 1D 신호 (정규화된 값 권장)

    Returns:
        정규화된 DTW 거리
    """
    a = np.ravel(a).astype(float)
    b = np.ravel(b).astype(float)
    na, nb = len(a), len(b)
    if na == 0 or nb == 0:
        return 0.0

    # Python 2중 루프이지만, 로컬 변수/슬라이스 캐싱으로 오버헤드 최소화
    D = np.full((na + 1, nb + 1), np.inf, dtype=float)
    D[0, 0] = 0.0
    b_local = b  # local alias
    for i in range(1, na + 1):
        ai = a[i - 1]
        Di = D[i]
        Dim1 = D[i - 1]
        for j in range(1, nb + 1):
            cost = abs(ai - b_local[j - 1])
            # min(Dim1[j], Di[j-1], Dim1[j-1])
            m1 = Dim1[j]
            m2 = Di[j - 1]
            m3 = Dim1[j - 1]
            if m2 < m1:
                m1 = m2
            if m3 < m1:
                m1 = m3
            Di[j] = cost + m1

    dist = float(D[na, nb])
    return dist / max(na + nb, 1)


def find_peak_times(
    signal: np.ndarray,
    fs: float,
    min_distance_sec: float = 0.4,
) -> np.ndarray:
    """
    신호에서 peak time(초)을 검출.

    Args:
        signal: 1D 신호
        fs: 샘플링 주파수(Hz)
        min_distance_sec: peak 간 최소 거리(초)

    Returns:
        peak time 배열(초)
    """
    sig = np.ravel(signal).astype(float)
    if sig.size == 0:
        return np.array([])
    min_dist = max(1, int(min_distance_sec * fs))
    peaks, _ = find_peaks(sig, distance=min_dist)
    return peaks.astype(float) / float(fs)


def concordance_ccc(x: np.ndarray, y: np.ndarray) -> float:
    """
    Concordance Correlation Coefficient(CCC).
    """
    x = np.asarray(x, dtype=float).ravel()
    y = np.asarray(y, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return 0.0

    mx, my = float(np.mean(x)), float(np.mean(y))
    sx2 = float(np.var(x, ddof=1)) if x.size > 1 else 0.0
    sy2 = float(np.var(y, ddof=1)) if y.size > 1 else 0.0
    cov = float(np.cov(x, y, ddof=1)[0, 1]) if x.size > 1 and y.size > 1 else 0.0

    denom = sx2 + sy2 + (mx - my) ** 2
    if denom == 0.0:
        return 0.0
    ccc = 2.0 * cov / denom
    return float(np.clip(ccc, -1.0, 1.0))


# ---------------------------------------------------------------------------
# Advanced metrics
# ---------------------------------------------------------------------------


def compute_bwmd_for_chunk(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float,
    lambda_p: float = 1.0,
    lambda_a: float = 0.5,
    min_distance_sec: float = 0.4,
) -> float:
    """
    BWMD(Bi‑Wasserstein‑like Morphology Distance) 한 청크 계산.

    Args:
        y_true, y_pred: 1D 신호 청크
        fs: 샘플링 주파수(Hz)
        lambda_p: peak timing 항 가중치
        lambda_a: area 항 가중치
        min_distance_sec: peak 검출 최소 간격(초)
    """
    y = np.ravel(y_true).astype(float)
    yhat = np.ravel(y_pred).astype(float)
    if y.size == 0 or yhat.size == 0:
        return 0.0

    max_y = float(np.max(np.abs(y)) + 1e-8)
    max_yhat = float(np.max(np.abs(yhat)) + 1e-8)
    a = y / max_y
    b = yhat / max_yhat

    dtw_norm = dtw_distance(a, b)

    pt_y = find_peak_times(y, fs, min_distance_sec=min_distance_sec)
    pt_yhat = find_peak_times(yhat, fs, min_distance_sec=min_distance_sec)

    if pt_y.size >= 2 and pt_yhat.size >= 2:
        k = min(pt_y.size, pt_yhat.size)
        diffs = np.abs(pt_y[:k] - pt_yhat[:k])
        peak_time_diff_mean = float(np.mean(diffs))
        rr = np.diff(pt_y)
        rr_med = float(np.median(rr)) if rr.size > 0 else (len(y) / fs)
        rr_norm = rr_med if rr_med > 1e-6 else (len(y) / fs)
        peak_time_diff_norm = peak_time_diff_mean / rr_norm
    elif pt_y.size >= 1 and pt_yhat.size >= 1:
        diff = abs(float(pt_y[0] - pt_yhat[0]))
        peak_time_diff_norm = diff / (len(y) / fs)
    else:
        peak_time_diff_norm = 0.5

    A_y = float(np.trapz(np.abs(y)))
    A_yhat = float(np.trapz(np.abs(yhat)))
    area_rel = abs(A_y - A_yhat) / (A_y + 1e-8)

    bwmd = (dtw_norm + lambda_p * peak_time_diff_norm) / (1.0 + lambda_a * area_rel)
    return float(bwmd)


def compute_snr_db(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-12) -> float:
    """
    SNR(dB) = var(signal) / var(error).
    """
    y = np.asarray(y_true, dtype=float).ravel()
    yhat = np.asarray(y_pred, dtype=float).ravel()
    if y.size == 0:
        return -100.0
    err = yhat - y
    signal_var = float(np.var(y, ddof=1)) if y.size > 1 else float(np.var(y))
    noise_var = float(np.var(err, ddof=1)) if err.size > 1 else float(np.var(err))
    if noise_var < eps:
        return 100.0
    if signal_var < eps:
        return -100.0
    return float(10.0 * np.log10(signal_var / noise_var))


def _sigmoid(x: float) -> float:
    x_clip = float(np.clip(x, -60.0, 60.0))
    return float(1.0 / (1.0 + np.exp(-x_clip)))


def compute_sre(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alpha_c: float = 1.0,
    alpha_s: float = 1.0,
    w_c: float = 2.0,
    w_s: float = 0.5,
) -> Tuple[float, float, float, float, float]:
    """
    SRE(Signal Reconstruction Error).

    SRE = RMSE * (1 + w_c*CRI + w_s*SI)
    CRI = sigmoid(alpha_c * (3 - SNR_dB))
    SI  = sigmoid(alpha_s * (SNR_dB - 15))

    Returns:
        sre, rmse, snr_db, cri, si
    """
    y = np.asarray(y_true, dtype=float).ravel()
    yhat = np.asarray(y_pred, dtype=float).ravel()
    if y.size == 0 or yhat.size == 0:
        return 0.0, 0.0, -100.0, 0.0, 0.0

    rmse_val = float(np.sqrt(np.mean((yhat - y) ** 2)))
    snr_db_val = compute_snr_db(y, yhat)

    cri = _sigmoid(alpha_c * (3.0 - snr_db_val))
    si = _sigmoid(alpha_s * (snr_db_val - 15.0))
    sre_val = rmse_val * (1.0 + w_c * cri + w_s * si)
    return float(sre_val), rmse_val, snr_db_val, cri, si


def compute_wcr(y_true: np.ndarray, y_pred: np.ndarray, beta: float = 0.6) -> float:
    """
    WCR(Weighted Correlation).

    WCR = beta*CCC + (1-beta)*SpearmanRho
    """
    x = np.asarray(y_true, dtype=float).ravel()
    y = np.asarray(y_pred, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return 0.0

    ccc = concordance_ccc(x, y)
    try:
        rho_s, _ = spearmanr(x, y)
        if np.isnan(rho_s):
            rho_s = 0.0
    except Exception:
        rx = np.argsort(np.argsort(x))
        ry = np.argsort(np.argsort(y))
        rho_s = float(np.corrcoef(rx, ry)[0, 1]) if rx.size > 1 else 0.0

    wcr_val = beta * ccc + (1.0 - beta) * float(rho_s)
    return float(np.clip(wcr_val, -1.0, 1.0))


def _jsd_from_histograms(p: np.ndarray, q: np.ndarray) -> float:
    p = p / (np.sum(p) + 1e-12)
    q = q / (np.sum(q) + 1e-12)
    m = 0.5 * (p + q)
    kl_pm = entropy(p, m)
    kl_qm = entropy(q, m)
    jsd = 0.5 * (kl_pm + kl_qm)
    jsd_norm = float(jsd / np.log(2.0)) if np.log(2.0) > 0 else float(jsd)
    return float(np.clip(jsd_norm, 0.0, 1.0))


def compute_edd(y_true: np.ndarray, y_pred: np.ndarray, bins: int = 60) -> float:
    """
    EDD(Error Distribution Distance).

    예측 오차 분포와 동일 분산의 Gaussian 분포 간 JSD 기반 거리.
    """
    x = np.asarray(y_true, dtype=float).ravel()
    y = np.asarray(y_pred, dtype=float).ravel()
    if x.size == 0 or y.size == 0:
        return 0.0

    e = y - x
    hist, edges = np.histogram(e, bins=bins, density=False)
    centers = 0.5 * (edges[:-1] + edges[1:])

    P = hist.astype(float) + 1e-12
    sigma = float(np.std(e, ddof=1)) if e.size > 1 else 1.0
    sigma = max(sigma, 1e-8)

    Q_pdf = (1.0 / (np.sqrt(2.0 * np.pi) * sigma)) * np.exp(-0.5 * (centers ** 2) / (sigma ** 2))
    Q = Q_pdf + 1e-12

    return _jsd_from_histograms(P, Q)


# ---------------------------------------------------------------------------
# HRV utilities (peak 기반)
# ---------------------------------------------------------------------------


def compute_hrv_metrics_from_peak_times(peak_times: np.ndarray) -> Dict[str, float]:
    """
    peak time(초)로부터 HRV 메트릭 계산.

    Returns keys:
        mean_rr, median_rr, sdnn, rmssd, pnn50, mean_hr
    """
    pt = np.asarray(peak_times, dtype=float).ravel()
    if pt.size < 2:
        return {
            "mean_rr": np.nan,
            "median_rr": np.nan,
            "sdnn": np.nan,
            "rmssd": np.nan,
            "pnn50": np.nan,
            "mean_hr": np.nan,
        }

    rr = np.diff(pt)
    mean_rr = float(np.mean(rr))
    median_rr = float(np.median(rr))
    sdnn = float(np.std(rr, ddof=1)) if rr.size > 1 else 0.0

    diff_rr = np.diff(rr)
    rmssd = float(np.sqrt(np.mean(diff_rr ** 2))) if diff_rr.size > 0 else 0.0
    nn50 = int(np.sum(np.abs(diff_rr) > 0.05)) if diff_rr.size > 0 else 0
    pnn50 = float(nn50) / float(diff_rr.size) if diff_rr.size > 0 else 0.0
    mean_hr = 60.0 / mean_rr if mean_rr > 0 else np.nan

    return {
        "mean_rr": mean_rr,
        "median_rr": median_rr,
        "sdnn": sdnn,
        "rmssd": rmssd,
        "pnn50": pnn50,
        "mean_hr": mean_hr,
    }


def compute_hrv_metrics(
    signal: np.ndarray,
    fs: float,
    min_distance_sec: float = 0.4,
) -> Dict[str, float]:
    """
    신호에서 peak를 검출하여 HRV 메트릭 계산.
    """
    peak_times = find_peak_times(signal, fs, min_distance_sec=min_distance_sec)
    return compute_hrv_metrics_from_peak_times(peak_times)


# ---------------------------------------------------------------------------
# Composite score v2
# ---------------------------------------------------------------------------


def percentile_normalize(
    value: float,
    reference_values: Sequence[float],
    p_lo: float = 5.0,
    p_hi: float = 95.0,
    lower_is_better: bool = True,
    eps: float = 1e-8,
) -> float:
    """
    reference 분포의 퍼센타일 범위로 값 정규화(0..1).
    """
    ref = np.asarray(reference_values, dtype=float).ravel()
    if ref.size < 3:
        lo, hi = 0.0, max(1.0, float(value))
    else:
        lo = float(np.percentile(ref, p_lo))
        hi = float(np.percentile(ref, p_hi))
        if np.isclose(hi, lo):
            hi = lo + eps

    v = float(np.clip(value, lo, hi))
    return float(np.clip((v - lo) / (hi - lo + eps), 0.0, 1.0))


def composite_v2_from_metrics(
    m_bwmd: float,
    m_sre: float,
    m_wcr: float,
    m_edd: float,
    ref_bwmd: Sequence[float],
    ref_sre: Sequence[float],
    ref_wcr: Sequence[float],
    ref_edd: Sequence[float],
    weights: Optional[Dict[str, float]] = None,
) -> CompositeResult:
    """
    BWMD/SRE/WCR/EDD 기반 Composite v2(0..100) 계산.
    """
    if weights is None:
        weights = {"BWMD": 0.3, "SRE": 0.2, "WCR": 0.3, "EDD": 0.2}

    n_bwmd = percentile_normalize(m_bwmd, ref_bwmd, lower_is_better=True)
    n_sre = percentile_normalize(m_sre, ref_sre, lower_is_better=True)
    n_edd = percentile_normalize(m_edd, ref_edd, lower_is_better=True)
    n_wcr = percentile_normalize(m_wcr, ref_wcr, lower_is_better=False)

    s_bwmd = 1.0 - n_bwmd
    s_sre = 1.0 - n_sre
    s_edd = 1.0 - n_edd
    s_wcr = n_wcr

    wsum = float(sum(weights.values()) + 1e-12)
    agg = (
        weights["BWMD"] * s_bwmd
        + weights["SRE"] * s_sre
        + weights["WCR"] * s_wcr
        + weights["EDD"] * s_edd
    ) / wsum

    score = float(np.clip(agg, 0.0, 1.0) * 100.0)
    breakdown = {"s_bwmd": s_bwmd, "s_sre": s_sre, "s_wcr": s_wcr, "s_edd": s_edd, "agg": agg}
    return CompositeResult(score=score, breakdown=breakdown)


# ---------------------------------------------------------------------------
# Subject/chunk helpers
# ---------------------------------------------------------------------------


def split_by_subjects(
    preds: np.ndarray,
    labels: np.ndarray,
    subject_counts: Sequence[int],
    segment_length: int,
) -> Tuple[List[np.ndarray], List[np.ndarray]]:
    """
    concatenated 신호를 subject별로 분할.
    """
    split_preds: List[np.ndarray] = []
    split_labels: List[np.ndarray] = []
    start = 0
    for count in subject_counts:
        length = int(count) * int(segment_length)
        end = start + length
        split_preds.append(np.asarray(preds[start:end]))
        split_labels.append(np.asarray(labels[start:end]))
        start = end
    return split_preds, split_labels


def compute_biosignal_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    fs: float,
    eval_seconds: int = -1,
    lambda_p: float = 1.0,
    lambda_a: float = 0.5,
    beta_wcr: float = 0.6,
    bins_edd: int = 60,
    min_peak_distance_sec: float = 0.4,
    sre_params: Optional[Dict[str, float]] = None,
) -> BiosignalMetrics:
    """
    단일 신호(또는 subject concat)에 대한 고급 메트릭 계산.

    BWMD는 eval_seconds 단위 청크 평균으로 계산하고,
    다른 지표는 concat 전체 기준으로 계산합니다.
    """
    y = np.asarray(y_true, dtype=float).ravel()
    yhat = np.asarray(y_pred, dtype=float).ravel()
    L = min(y.size, yhat.size)
    if L == 0:
        return BiosignalMetrics(
            bwmd=0.0,
            sre=0.0,
            wcr=0.0,
            edd=0.0,
            rmse=0.0,
            snr_db=-100.0,
            cri=0.0,
            si=0.0,
            hrv_label={},
            hrv_pred={},
            hrv_diff={},
        )

    y = y[:L]
    yhat = yhat[:L]

    if eval_seconds == -1:
        chunk_len = L
    else:
        chunk_len = int(eval_seconds * fs)
        chunk_len = max(chunk_len, 1)

    bwmd_vals: List[float] = []
    y_chunks: List[np.ndarray] = []
    yhat_chunks: List[np.ndarray] = []

    for start in range(0, L, chunk_len):
        end = min(start + chunk_len, L)
        y_c = y[start:end]
        yhat_c = yhat[start:end]
        if y_c.size == 0 or yhat_c.size == 0:
            continue
        try:
            bwmd_vals.append(
                compute_bwmd_for_chunk(
                    y_c,
                    yhat_c,
                    fs=fs,
                    lambda_p=lambda_p,
                    lambda_a=lambda_a,
                    min_distance_sec=min_peak_distance_sec,
                )
            )
        except Exception as e:
            # BWMD 계산 실패 시 경고 (한 번만 출력)
            warnings.warn(
                f"[metrics] BWMD 계산 실패 (청크 {i}): {e}",
                stacklevel=2,
            )
        y_chunks.append(y_c)
        yhat_chunks.append(yhat_c)

    bwmd_mean = float(np.mean(bwmd_vals)) if bwmd_vals else 0.0

    y_concat = np.concatenate(y_chunks) if y_chunks else y
    yhat_concat = np.concatenate(yhat_chunks) if yhat_chunks else yhat

    wcr_val = compute_wcr(y_concat, yhat_concat, beta=beta_wcr)
    edd_val = compute_edd(y_concat, yhat_concat, bins=bins_edd)

    params = sre_params or {}
    sre_val, rmse_val, snr_db_val, cri_val, si_val = compute_sre(y_concat, yhat_concat, **params)

    pt_y = find_peak_times(y_concat, fs, min_distance_sec=min_peak_distance_sec)
    pt_yhat = find_peak_times(yhat_concat, fs, min_distance_sec=min_peak_distance_sec)
    hrv_label = compute_hrv_metrics_from_peak_times(pt_y)
    hrv_pred = compute_hrv_metrics_from_peak_times(pt_yhat)

    hrv_diff: Dict[str, float] = {}
    for k in hrv_label.keys():
        a = hrv_label.get(k, np.nan)
        b = hrv_pred.get(k, np.nan)
        if np.isnan(a) or np.isnan(b):
            hrv_diff[k] = np.nan
        else:
            hrv_diff[k] = float(abs(a - b))

    return BiosignalMetrics(
        bwmd=bwmd_mean,
        sre=sre_val,
        wcr=wcr_val,
        edd=edd_val,
        rmse=rmse_val,
        snr_db=snr_db_val,
        cri=cri_val,
        si=si_val,
        hrv_label=hrv_label,
        hrv_pred=hrv_pred,
        hrv_diff=hrv_diff,
    )


def compute_subject_metrics(
    preds: np.ndarray,
    labels: np.ndarray,
    subject_counts: Sequence[int],
    fs: float,
    eval_seconds: int = 10,
    segment_seconds: Optional[int] = None,
    **kwargs,
) -> List[BiosignalMetrics]:
    """
    concatenated 신호를 subject별로 분할한 뒤 subject 단위 메트릭 리스트 반환.
    """
    seg_len = int((segment_seconds or eval_seconds) * fs)
    preds_split, labels_split = split_by_subjects(preds, labels, subject_counts, segment_length=seg_len)

    subject_metrics: List[BiosignalMetrics] = []
    for p_sub, l_sub in zip(preds_split, labels_split):
        subject_metrics.append(
            compute_biosignal_metrics(
                l_sub,
                p_sub,
                fs=fs,
                eval_seconds=eval_seconds,
                **kwargs,
            )
        )
    return subject_metrics
