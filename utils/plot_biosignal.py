"""
생체신호(Biosignal) 전용 시각화 유틸리티

학습 곡선(loss)이나 범용 prediction plot은 utils/plot.py에서 담당하고,
이 모듈은 신호 특화(Time‑domain/PSD/subject‑chunk 비교) 플롯을 제공합니다.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Optional

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.cm import get_cmap
from scipy.signal import welch

from .signal import bandpass_filter, detrend as detrend_signal


def z_normalize(signal: np.ndarray, eps: float = 1e-8) -> np.ndarray:
    """z‑normalization: (x - mean) / std"""
    x = np.asarray(signal, dtype=float).ravel()
    std = float(np.std(x)) if x.size > 0 else 0.0
    if std < eps:
        return x * 0.0
    return (x - float(np.mean(x))) / std


def process_signal(
    signal: np.ndarray,
    fs: float = 30.0,
    detrend_on: bool = True,
    bpf_on: bool = True,
    lambda_val: float = 100.0,
    low: float = 0.75,
    high: float = 3.0,
) -> np.ndarray:
    """
    평가/플롯용 전처리(detrend + bandpass + z‑norm).
    """
    x = np.asarray(signal, dtype=float).ravel()
    if x.size == 0:
        return x
    if detrend_on:
        x = detrend_signal(x, lambda_val=lambda_val)
    if bpf_on:
        x = bandpass_filter(x, fs=fs, low=low, high=high)
    return z_normalize(x)


def plot_subject_signals_chunks(
    subject_id: int,
    pred_signal: np.ndarray,
    label_signal: np.ndarray,
    fs: float = 30.0,
    eval_seconds: int = -1,
    detrend_on: bool = True,
    bpf_on: bool = True,
    lambda_val: float = 100.0,
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    단일 모델 prediction vs label을 subject별/청크별로 Time‑domain & PSD 플롯.

    Args:
        subject_id: subject index (1‑based 권장)
        pred_signal, label_signal: 1D signals
        fs: sampling rate
        eval_seconds: -1이면 전체, 그 외는 eval_seconds 단위 청크
        save_path: 저장할 파일 경로(.png). None이면 저장 안 함
        show: True면 plt.show()
    """
    p = np.asarray(pred_signal, dtype=float).ravel()
    l = np.asarray(label_signal, dtype=float).ravel()
    L = min(p.size, l.size)
    p = p[:L]
    l = l[:L]

    def proc(x: np.ndarray) -> np.ndarray:
        return process_signal(x, fs=fs, detrend_on=detrend_on, bpf_on=bpf_on, lambda_val=lambda_val)

    if eval_seconds == -1:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))
        p_proc = proc(p)
        l_proc = proc(l)

        axs[0].plot(p_proc, label="Prediction", color="tab:blue")
        axs[0].plot(l_proc, label="Label", color="tab:red", alpha=0.7)
        axs[0].set_title(f"Subject {subject_id} - Time Domain")
        axs[0].set_xlabel("Time (frame)")
        axs[0].set_ylabel("Amplitude (z-score)")
        axs[0].legend()

        nperseg = min(256, L) if L > 0 else 1
        f_p, psd_p = welch(p_proc, fs=fs, nperseg=nperseg)
        f_l, psd_l = welch(l_proc, fs=fs, nperseg=nperseg)
        axs[1].plot(f_p, psd_p, label="Pred PSD", color="tab:blue")
        axs[1].plot(f_l, psd_l, label="Label PSD", color="tab:red", alpha=0.7)
        axs[1].set_title("Power Spectral Density")
        axs[1].set_xlabel("Frequency (Hz)")
        axs[1].set_ylabel("PSD")
        axs[1].legend()

    else:
        chunk_len = max(1, int(eval_seconds * fs))
        n_chunks = int(np.ceil(L / chunk_len))
        fig, axs = plt.subplots(n_chunks, 2, figsize=(12, 4 * n_chunks))
        if n_chunks == 1:
            axs = np.array([axs])

        for i in range(n_chunks):
            start = i * chunk_len
            end = min((i + 1) * chunk_len, L)
            p_c = proc(p[start:end])
            l_c = proc(l[start:end])

            axs[i, 0].plot(p_c, label="Prediction", color="tab:blue")
            axs[i, 0].plot(l_c, label="Label", color="tab:red", alpha=0.7)
            axs[i, 0].set_title(f"Subject {subject_id} - Segment {i + 1} Time")
            axs[i, 0].set_xlabel("Time (frame)")
            axs[i, 0].set_ylabel("Amplitude (z-score)")
            axs[i, 0].legend()

            nperseg = min(256, end - start) if (end - start) > 0 else 1
            f_p, psd_p = welch(p_c, fs=fs, nperseg=nperseg)
            f_l, psd_l = welch(l_c, fs=fs, nperseg=nperseg)
            axs[i, 1].plot(f_p, psd_p, label="Pred PSD", color="tab:blue")
            axs[i, 1].plot(f_l, psd_l, label="Label PSD", color="tab:red", alpha=0.7)
            axs[i, 1].set_title(f"Subject {subject_id} - Segment {i + 1} PSD")
            axs[i, 1].set_xlabel("Frequency (Hz)")
            axs[i, 1].set_ylabel("PSD")
            axs[i, 1].legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)


def plot_multi_subject_signals_chunks(
    subject_id: int,
    preds_dict: Dict[str, np.ndarray],
    label_signal: np.ndarray,
    fs: float = 30.0,
    eval_seconds: int = -1,
    detrend_on: bool = True,
    bpf_on: bool = True,
    lambda_val: float = 100.0,
    save_path: Optional[Path] = None,
    show: bool = False,
):
    """
    여러 모델(preds_dict) prediction과 단일 label을 청크별 Time‑domain/PSD로 비교 플롯.
    """
    model_names = list(preds_dict.keys())
    cmap = get_cmap("tab10")
    l = np.asarray(label_signal, dtype=float).ravel()

    def proc(x: np.ndarray) -> np.ndarray:
        return process_signal(x, fs=fs, detrend_on=detrend_on, bpf_on=bpf_on, lambda_val=lambda_val)

    lengths = [len(np.asarray(preds_dict[m]).ravel()) for m in model_names] + [l.size]
    L = int(min(lengths))
    l = l[:L]
    preds = {m: np.asarray(preds_dict[m], dtype=float).ravel()[:L] for m in model_names}

    if eval_seconds == -1:
        fig, axs = plt.subplots(2, 1, figsize=(10, 8))

        for i, name in enumerate(model_names):
            axs[0].plot(proc(preds[name]), label=name, color=cmap(i))
        axs[0].plot(proc(l), "--", label="Label", color="black")
        axs[0].set_title(f"Subject {subject_id} - Time Domain")
        axs[0].legend()

        nperseg = min(256, L) if L > 0 else 1
        for i, name in enumerate(model_names):
            f_p, psd_p = welch(proc(preds[name]), fs=fs, nperseg=nperseg)
            axs[1].plot(f_p, psd_p, label=f"{name} PSD", color=cmap(i))
        f_l, psd_l = welch(proc(l), fs=fs, nperseg=nperseg)
        axs[1].plot(f_l, psd_l, "--", label="Label PSD", color="black")
        axs[1].set_title("Power Spectral Density")
        axs[1].legend()

    else:
        chunk_len = max(1, int(eval_seconds * fs))
        n_chunks = int(np.ceil(L / chunk_len))
        fig, axs = plt.subplots(n_chunks, 2, figsize=(12, 4 * n_chunks))
        if n_chunks == 1:
            axs = np.array([axs])

        for seg in range(n_chunks):
            start = seg * chunk_len
            end = min((seg + 1) * chunk_len, L)

            for i, name in enumerate(model_names):
                axs[seg, 0].plot(proc(preds[name][start:end]), label=name, color=cmap(i))
            axs[seg, 0].plot(proc(l[start:end]), "--", label="Label", color="black")
            axs[seg, 0].set_title(f"Subject {subject_id} - Segment {seg + 1} Time")
            axs[seg, 0].legend()

            nperseg = min(256, end - start) if (end - start) > 0 else 1
            for i, name in enumerate(model_names):
                f_p, psd_p = welch(proc(preds[name][start:end]), fs=fs, nperseg=nperseg)
                axs[seg, 1].plot(f_p, psd_p, label=f"{name} PSD", color=cmap(i))
            f_l, psd_l = welch(proc(l[start:end]), fs=fs, nperseg=nperseg)
            axs[seg, 1].plot(f_l, psd_l, "--", label="Label PSD", color="black")
            axs[seg, 1].set_title(f"Subject {subject_id} - Segment {seg + 1} PSD")
            axs[seg, 1].legend()

    plt.tight_layout()
    if save_path is not None:
        save_path.parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(save_path, dpi=150)
    if show:
        plt.show()
    plt.close(fig)

