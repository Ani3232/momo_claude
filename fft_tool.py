"""
fft_tool.py
-----------
FFT analysis tool for agent use.
Reads a headerless single-column CSV file and performs FFT analysis.
Returns a structured JSON summary of dominant frequencies, band energy,
and spectral statistics. Optionally saves publication-quality plots
(PNG 1000 DPI + SVG) following the graph_design skill conventions.

Usage (as an agent tool):
    result = fft_analysis("path/to/signal.csv", sample_rate=8000, plot=True)
    # result is a dict — pass as JSON to the LLM
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
from typing import Optional
from pathlib import Path
from setup import workspace
# =============================================================================
# Graph Design Skill — Configuration
# =============================================================================
FIG_WIDTH, ASPECT_RATIO = 7.25, 1.3          # double-column default
FIG_HEIGHT = FIG_WIDTH / ASPECT_RATIO

COLORS = {
    "primary":     "#88CCEE",   # light blue  — main spectrum line
    "secondary":   "#CC6677",   # dusty red   — dominant peak markers
    "tertiary":    "#DDCC77",   # yellowish   — band fills
    "quaternary":  "#6699CC",   # blue        — secondary band
    "quinary":     "#888888",   # gray        — noise floor
    "accent":      "#EE7733",   # orange      — threshold / highlight
    "grid":        "#CCCCCC",
}

MARKER_SIZE        = 10
MARKER_EDGE_WIDTH  = 1.5
DPI_PNG            = 1000
DPI_SVG            = 72

plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.labelsize":   11,
    "axes.titlesize":   12,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
})


def _save_fig(fig, base_path: str):
    """Save as PNG (1000 DPI) + SVG — graph_design convention."""
    # Resolve relative paths against workspace
    WORKSPACE = workspace
    if not os.path.isabs(base_path):
        base_path = str(WORKSPACE / base_path)

    os.makedirs(os.path.dirname(base_path) or ".", exist_ok=True)
    fig.savefig(f"{base_path}.png", dpi=DPI_PNG, bbox_inches="tight", facecolor="white")
    fig.savefig(f"{base_path}.svg", dpi=DPI_SVG, bbox_inches="tight", facecolor="white")
    print(f"  Exported: {base_path}.png  ({DPI_PNG} DPI)")
    print(f"  Exported: {base_path}.svg  (vector)")

# =============================================================================
# Core FFT Analysis
# =============================================================================

# =============================================================================
# CSV Loader
# =============================================================================

def _load_csv(csv_path: str) -> np.ndarray:
    """
    Load a headerless, single-column CSV file into a 1-D numpy array.
    Skips blank lines and raises clear errors on bad data.
    """
    WORKSPACE = workspace
    csv_path = str(WORKSPACE / csv_path) if not os.path.isabs(csv_path) else csv_path

    if not os.path.isfile(csv_path):
        raise FileNotFoundError(f"CSV not found: {csv_path!r}")

    values = []
    with open(csv_path, "r") as f:
        for lineno, raw in enumerate(f, start=1):
            line = raw.strip()
            if not line:
                continue                        # skip blank lines
            # reject multi-column rows
            cols = line.split(",")
            if len(cols) > 1:
                raise ValueError(
                    f"Line {lineno}: expected 1 column, got {len(cols)}. "
                    f"Make sure the CSV has no header and only one column."
                )
            try:
                values.append(float(cols[0]))
            except ValueError:
                raise ValueError(
                    f"Line {lineno}: cannot convert {cols[0]!r} to float."
                )

    if not values:
        raise ValueError(f"CSV file is empty: {csv_path!r}")

    return np.array(values, dtype=float)


# =============================================================================
# Core FFT Analysis
# =============================================================================

def fft_analysis(
    csv_path:     str,
    sample_rate:  float,
    top_n:        int             = 10,
    window:       str             = "hann",    # "hann" | "hamming" | "blackman" | "none"
    detrend:      bool            = True,
    plot:         bool            = False,
    plot_path:    str             = "fft_output",
    plot_title:   str             = "FFT Spectrum Analysis",
    freq_min:     Optional[float] = None,      # zoom x-axis (Hz)
    freq_max:     Optional[float] = None,
    log_scale:    bool            = False,     # log x-axis (useful for wide-band signals)
) -> dict:
    """
    Run FFT analysis on a headerless single-column CSV file.

    Parameters
    ----------
    csv_path    : path to a headerless, single-column CSV file
    sample_rate : samples per second (Hz)
    top_n       : number of dominant frequencies to return
    window      : windowing function to reduce spectral leakage
    detrend     : remove linear trend before analysis
    plot        : if True, save publication-quality spectrum plot
    plot_path   : file path base (no extension) for saved plots
    plot_title  : figure title
    freq_min/max: optional x-axis limits for the plot
    log_scale   : use logarithmic frequency axis

    Returns
    -------
    dict with full spectral summary — safe to serialise as JSON
    """

    x = _load_csv(csv_path)
    print(f"  Loaded {len(x)} samples from {csv_path!r}")
    N = len(x)

    # ── Pre-processing ────────────────────────────────────────────────────────
    if detrend:
        x = x - np.polyval(np.polyfit(np.arange(N), x, 1), np.arange(N))

    x -= x.mean()   # demean after detrend

    # ── Windowing ─────────────────────────────────────────────────────────────
    window_fns = {
        "hann":     np.hanning,
        "hamming":  np.hamming,
        "blackman": np.blackman,
        "none":     np.ones,
    }
    if window not in window_fns:
        raise ValueError(f"Unknown window '{window}'. Choose from: {list(window_fns)}")
    win      = window_fns[window](N)
    win_gain = np.mean(win ** 2) ** 0.5   # amplitude correction factor
    x_win    = x * win

    # ── FFT ───────────────────────────────────────────────────────────────────
    fft_raw    = np.fft.rfft(x_win)
    freqs      = np.fft.rfftfreq(N, d=1.0 / sample_rate)
    magnitudes = (np.abs(fft_raw) * 2) / (N * win_gain)   # normalised amplitude
    power_db   = 20 * np.log10(magnitudes + 1e-12)         # dB (floor at -240 dB)

    # ── Dominant Frequencies ─────────────────────────────────────────────────
    top_idx = np.argsort(magnitudes)[-top_n:][::-1]
    dominant = [
        {
            "rank":          int(rank + 1),
            "frequency_hz":  round(float(freqs[i]), 6),
            "magnitude":     round(float(magnitudes[i]), 8),
            "power_db":      round(float(power_db[i]), 4),
        }
        for rank, i in enumerate(top_idx)
    ]

    # ── Spectral Statistics ───────────────────────────────────────────────────
    total_power    = float(np.sum(magnitudes ** 2))
    noise_floor_db = float(np.percentile(power_db, 10))   # 10th percentile ≈ noise
    peak_db        = float(np.max(power_db))
    snr_db         = round(peak_db - noise_floor_db, 4)

    # Spectral centroid
    centroid_hz = float(np.sum(freqs * magnitudes) / (np.sum(magnitudes) + 1e-12))

    # Spectral bandwidth (RMS spread around centroid)
    bandwidth_hz = float(
        np.sqrt(np.sum(((freqs - centroid_hz) ** 2) * magnitudes)
                / (np.sum(magnitudes) + 1e-12))
    )

    # ── Band Energy ───────────────────────────────────────────────────────────
    nyquist    = sample_rate / 2
    band_edges = [0, nyquist * 0.05, nyquist * 0.25, nyquist * 0.5, nyquist]
    band_names = ["DC–5%Ny", "5–25%Ny", "25–50%Ny", "50–100%Ny"]

    band_energy = {}
    for name, lo, hi in zip(band_names, band_edges, band_edges[1:]):
        mask = (freqs >= lo) & (freqs < hi)
        frac = float(np.sum(magnitudes[mask] ** 2) / (total_power + 1e-12))
        band_energy[name] = {
            "range_hz":         [round(lo, 4), round(hi, 4)],
            "energy_fraction":  round(frac, 6),
            "energy_pct":       round(frac * 100, 4),
        }

    # ── Assemble Result ───────────────────────────────────────────────────────
    result = {
        "metadata": {
            "sample_count":   N,
            "sample_rate_hz": sample_rate,
            "nyquist_hz":     nyquist,
            "frequency_resolution_hz": round(float(freqs[1] - freqs[0]), 8),
            "window_function": window,
            "detrended":      detrend,
            "duration_s":     round(N / sample_rate, 6),
        },
        "spectral_stats": {
            "peak_db":          round(peak_db, 4),
            "noise_floor_db":   round(noise_floor_db, 4),
            "snr_db":           snr_db,
            "centroid_hz":      round(centroid_hz, 6),
            "bandwidth_hz":     round(bandwidth_hz, 6),
            "total_power":      round(total_power, 8),
        },
        "dominant_frequencies": dominant,
        "band_energy":          band_energy,
    }

    # ── Optional Plot ─────────────────────────────────────────────────────────
    if plot:
        _plot_spectrum(
            freqs, magnitudes, power_db, result,
            plot_path, plot_title, freq_min, freq_max, log_scale
        )
        result["plots"] = {
            "png": f"{plot_path}.png",
            "svg": f"{plot_path}.svg",
        }

    return result


# =============================================================================
# Publication-Quality Spectrum Plot
# =============================================================================

def _plot_spectrum(freqs, magnitudes, power_db, result,
                   plot_path, title, freq_min, freq_max, log_scale):
    """
    Two-panel figure:
      Top    — linear amplitude spectrum with dominant peaks annotated
      Bottom — dB spectrum with noise floor reference
    """
    fig, (ax_amp, ax_db) = plt.subplots(
        2, 1,
        figsize=(FIG_WIDTH, FIG_HEIGHT * 1.5),  # taller for 2-panel
        sharex=True
    )

    meta   = result["metadata"]
    stats  = result["spectral_stats"]
    dom    = result["dominant_frequencies"]
    nyq    = meta["nyquist_hz"]

    # ── X limits ─────────────────────────────────────────────────────────────
    xlo = freq_min if freq_min is not None else (freqs[1] if log_scale else 0)
    xhi = freq_max if freq_max is not None else nyq

    # ── Band region fills (z-order 0) ─────────────────────────────────────────
    band_colors = [COLORS["primary"], COLORS["tertiary"],
                   COLORS["quaternary"], COLORS["quinary"]]
    for ax in (ax_amp, ax_db):
        for (name, bdata), bcolor in zip(result["band_energy"].items(), band_colors):
            lo, hi = bdata["range_hz"]
            ax.axvspan(lo, hi, alpha=0.08, color=bcolor, zorder=0)

    # ── Amplitude spectrum (top panel) ────────────────────────────────────────
    ax_amp.plot(freqs, magnitudes,
                color=COLORS["primary"], linewidth=1.2, zorder=3, label="Amplitude")

    # Dominant peak markers
    top_freqs = [d["frequency_hz"] for d in dom[:5]]
    top_mags  = [d["magnitude"]    for d in dom[:5]]
    ax_amp.plot(top_freqs, top_mags, "o",
                color=COLORS["secondary"],
                markersize=MARKER_SIZE,
                markeredgecolor="#FFFFFF",
                markeredgewidth=MARKER_EDGE_WIDTH,
                zorder=4, label=f"Top {min(5, len(dom))} peaks")

    # Annotate top-3 peaks
    for d in dom[:3]:
        f, m = d["frequency_hz"], d["magnitude"]
        label = f'{f:.2f} Hz' if f < 1000 else f'{f/1000:.2f} kHz'
        ax_amp.annotate(
            label, (f, m),
            textcoords="offset points", xytext=(8, 6),
            fontsize=8, color=COLORS["secondary"], zorder=5,
            arrowprops=dict(arrowstyle="-", color=COLORS["secondary"],
                            lw=0.8, alpha=0.7)
        )

    ax_amp.set_ylabel("Amplitude", fontsize=11)
    ax_amp.spines["top"].set_visible(False)
    ax_amp.spines["right"].set_visible(False)
    ax_amp.grid(True, which="major", linestyle="-",  alpha=0.5, color=COLORS["grid"])
    ax_amp.grid(True, which="minor", linestyle=":",  alpha=0.3, color=COLORS["grid"])
    ax_amp.legend(fontsize=8, framealpha=0.7, loc="upper right")

    # ── dB spectrum (bottom panel) ────────────────────────────────────────────
    ax_db.plot(freqs, power_db,
               color=COLORS["quaternary"], linewidth=1.2, zorder=3, label="Power (dB)")

    # Noise floor reference line (z-order 2)
    nf = stats["noise_floor_db"]
    ax_db.axhline(nf, color=COLORS["accent"], linestyle="--",
                  linewidth=1.2, zorder=2, alpha=0.85, label=f"Noise floor ≈ {nf:.1f} dB")

    # Fill below noise floor
    ax_db.fill_between(freqs, power_db.min() - 5, nf,
                        alpha=0.10, color=COLORS["accent"], zorder=1)

    ax_db.set_ylabel("Power (dB)", fontsize=11)
    ax_db.set_xlabel("Frequency (Hz)", fontsize=11)
    ax_db.spines["top"].set_visible(False)
    ax_db.spines["right"].set_visible(False)
    ax_db.grid(True, which="major", linestyle="-",  alpha=0.5, color=COLORS["grid"])
    ax_db.grid(True, which="minor", linestyle=":",  alpha=0.3, color=COLORS["grid"])
    ax_db.legend(fontsize=8, framealpha=0.7, loc="upper right")

    # ── Shared x-axis settings ────────────────────────────────────────────────
    ax_db.set_xlim(xlo, xhi)
    if log_scale:
        ax_db.set_xscale("log")
        ax_amp.set_xscale("log")

    # ── Title + stats annotation ──────────────────────────────────────────────
    fig.suptitle(title, fontsize=12, fontweight="bold", y=1.01)
    stats_text = (
        f"fs={meta['sample_rate_hz']} Hz  |  N={meta['sample_count']}  |  "
        f"Δf={meta['frequency_resolution_hz']:.4f} Hz  |  "
        f"SNR={stats['snr_db']:.1f} dB  |  "
        f"Centroid={stats['centroid_hz']:.2f} Hz"
    )
    fig.text(0.5, -0.01, stats_text, ha="center", fontsize=8,
             color=COLORS["quinary"], style="italic")

    plt.tight_layout()
    _save_fig(fig, plot_path)
    plt.close(fig)


# =============================================================================
# Agent Tool Wrapper — returns JSON string for LLM consumption
# =============================================================================

def fft_tool(
    csv_path:    str,
    sample_rate: float,
    top_n:       int   = 10,
    window:      str   = "hann",
    detrend:     bool  = True,
    plot:        bool  = True,
    plot_path:   str   = "fft_output",
    plot_title:  str   = "FFT Spectrum Analysis",
    freq_min:    float = None,
    freq_max:    float = None,
    log_scale:   bool  = False,
) -> str:
    """
    Agent-facing wrapper — returns a compact JSON string.
    Register this as a tool in your agent's tool registry.
    """
    result = fft_analysis(
        csv_path=csv_path, sample_rate=sample_rate,
        top_n=top_n, window=window, detrend=detrend,
        plot=plot, plot_path=plot_path, plot_title=plot_title,
        freq_min=freq_min, freq_max=freq_max, log_scale=log_scale,
    )
    return json.dumps(result, indent=2)


# =============================================================================
# Quick smoke-test
# =============================================================================

if __name__ == "__main__":
    # Synthetic signal: 50 Hz + 200 Hz + 1 kHz + noise
    FS  = 8000
    T   = 2.0
    t   = np.linspace(0, T, int(FS * T), endpoint=False)
    sig = (
        1.0  * np.sin(2 * np.pi *   50 * t) +
        0.6  * np.sin(2 * np.pi *  200 * t) +
        0.3  * np.sin(2 * np.pi * 1000 * t) +
        0.05 * np.random.randn(len(t))
    )

    # Write to a headerless single-column CSV
    os.makedirs("fft_output", exist_ok=True)
    csv_path = "fft_output/smoke_test_signal.csv"
    np.savetxt(csv_path, sig, fmt="%.8f")
    print(f"  Wrote {len(sig)} samples → {csv_path!r}")

    out = fft_analysis(
        csv_path=csv_path,
        sample_rate=FS,
        top_n=10,
        window="hann",
        detrend=True,
        plot=True,
        plot_path="fft_output/smoke_test",
        plot_title="Smoke Test — 50 Hz + 200 Hz + 1 kHz",
        log_scale=False,
    )

    print("\n── Spectral Stats ──────────────────────────────────")
    for k, v in out["spectral_stats"].items():
        print(f"  {k:25s}: {v}")

    print("\n── Top 5 Dominant Frequencies ──────────────────────")
    for d in out["dominant_frequencies"][:5]:
        print(f"  #{d['rank']}  {d['frequency_hz']:>10.3f} Hz  |  "
              f"{d['magnitude']:.6f} amp  |  {d['power_db']:.2f} dB")

    print("\n── Band Energy ─────────────────────────────────────")
    for band, bdata in out["band_energy"].items():
        print(f"  {band:15s}: {bdata['energy_pct']:.2f}%")