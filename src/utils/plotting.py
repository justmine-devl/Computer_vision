from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np

from utils.image_frequency import dark_channel, fft_log_magnitude, gradient_map, luminance


def save_sample_grid(samples, out_path: Path, title: str) -> None:
    cols = min(6, len(samples))
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3 * rows))
    axes = np.array(axes).reshape(rows, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for ax, (domain, path, img) in zip(axes.ravel(), samples):
        ax.imshow(img)
        ax.set_title(f"{domain}\n{path.name}", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_fft_grid(samples, out_path: Path, title: str) -> None:
    cols = min(6, len(samples))
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows * 2, cols, figsize=(3 * cols, 5 * rows))
    axes = np.array(axes).reshape(rows * 2, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (domain, path, img) in enumerate(samples):
        r = (i // cols) * 2
        c = i % cols
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"{domain}\n{path.name}", fontsize=8)
        axes[r + 1, c].imshow(fft_log_magnitude(luminance(img)), cmap="magma")
        axes[r + 1, c].set_title("log FFT magnitude", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_gradient_dark_channel_grid(samples, out_path: Path, title: str) -> None:
    cols = min(6, len(samples))
    rows = int(np.ceil(len(samples) / cols))
    fig, axes = plt.subplots(rows * 3, cols, figsize=(3 * cols, 7 * rows))
    axes = np.array(axes).reshape(rows * 3, cols)
    for ax in axes.ravel():
        ax.axis("off")
    for i, (domain, path, img) in enumerate(samples):
        r = (i // cols) * 3
        c = i % cols
        axes[r, c].imshow(img)
        axes[r, c].set_title(f"{domain}\n{path.name}", fontsize=8)
        axes[r + 1, c].imshow(gradient_map(luminance(img)), cmap="gray")
        axes[r + 1, c].set_title("gradient magnitude", fontsize=8)
        axes[r + 2, c].imshow(dark_channel(img), cmap="viridis")
        axes[r + 2, c].set_title("dark channel proxy", fontsize=8)
    fig.suptitle(title)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_radial_profiles(domain_profiles: dict[str, list[np.ndarray]], out_path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    for domain, profiles in sorted(domain_profiles.items()):
        if not profiles:
            continue
        arr = np.stack(profiles)
        mean = arr.mean(axis=0)
        std = arr.std(axis=0)
        x = np.linspace(0, 1, len(mean))
        ax.plot(x, mean, label=domain)
        ax.fill_between(x, mean - std, mean + std, alpha=0.16)
    ax.set_title(title)
    ax.set_xlabel("normalized spatial frequency")
    ax.set_ylabel("normalized log magnitude")
    ax.grid(True, alpha=0.25)
    ax.legend()
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_metric_bars(stats: list[dict[str, object]], metrics: list[str], out_path: Path, title: str) -> None:
    domains = sorted(set(str(row["domain"]) for row in stats))
    x = np.arange(len(metrics))
    width = 0.8 / max(len(domains), 1)
    fig, ax = plt.subplots(figsize=(10, 4))
    for i, domain in enumerate(domains):
        rows = [r for r in stats if r["domain"] == domain]
        means = [np.mean([float(r[m]) for r in rows]) for m in metrics]
        ax.bar(x + i * width, means, width, label=domain)
    ax.set_xticks(x + width * (len(domains) - 1) / 2)
    ax.set_xticklabels(metrics, rotation=20)
    ax.set_title(title)
    ax.legend(fontsize=8)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


