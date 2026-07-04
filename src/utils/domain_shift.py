from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

from utils.image_frequency import (
    fft_log_magnitude,
    image_statistics,
    list_images,
    load_rgb_canvas,
    luminance,
    radial_profile,
)
from utils.plotting import (
    save_fft_grid,
    save_gradient_dark_channel_grid,
    save_metric_bars,
    save_radial_profiles,
    save_sample_grid,
)


@dataclass(frozen=True)
class DomainSpec:
    name: str
    root: Path
    exclude_tokens: tuple[str, ...] = ("yolo", "pascal", "voc", "annotation", "label")


@dataclass(frozen=True)
class DomainShiftConfig:
    output_dir: Path
    samples_per_domain: int = 8
    image_size: int = 384
    seed: int = 42
    title: str = "Domain Shift Analysis"
    report_name: str = "domain_shift_report.md"
    interpretation: str = ""


class DomainShiftAnalyzer:
    def __init__(self, domains: list[DomainSpec], config: DomainShiftConfig):
        self.domains = domains
        self.config = config

    def collect_samples(self):
        samples = []
        counts = {}
        for domain in self.domains:
            paths = list_images(
                domain.root,
                max_images=self.config.samples_per_domain,
                seed=self.config.seed,
                exclude_tokens=domain.exclude_tokens,
            )
            counts[domain.name] = len(paths)
            for path in paths:
                samples.append((domain.name, path, load_rgb_canvas(path, self.config.image_size)))
        missing = [name for name, count in counts.items() if count == 0]
        if missing:
            raise RuntimeError(f"No images found for domains {missing}. Counts: {counts}")
        return samples, counts

    def run(self) -> dict[str, Path]:
        out_dir = self.config.output_dir
        fig_dir = out_dir / "figures"
        fig_dir.mkdir(parents=True, exist_ok=True)

        samples, counts = self.collect_samples()
        save_sample_grid(samples, fig_dir / "sample_contact_sheet.png", self.config.title)
        save_fft_grid(samples, fig_dir / "fft_examples.png", f"{self.config.title}: FFT")
        save_gradient_dark_channel_grid(
            samples,
            fig_dir / "gradient_dark_channel_examples.png",
            f"{self.config.title}: edges and dark channel",
        )

        profiles: dict[str, list] = {domain.name: [] for domain in self.domains}
        stats: list[dict[str, object]] = []
        for domain, path, img in samples:
            gray = luminance(img)
            profiles[domain].append(radial_profile(fft_log_magnitude(gray)))
            row: dict[str, object] = {"domain": domain, "path": str(path)}
            row.update(image_statistics(img))
            stats.append(row)

        save_radial_profiles(profiles, fig_dir / "radial_fft_profiles.png", f"{self.config.title}: radial FFT profile")
        save_metric_bars(
            stats,
            ["low_energy", "mid_energy", "high_energy"],
            fig_dir / "frequency_band_energy.png",
            f"{self.config.title}: FFT energy bands",
        )
        save_metric_bars(
            stats,
            ["std_luma", "mean_saturation", "mean_gradient", "mean_dark_channel"],
            fig_dir / "spatial_color_metrics.png",
            f"{self.config.title}: spatial/color metrics",
        )

        metrics_path = out_dir / "domain_stats.csv"
        with metrics_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(stats[0].keys()))
            writer.writeheader()
            writer.writerows(stats)

        report_path = out_dir / self.config.report_name
        report_path.write_text(self._report_text(counts), encoding="utf-8")
        return {"output_dir": out_dir, "metrics": metrics_path, "report": report_path}

    def _report_text(self, counts: dict[str, int]) -> str:
        counts_text = "\n".join(f"- {name}: {count}" for name, count in counts.items())
        return f"""# {self.config.title}

## Sample Counts

{counts_text}

## Generated Figures

- `figures/sample_contact_sheet.png`
- `figures/fft_examples.png`
- `figures/radial_fft_profiles.png`
- `figures/frequency_band_energy.png`
- `figures/gradient_dark_channel_examples.png`
- `figures/spatial_color_metrics.png`
- `domain_stats.csv`

## Interpretation

{self.config.interpretation}
"""
