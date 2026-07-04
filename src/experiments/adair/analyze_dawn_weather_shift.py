import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
SRC_DIR = ROOT / "src"
DL_NETS_DIR = ROOT / "dl_nets"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DL_NETS_DIR))


from utils.domain_shift import DomainShiftAnalyzer, DomainShiftConfig, DomainSpec


def run_group(name: str, domains: list[DomainSpec], output_dir: Path, samples: int, image_size: int, seed: int) -> None:
    interpretation = (
        "DAWN weather images are real traffic-domain examples, while common restoration benchmarks often encode "
        "cleaner and more controlled degradation patterns. Differences in FFT energy bands, edge maps, saturation, "
        "and dark-channel statistics indicate whether a restoration model trained on the benchmark is likely to "
        "transfer zero-shot to DAWN."
    )
    analyzer = DomainShiftAnalyzer(
        domains=domains,
        config=DomainShiftConfig(
            output_dir=output_dir / name,
            samples_per_domain=samples,
            image_size=image_size,
            seed=seed,
            title=f"DAWN {name.title()} vs Restoration Benchmark Domain Shift",
            report_name=f"dawn_{name}_domain_shift_report.md",
            interpretation=interpretation,
        ),
    )
    print(analyzer.run())


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare DAWN Rain/Snow with Rain100L, RainDrop, and Snow100K.")
    parser.add_argument("--dawn-rain-dir", required=True)
    parser.add_argument("--dawn-snow-dir", required=True)
    parser.add_argument("--rain100l-dir", required=True)
    parser.add_argument("--raindrop-dir", required=True)
    parser.add_argument("--snow100k-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-domain", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    run_group(
        "rain",
        [
            DomainSpec("DAWN-Rain", Path(args.dawn_rain_dir)),
            DomainSpec("Rain100L", Path(args.rain100l_dir)),
            DomainSpec("RainDrop", Path(args.raindrop_dir)),
        ],
        output_dir,
        args.samples_per_domain,
        args.image_size,
        args.seed,
    )
    run_group(
        "snow",
        [
            DomainSpec("DAWN-Snow", Path(args.dawn_snow_dir)),
            DomainSpec("Snow100K", Path(args.snow100k_dir)),
        ],
        output_dir,
        args.samples_per_domain,
        args.image_size,
        args.seed,
    )


if __name__ == "__main__":
    main()

