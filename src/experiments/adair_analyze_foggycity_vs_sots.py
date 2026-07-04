import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DL_NETS_DIR = ROOT / "dl_nets"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DL_NETS_DIR))


from utils.domain_shift import DomainShiftAnalyzer, DomainShiftConfig, DomainSpec


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Foggy Cityscapes with SOTS to explain why AdaIR restores it better than DAWN-Fog.")
    parser.add_argument("--sots-dir", required=True)
    parser.add_argument("--foggy-city-dir", required=True)
    parser.add_argument("--foggy-subset", default="Medium_Fog", choices=["Dense_Fog", "Medium_Fog", "No_Fog"])
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-domain", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    interpretation = (
        "Foggy Cityscapes is closer to restoration benchmark haze than DAWN-Fog because its fog distribution is "
        "more controlled and urban daytime structure remains visible. When its radial FFT profile and edge maps "
        "are closer to SOTS, AdaIR can reuse its learned dehazing behavior more effectively. This supports the "
        "slide observation that Foggy Cityscapes restores better while DAWN-Fog fails due to stronger real-domain shift."
    )
    analyzer = DomainShiftAnalyzer(
        domains=[
            DomainSpec("SOTS", Path(args.sots_dir)),
            DomainSpec("Foggy Cityscapes", Path(args.foggy_city_dir) / args.foggy_subset),
        ],
        config=DomainShiftConfig(
            output_dir=Path(args.output_dir),
            samples_per_domain=args.samples_per_domain,
            image_size=args.image_size,
            seed=args.seed,
            title=f"SOTS Haze vs Foggy Cityscapes ({args.foggy_subset}) Similarity",
            report_name="foggycity_vs_sots_similarity_report.md",
            interpretation=interpretation,
        ),
    )
    print(analyzer.run())


if __name__ == "__main__":
    main()

