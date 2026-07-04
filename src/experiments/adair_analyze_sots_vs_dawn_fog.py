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
    parser = argparse.ArgumentParser(description="Compare synthetic SOTS haze against real DAWN-Fog in frequency/edge space.")
    parser.add_argument("--sots-dir", required=True)
    parser.add_argument("--dawn-fog-dir", required=True)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--samples-per-domain", type=int, default=8)
    parser.add_argument("--image-size", type=int, default=384)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    interpretation = (
        "SOTS haze is a synthetic restoration benchmark domain with relatively structured atmospheric haze. "
        "DAWN-Fog is real traffic fog, where visibility is mixed with headlights, road structures, exposure, "
        "compression and object-scale occlusion. If SOTS keeps stronger mid/high-frequency texture responses "
        "while DAWN-Fog suppresses or distorts them, AdaIR's AFLB/frequency modules may not receive the same "
        "degradation cues learned from SOTS. This explains why DAWN-Fog outputs can remain nearly unchanged."
    )
    analyzer = DomainShiftAnalyzer(
        domains=[
            DomainSpec("SOTS", Path(args.sots_dir)),
            DomainSpec("DAWN-Fog", Path(args.dawn_fog_dir)),
        ],
        config=DomainShiftConfig(
            output_dir=Path(args.output_dir),
            samples_per_domain=args.samples_per_domain,
            image_size=args.image_size,
            seed=args.seed,
            title="SOTS Haze vs DAWN-Fog Frequency-Domain Shift",
            report_name="sots_vs_dawn_fog_frequency_report.md",
            interpretation=interpretation,
        ),
    )
    print(analyzer.run())


if __name__ == "__main__":
    main()

