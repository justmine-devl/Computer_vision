import argparse
import csv
import json
import math
import random
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
SRC_DIR = ROOT / "src"
DL_NETS_DIR = ROOT / "dl_nets"
sys.path.insert(0, str(SRC_DIR))
sys.path.insert(0, str(DL_NETS_DIR))

import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader


@dataclass
class TrainConfig:
    adair_repo: str
    output_dir: str
    pretrained_ckpt: str = ""
    resume: str = ""
    epochs: int = 150
    batch_size: int = 8
    lr: float = 2e-4
    weight_decay: float = 1e-4
    patch_size: int = 128
    num_workers: int = 16
    de_type: tuple[str, ...] = (
        "denoise_15",
        "denoise_25",
        "denoise_50",
        "derain",
        "dehaze",
        "deblur",
        "enhance",
    )
    data_file_dir: str = "data_dir/"
    denoise_dir: str = "data/Train/Denoise/"
    gopro_dir: str = "data/Train/Deblur/"
    enhance_dir: str = "data/Train/Enhance/"
    derain_dir: str = "data/Train/Derain/"
    dehaze_dir: str = "data/Train/Dehaze/"
    trainable: str = "full"
    max_minutes: float = 0.0
    max_train_steps_per_epoch: int = 0
    grad_clip: float = 1.0
    amp: bool = False
    seed: int = 42
    log_interval: int = 50


def seed_everything(seed: int) -> None:
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def ensure_trailing_sep(path: str) -> str:
    if not path:
        return path
    return path if path.endswith(("/", "\\")) else path + "/"


def import_adair_components(adair_repo: Path):
    sys.path.insert(0, str(adair_repo))
    from net.model import AdaIR
    from utils.dataset_utils import AdaIRTrainDataset
    from utils.schedulers import LinearWarmupCosineAnnealingLR

    return AdaIR, AdaIRTrainDataset, LinearWarmupCosineAnnealingLR


def make_dataset_options(cfg: TrainConfig) -> argparse.Namespace:
    return argparse.Namespace(
        epochs=cfg.epochs,
        batch_size=cfg.batch_size,
        lr=cfg.lr,
        de_type=list(cfg.de_type),
        patch_size=cfg.patch_size,
        num_workers=cfg.num_workers,
        data_file_dir=ensure_trailing_sep(cfg.data_file_dir),
        denoise_dir=ensure_trailing_sep(cfg.denoise_dir),
        gopro_dir=ensure_trailing_sep(cfg.gopro_dir),
        enhance_dir=ensure_trailing_sep(cfg.enhance_dir),
        derain_dir=ensure_trailing_sep(cfg.derain_dir),
        dehaze_dir=ensure_trailing_sep(cfg.dehaze_dir),
    )


def load_model_state(model: nn.Module, ckpt_path: Path) -> None:
    ckpt = torch.load(ckpt_path, map_location="cpu")
    state = ckpt.get("state_dict", ckpt.get("model", ckpt))
    cleaned = {}
    for key, value in state.items():
        if key.startswith("net."):
            key = key[4:]
        cleaned[key] = value
    missing, unexpected = model.load_state_dict(cleaned, strict=False)
    if missing or unexpected:
        print(f"load_state_dict missing={len(missing)} unexpected={len(unexpected)}")
        if missing:
            print("missing sample:", missing[:10])
        if unexpected:
            print("unexpected sample:", unexpected[:10])


def set_trainable(model: nn.Module, mode: str) -> list[str]:
    if mode == "full":
        for p in model.parameters():
            p.requires_grad = True
        return [name for name, _ in model.named_parameters()]

    for p in model.parameters():
        p.requires_grad = False

    if mode == "frequency":
        prefixes = ("fre1.", "fre2.", "fre3.")
    elif mode == "decoder_frequency":
        prefixes = (
            "fre1.",
            "fre2.",
            "fre3.",
            "latent.",
            "decoder_level3.",
            "decoder_level2.",
            "decoder_level1.",
            "refinement.",
            "output.",
        )
    else:
        raise ValueError("trainable must be one of: full, frequency, decoder_frequency")

    opened = []
    for name, p in model.named_parameters():
        if name.startswith(prefixes):
            p.requires_grad = True
            opened.append(name)
    return opened


def parameter_counts(model: nn.Module) -> tuple[int, int]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable


def read_history(path: Path) -> list[dict[str, float]]:
    if not path.exists():
        return []
    rows = []
    with path.open("r", newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            rows.append(
                {
                    "epoch": int(row["epoch"]),
                    "train_loss": float(row["train_loss"]),
                    "lr": float(row["lr"]),
                    "elapsed_minutes": float(row["elapsed_minutes"]),
                }
            )
    return rows


def write_history(path: Path, rows: list[dict[str, float]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    rows = sorted(rows, key=lambda r: int(r["epoch"]))
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "lr", "elapsed_minutes"])
        writer.writeheader()
        writer.writerows(rows)


def append_or_replace_history(path: Path, new_row: dict[str, float]) -> list[dict[str, float]]:
    rows = [r for r in read_history(path) if int(r["epoch"]) != int(new_row["epoch"])]
    rows.append(new_row)
    write_history(path, rows)
    return rows


def plot_training_curve(history: list[dict[str, float]], out_path: Path) -> None:
    if not history:
        return
    history = sorted(history, key=lambda r: int(r["epoch"]))
    epochs = [int(r["epoch"]) for r in history]
    losses = [float(r["train_loss"]) for r in history]
    lrs = [float(r["lr"]) for r in history]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(epochs, losses, marker="o", linewidth=1.8)
    axes[0].set_title("Training Loss")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("L1 loss")
    axes[0].grid(True, alpha=0.3)

    axes[1].plot(epochs, lrs, marker="o", color="tab:orange", linewidth=1.8)
    axes[1].set_title("Learning Rate")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("LR")
    axes[1].grid(True, alpha=0.3)
    axes[1].ticklabel_format(axis="y", style="sci", scilimits=(0, 0))

    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=180)
    plt.close(fig)


def save_checkpoint(path: Path, model: nn.Module, optimizer, scheduler, scaler, epoch: int, cfg: TrainConfig) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "epoch": epoch,
            "model": model.state_dict(),
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict() if scheduler is not None else None,
            "scaler": scaler.state_dict() if scaler is not None else None,
            "config": asdict(cfg),
        },
        path,
    )


def parse_args() -> TrainConfig:
    parser = argparse.ArgumentParser(
        description="AdaIR original-setting training with CSV logging and training-curve plotting."
    )
    parser.add_argument("--adair-root", "--adair-repo", dest="adair_repo", default=str(ROOT / "dl_nets" / "AdaIR"), help="Path to AdaIR network code containing net/ and utils/.")
    parser.add_argument("--output-dir", required=True, help="Writable experiment output directory.")
    parser.add_argument("--pretrained-ckpt", default="", help="Optional checkpoint to initialize model weights.")
    parser.add_argument("--resume", default="", help="Resume from last.ckpt and continue the same curve/log.")

    parser.add_argument("--epochs", type=int, default=150, help="Total target epochs. Resume continues until this count.")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2e-4)
    parser.add_argument("--weight-decay", type=float, default=1e-4)
    parser.add_argument("--patch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=16)
    parser.add_argument(
        "--de-type",
        nargs="+",
        default=["denoise_15", "denoise_25", "denoise_50", "derain", "dehaze", "deblur", "enhance"],
        help="Same task names as AdaIR/options.py.",
    )

    parser.add_argument("--data-file-dir", default="data_dir/")
    parser.add_argument("--denoise-dir", default="data/Train/Denoise/")
    parser.add_argument("--gopro-dir", default="data/Train/Deblur/")
    parser.add_argument("--enhance-dir", default="data/Train/Enhance/")
    parser.add_argument("--derain-dir", default="data/Train/Derain/")
    parser.add_argument("--dehaze-dir", default="data/Train/Dehaze/")

    parser.add_argument("--trainable", default="full", choices=["full", "frequency", "decoder_frequency"])
    parser.add_argument("--max-minutes", type=float, default=0.0, help="0 means no wall-clock limit.")
    parser.add_argument("--max-train-steps-per-epoch", type=int, default=0, help="0 means full epoch.")
    parser.add_argument("--grad-clip", type=float, default=1.0)
    parser.add_argument("--amp", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--log-interval", type=int, default=50)
    args = parser.parse_args()

    return TrainConfig(
        adair_repo=args.adair_repo,
        output_dir=args.output_dir,
        pretrained_ckpt=args.pretrained_ckpt,
        resume=args.resume,
        epochs=args.epochs,
        batch_size=args.batch_size,
        lr=args.lr,
        weight_decay=args.weight_decay,
        patch_size=args.patch_size,
        num_workers=args.num_workers,
        de_type=tuple(args.de_type),
        data_file_dir=args.data_file_dir,
        denoise_dir=args.denoise_dir,
        gopro_dir=args.gopro_dir,
        enhance_dir=args.enhance_dir,
        derain_dir=args.derain_dir,
        dehaze_dir=args.dehaze_dir,
        trainable=args.trainable,
        max_minutes=args.max_minutes,
        max_train_steps_per_epoch=args.max_train_steps_per_epoch,
        grad_clip=args.grad_clip,
        amp=args.amp,
        seed=args.seed,
        log_interval=args.log_interval,
    )


def main() -> None:
    cfg = parse_args()
    seed_everything(cfg.seed)

    out_dir = Path(cfg.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "config.json").write_text(json.dumps(asdict(cfg), indent=2), encoding="utf-8")

    AdaIR, AdaIRTrainDataset, LinearWarmupCosineAnnealingLR = import_adair_components(Path(cfg.adair_repo).resolve())
    dataset_opt = make_dataset_options(cfg)
    trainset = AdaIRTrainDataset(dataset_opt)
    trainloader = DataLoader(
        trainset,
        batch_size=cfg.batch_size,
        pin_memory=True,
        shuffle=True,
        drop_last=True,
        num_workers=cfg.num_workers,
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = AdaIR(decoder=True)
    if cfg.pretrained_ckpt:
        load_model_state(model, Path(cfg.pretrained_ckpt))
    opened = set_trainable(model, cfg.trainable)
    model.to(device)

    total_params, trainable_params = parameter_counts(model)
    (out_dir / "trainable_layers.txt").write_text("\n".join(opened) + "\n", encoding="utf-8")
    print(f"device={device}")
    if device.type == "cuda":
        print(f"gpu={torch.cuda.get_device_name(0)}")
    print(f"dataset_size={len(trainset)}")
    print(f"tasks={list(cfg.de_type)}")
    print(f"params total={total_params:,} trainable={trainable_params:,} ({trainable_params / total_params:.2%})")

    optimizer = optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=cfg.lr,
        weight_decay=cfg.weight_decay,
    )
    scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=15, max_epochs=180)
    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.type == "cuda")
    loss_fn = nn.L1Loss()

    history_path = out_dir / "training_log.csv"
    curve_path = out_dir / "training_curve.png"
    start_epoch = 0
    if cfg.resume:
        resume = torch.load(cfg.resume, map_location="cpu")
        model.load_state_dict(resume["model"], strict=True)
        optimizer.load_state_dict(resume["optimizer"])
        if resume.get("scheduler"):
            scheduler.load_state_dict(resume["scheduler"])
        if resume.get("scaler") and scaler is not None:
            scaler.load_state_dict(resume["scaler"])
        start_epoch = int(resume["epoch"]) + 1
    elif history_path.exists():
        history_path.unlink()

    print(f"start_epoch={start_epoch} target_epochs={cfg.epochs}")
    start_time = time.time()

    for epoch in range(start_epoch, cfg.epochs):
        model.train()
        running = 0.0
        steps = 0
        for batch in trainloader:
            if cfg.max_train_steps_per_epoch > 0 and steps >= cfg.max_train_steps_per_epoch:
                break
            if cfg.max_minutes > 0 and (time.time() - start_time) / 60.0 >= cfg.max_minutes:
                print("Reached max_minutes; saving last checkpoint and current curve.")
                save_checkpoint(out_dir / "last.ckpt", model, optimizer, scheduler, scaler, max(epoch - 1, 0), cfg)
                plot_training_curve(read_history(history_path), curve_path)
                return

            (_, _), degrad_patch, clean_patch = batch
            degrad_patch = degrad_patch.to(device, non_blocking=True)
            clean_patch = clean_patch.to(device, non_blocking=True)

            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=cfg.amp and device.type == "cuda"):
                restored = model(degrad_patch)
                loss = loss_fn(restored, clean_patch)

            scaler.scale(loss).backward()
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

            running += float(loss.detach().cpu())
            steps += 1
            if cfg.log_interval > 0 and steps % cfg.log_interval == 0:
                print(f"epoch={epoch:03d} step={steps:05d}/{len(trainloader):05d} loss={running / steps:.6f}")

        scheduler.step(epoch)
        elapsed = (time.time() - start_time) / 60.0
        train_loss = running / max(steps, 1)
        current_lr = optimizer.param_groups[0]["lr"]
        history = append_or_replace_history(
            history_path,
            {
                "epoch": epoch,
                "train_loss": train_loss,
                "lr": current_lr,
                "elapsed_minutes": elapsed,
            },
        )
        plot_training_curve(history, curve_path)
        save_checkpoint(out_dir / "last.ckpt", model, optimizer, scheduler, scaler, epoch, cfg)
        print(f"epoch={epoch:03d} train_loss={train_loss:.6f} lr={current_lr:.8g} elapsed_min={elapsed:.1f}")

    plot_training_curve(read_history(history_path), curve_path)


if __name__ == "__main__":
    main()
