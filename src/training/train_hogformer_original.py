import os
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import time
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import DataLoader
from pathlib import Path
import sys

# Path prepending to find local utilities
ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "dl_nets" / "HOGformer"))

from utils.dataset_utils import AdaIRTrainDataset
from net.model import HOGformer as AdaIR
from utils.schedulers import LinearWarmupCosineAnnealingLR
from utils.options import options as opt

# Conditional imports for Foggy Cityscapes
from utils.foggy_dataset import build_foggy_datasets
from skimage.metrics import peak_signal_noise_ratio, structural_similarity

try:
    import wandb
    _HAS_WANDB = True
except ImportError:
    _HAS_WANDB = False

import lightning.pytorch as pl
from lightning.pytorch.loggers import WandbLogger, TensorBoardLogger
from lightning.pytorch.callbacks import ModelCheckpoint

try:
    import matplotlib
    matplotlib.use('Agg')  # Non-interactive backend
    import matplotlib.pyplot as plt
    _HAS_MATPLOTLIB = True
except ImportError:
    _HAS_MATPLOTLIB = False


class HOGLayer(nn.Module):
    def __init__(self, nbins=9, cell_size=8, block_size=2, signed_gradient=False, eps=1e-8):
        super(HOGLayer, self).__init__()
        self.nbins = nbins
        self.cell_size = cell_size
        self.block_size = block_size
        self.signed_gradient = signed_gradient
        self.eps = eps
        if not self.signed_gradient:
            angles = torch.tensor([(i * np.pi / self.nbins) for i in range(self.nbins)])
            self.bin_width = np.pi / self.nbins
        else:
            angles = torch.tensor([(i * 2 * np.pi / self.nbins) for i in range(self.nbins)])
            self.bin_width = 2 * np.pi / self.nbins
        self.register_buffer('angles', angles.view(1, -1, 1, 1))
        self.register_buffer('dx_filter', torch.tensor([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]]).float().view(1, 1, 3, 3))
        self.register_buffer('dy_filter', torch.tensor([[-1, -2, -1], [0, 0, 0], [1, 2, 1]]).float().view(1, 1, 3, 3))

    def forward(self, x):
        batch_size, channels, height, width = x.size()
        if channels == 3:
            x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
            x_gray = x_gray.unsqueeze(1)
        else:
            x_gray = x

        dx = F.conv2d(x_gray, self.dx_filter, padding=1)
        dy = F.conv2d(x_gray, self.dy_filter, padding=1)
        magnitude = torch.sqrt(dx**2 + dy**2 + self.eps)
        orientation = torch.atan2(dy, dx + self.eps)

        if not self.signed_gradient:
            orientation = torch.abs(orientation)

        delta = torch.abs(orientation - self.angles)
        if self.signed_gradient:
            delta = torch.min(delta, 2 * np.pi - delta)
        else:
            delta = torch.min(delta, np.pi - delta)
        weights = torch.relu(1.0 - delta / self.bin_width)
        new_height = (height // self.cell_size) * self.cell_size
        new_width = (width // self.cell_size) * self.cell_size
        if height % self.cell_size != 0 or width % self.cell_size != 0:
            magnitude = magnitude[:, :, :new_height, :new_width]
            weights = weights[:, :, :new_height, :new_width]

        weighted_magnitude = weights * magnitude
        hist = F.avg_pool2d(weighted_magnitude, kernel_size=self.cell_size, stride=self.cell_size)

        if self.block_size > 1:
            B, C, Hc, Wc = hist.shape
            if Hc >= self.block_size and Wc >= self.block_size:
                blocks = F.unfold(hist, kernel_size=self.block_size, stride=1)
                blocks = blocks.permute(0, 2, 1).reshape(-1, C * self.block_size**2)
                block_norm = torch.norm(blocks, p=2, dim=1, keepdim=True)
                blocks = blocks / (block_norm + self.eps)
                num_blocks = (Hc - self.block_size + 1) * (Wc - self.block_size + 1)
                out_hist = blocks.reshape(B, num_blocks, -1).reshape(B, -1)
            else:
                out_hist = hist.reshape(B, -1)
        else:
            out_hist = hist.reshape(batch_size, -1)

        return out_hist


class HOGLoss(nn.Module):
    def __init__(self, nbins=9, cell_size=8, block_size=2, signed_gradient=False, loss_type='l1', eps=1e-8):
        super(HOGLoss, self).__init__()
        self.hog_layer = HOGLayer(nbins=nbins, cell_size=cell_size, block_size=block_size, signed_gradient=signed_gradient, eps=eps)
        self.loss_type = loss_type

    def forward(self, pred, target):
        hog_pred = self.hog_layer(pred)
        hog_target = self.hog_layer(target)
        if self.loss_type.lower() == 'l1':
            loss = F.l1_loss(hog_pred, hog_target)
        else:
            loss = F.mse_loss(hog_pred, hog_target)
        return loss


class HOGformerTrainingModule(pl.LightningModule):
    def __init__(self, opt):
        super().__init__()
        self.save_hyperparameters()
        self.opt = opt
        self.net = AdaIR()
        self.loss_fn = nn.L1Loss()
        self.cri_HOGloss = HOGLoss()

        # History structures for curve plotting
        self.history = {
            'epochs': [],
            'train_loss': [],
            'train_loss_l1': [],
            'train_loss_pearson': [],
            'train_loss_hog': []
        }
        self.epoch_metrics = []

        # Load pretrained checkpoint if fine-tuning
        if self.opt.dataset_type == 'foggy' and self.opt.pretrained_ckpt and os.path.exists(self.opt.pretrained_ckpt):
            self.load_pretrained_weights(self.opt.pretrained_ckpt)

        # Apply freezing strategies if requested
        self.apply_freezing_strategy()

    def load_pretrained_weights(self, path):
        print(f"Loading pretrained weights from: {path}")
        ckpt = torch.load(path, map_location='cpu')
        state = ckpt.get('state_dict', ckpt)
        new_state = {}
        for k, v in state.items():
            if k.startswith('net.'):
                new_state[k[4:]] = v
            else:
                new_state[k] = v
        missing, unexpected = self.net.load_state_dict(new_state, strict=False)
        if missing:
            print(f"Pretrained load: missing keys: {missing[:5]}...")
        if unexpected:
            print(f"Pretrained load: unexpected keys: {unexpected[:5]}...")

    def apply_freezing_strategy(self):
        if self.opt.dataset_type == 'foggy':
            if self.opt.freeze_encoder:
                encoder_modules = [
                    self.net.patch_embed,
                    self.net.encoder_level1,
                    self.net.down1_2,
                    self.net.encoder_level2,
                    self.net.down2_3,
                    self.net.encoder_level3,
                    self.net.down3_4,
                    self.net.skip_patch_embed1,
                    self.net.skip_patch_embed2,
                    self.net.skip_patch_embed3,
                    self.net.reduce_chan_level_1,
                    self.net.reduce_chan_level_2,
                    self.net.reduce_chan_level_3,
                ]
                for m in encoder_modules:
                    for p in m.parameters():
                        p.requires_grad = False
                print("Frozen HOGformer Encoder modules.")

            if self.opt.freeze_latent:
                for p in self.net.latent.parameters():
                    p.requires_grad = False
                print("Frozen HOGformer Latent bottleneck modules.")

    def pearson_correlation_loss(self, x1, x2):
        assert x1.shape == x2.shape
        b = x1.shape[0]
        dim = -1
        x1, x2 = x1.reshape(b, -1), x2.reshape(b, -1)
        x1_mean, x2_mean = x1.mean(dim=dim, keepdims=True), x2.mean(dim=dim, keepdims=True)
        numerator = ((x1 - x1_mean) * (x2 - x2_mean)).sum(dim=dim, keepdims=True)
        std1 = (x1 - x1_mean).pow(2).sum(dim=dim, keepdims=True).sqrt()
        std2 = (x2 - x2_mean).pow(2).sum(dim=dim, keepdims=True).sqrt()
        denominator = std1 * std2
        corr = numerator.div(denominator + 1e-6)
        pearson = (1. - corr) / 2.
        return pearson[~pearson.isnan() * ~pearson.isinf()].mean()

    def forward(self, x):
        return self.net(x)

    def training_step(self, batch, batch_idx):
        if self.opt.dataset_type == 'multi':
            ([clean_name, de_id], degrad_patch, clean_patch) = batch
        else:  # foggy
            ((_, _), degrad_patch, clean_patch) = batch

        restored = self.net(degrad_patch)
        l_l1 = self.loss_fn(restored, clean_patch)
        l_pear = self.pearson_correlation_loss(restored, clean_patch)

        # Skip HOG loss in the last 5 epochs
        if self.current_epoch < self.opt.epochs - 5:
            l_hog = self.cri_HOGloss(restored, clean_patch)
            loss = l_l1 + l_pear + l_hog
        else:
            l_hog = torch.tensor(0.0, device=restored.device)
            loss = l_l1 + l_pear

        # Log metrics
        self.log("train_loss", loss, on_epoch=True, prog_bar=True)
        self.log("train_loss_l1", l_l1, on_epoch=True)
        self.log("train_loss_pearson", l_pear, on_epoch=True)
        self.log("train_loss_hog", l_hog, on_epoch=True)

        self.epoch_metrics.append({
            'loss': loss.detach().item(),
            'l1': l_l1.detach().item(),
            'pear': l_pear.detach().item(),
            'hog': l_hog.detach().item()
        })

        return loss

    def validation_step(self, batch, batch_idx):
        if self.opt.dataset_type == 'foggy':
            (_, _), foggy, clean = batch
            restored = self.net(foggy)
            
            # Compute metric averages on the batch
            r = restored.detach().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
            c = clean.detach().cpu().clamp(0, 1).numpy().transpose(0, 2, 3, 1)
            psnr_val = peak_signal_noise_ratio(c[0], r[0], data_range=1)
            ssim_val = structural_similarity(c[0], r[0], data_range=1, channel_axis=-1)
            
            self.log("val_psnr", psnr_val, on_epoch=True, prog_bar=True)
            self.log("val_ssim", ssim_val, on_epoch=True, prog_bar=True)
            return {"val_psnr": psnr_val, "val_ssim": ssim_val}

    def on_train_epoch_end(self):
        if not self.epoch_metrics:
            return

        avg_loss = np.mean([m['loss'] for m in self.epoch_metrics])
        avg_l1   = np.mean([m['l1'] for m in self.epoch_metrics])
        avg_pear = np.mean([m['pear'] for m in self.epoch_metrics])
        avg_hog  = np.mean([m['hog'] for m in self.epoch_metrics])

        self.history['epochs'].append(self.current_epoch + 1)
        self.history['train_loss'].append(avg_loss)
        self.history['train_loss_l1'].append(avg_l1)
        self.history['train_loss_pearson'].append(avg_pear)
        self.history['train_loss_hog'].append(avg_hog)

        # Clear step accumulator
        self.epoch_metrics = []

        # Graph the training curve
        self.graph_training_curve()

    def graph_training_curve(self):
        if not _HAS_MATPLOTLIB:
            return
        try:
            epochs = self.history['epochs']
            plt.figure(figsize=(10, 6))
            plt.plot(epochs, self.history['train_loss'], label='Total Loss', color='#1f77b4', linewidth=2.5)
            plt.plot(epochs, self.history['train_loss_l1'], label='L1 Loss (Reconstruction)', color='#2ca02c', linestyle='--', linewidth=1.8)
            plt.plot(epochs, self.history['train_loss_pearson'], label='Pearson Correlation Loss', color='#ff7f0e', linestyle='-.', linewidth=1.8)
            plt.plot(epochs, self.history['train_loss_hog'], label='HOG Feature Loss', color='#d62728', linestyle=':', linewidth=1.8)
            
            plt.title('HOGformer Training Curves', fontsize=14, fontweight='bold', pad=15)
            plt.xlabel('Epoch', fontsize=12)
            plt.ylabel('Loss Value', fontsize=12)
            plt.grid(True, linestyle=':', alpha=0.6)
            plt.legend(fontsize=10, loc='upper right')
            
            plt.tight_layout()
            os.makedirs('logs', exist_ok=True)
            curve_path = os.path.join('logs', 'training_curves.png')
            plt.savefig(curve_path, dpi=300)
            plt.close()
            print(f"[Epoch {self.current_epoch + 1}] Training curve successfully updated at: {curve_path}")
        except Exception as e:
            print(f"Failed to generate training curve graph: {e}")

    def configure_optimizers(self):
        optimizer = optim.AdamW(filter(lambda p: p.requires_grad, self.parameters()), lr=self.opt.lr, weight_decay=self.opt.weight_decay)
        scheduler = LinearWarmupCosineAnnealingLR(optimizer=optimizer, warmup_epochs=self.opt.warmup_epochs, max_epochs=self.opt.epochs)
        return [optimizer], [scheduler]


def main():
    print("HOGformer Unified Options:")
    print(opt)

    # Logger Setup
    if opt.wblogger is not None and _HAS_WANDB:
        logger = WandbLogger(project=opt.wblogger, name=f"HOGformer-{opt.dataset_type}")
    else:
        logger = TensorBoardLogger(save_dir="logs/")

    # Select Checkpoint callbacks based on mode
    callbacks = []
    if opt.dataset_type == 'foggy':
        callbacks.append(ModelCheckpoint(dirpath=opt.ckpt_dir, monitor="val_psnr", mode="max", save_top_k=1, filename="best_psnr"))
        callbacks.append(ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1, filename="epoch_{epoch}"))
    else:
        callbacks.append(ModelCheckpoint(dirpath=opt.ckpt_dir, every_n_epochs=1, save_top_k=-1))

    # Dataset Selection
    if opt.dataset_type == 'multi':
        trainset = AdaIRTrainDataset(opt)
        trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                                 drop_last=True, num_workers=opt.num_workers)
        valloader = None
    else:  # foggy fine-tuning
        trainset, valset = build_foggy_datasets(
            data_root=opt.data_root,
            patch_size=opt.patch_size,
            beta_filter=opt.beta_filter if opt.beta_filter else None
        )
        trainloader = DataLoader(trainset, batch_size=opt.batch_size, pin_memory=True, shuffle=True,
                                 drop_last=True, num_workers=opt.num_workers)
        valloader = DataLoader(valset, batch_size=1, pin_memory=True, shuffle=False,
                               drop_last=False, num_workers=opt.num_workers)

    model = HOGformerTrainingModule(opt)

    trainer = pl.Trainer(
        max_epochs=opt.epochs,
        accelerator="gpu" if torch.cuda.is_available() else "cpu",
        devices=opt.num_gpus if torch.cuda.is_available() else 1,
        accumulate_grad_batches=opt.grad_accum_steps,
        logger=logger,
        callbacks=callbacks
    )

    if valloader is not None:
        trainer.fit(model=model, train_dataloaders=trainloader, val_dataloaders=valloader)
    else:
        trainer.fit(model=model, train_dataloaders=trainloader)


if __name__ == '__main__':
    main()
