from torch import Tensor
import torch
from torch import nn
import torch.nn.functional as F
from typing import *

from torchmetrics.functional.audio.sdr import signal_distortion_ratio as sdr
from torchmetrics.functional.audio import (
    scale_invariant_signal_distortion_ratio as si_sdr,
    signal_noise_ratio as snr,
)

# ============================================================
# Basic losses (existing, unchanged)
# ============================================================

def y_mse(preds, target):
    return F.mse_loss(preds, target, reduction="mean")


def stft_mse(preds, target):
    """Complex STFT MSE"""
    return F.mse_loss(preds.real, target.real) + F.mse_loss(preds.imag, target.imag)


def mag_mse(preds, target):
    """Magnitude MSE from complex STFT"""
    pred_mag = torch.sqrt(preds.real**2 + preds.imag**2 + 1e-12)
    tgt_mag  = torch.sqrt(target.real**2 + target.imag**2 + 1e-12)
    return F.mse_loss(pred_mag, tgt_mag, reduction="mean")


def neg_sdr(preds, target):
    batch_size = target.shape[0]
    x = sdr(preds=preds, target=target)
    return -torch.mean(x.view(batch_size, -1), dim=1)


def neg_si_sdr(preds, target):
    batch_size = target.shape[0]
    x = si_sdr(preds=preds, target=target)
    return -torch.mean(x.view(batch_size, -1), dim=1)


def neg_snr(preds, target):
    batch_size = target.shape[0]
    x = snr(preds=preds, target=target)
    return -torch.mean(x.view(batch_size, -1), dim=1)


# ============================================================
# SpatialNet-friendly losses (NEW)
# ============================================================

def phase_loss(preds, target):
    """
    Lightweight phase loss for SpatialNet
    """
    pred_phase = torch.atan2(preds.imag, preds.real)
    tgt_phase  = torch.atan2(target.imag, target.real)

    # Instantaneous phase (wrapped)
    ip = torch.angle(torch.exp(1j * (pred_phase - tgt_phase)))
    ip_loss = torch.mean(torch.abs(ip))

    # Group delay (frequency derivative)
    gd_pred = torch.diff(pred_phase, dim=-1)
    gd_tgt  = torch.diff(tgt_phase, dim=-1)
    gd_loss = torch.mean(torch.abs(gd_pred - gd_tgt))

    return ip_loss + gd_loss

# ============================================================
# Loss Manager
# ============================================================

class LossManager(nn.Module):
    """
    pred_time : time-domain waveform (B, 1, T) or (B, T)
    pred_stft : complex STFT (B, T, F) or (B, F, T)
    tgt_time  : reference waveform
    tgt_stft  : reference STFT
    """

    def __init__(self, type: List[str], weight: List[float]):
        super().__init__()
        self.loss_names = type
        self.loss_weights = [float(w) for w in weight]

        # loss_name → (function, domain)
        self.loss_dict = {
            # time-domain
            "y_mse":        (y_mse, "time"),
            "neg_sdr":      (neg_sdr, "time"),
            "neg_si_sdr":   (neg_si_sdr, "time"),
            "neg_snr":      (neg_snr, "time"),

            # frequency / STFT
            "stft_mse":     (stft_mse, "stft"),
            "mag_mse":      (mag_mse, "stft"),

            # SpatialNet-recommended
            "phase_loss":   (phase_loss, "stft"),
            "complex_mse":  (complex_mse, "stft"),
            "stft_consistency": (stft_consistency, "stft_pair"),
        }

    def __call__(
        self,
        pred_time: Tensor,
        pred_stft: Tensor,
        tgt_time: Tensor,
        tgt_stft: Tensor,
        **kwargs,
    ):
        """
        Optional kwargs:
          pred_stft_recons : required if using stft_consistency
        """

        assert pred_time.shape == tgt_time.shape
        assert pred_stft.shape == tgt_stft.shape

        total_loss = 0.0
        items = {}

        for name, w in zip(self.loss_names, self.loss_weights):
            if name not in self.loss_dict:
                raise KeyError(f"Unknown loss: {name}")

            loss_fn, domain = self.loss_dict[name]

            if domain == "time":
                val = loss_fn(pred_time, tgt_time)

            elif domain == "stft":
                val = loss_fn(pred_stft, tgt_stft)

            elif domain == "stft_pair":
                if "pred_stft_recons" not in kwargs:
                    raise ValueError(
                        "stft_consistency requires pred_stft_recons"
                    )
                val = loss_fn(pred_stft, kwargs["pred_stft_recons"])

            else:
                raise RuntimeError(f"Unsupported loss domain: {domain}")

            val = val.mean()
            items[name] = val
            total_loss += w * val

        return total_loss, items
