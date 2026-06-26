from torchmetrics.functional.audio.snr import signal_noise_ratio as snr
from torchmetrics.functional.audio import scale_invariant_signal_distortion_ratio as si_sdr
import numpy as np
import torch
from pesq import pesq

def calculate_snr(mixed_signal, clean_signal):
    if not isinstance(mixed_signal, torch.Tensor):
        mixed_signal = torch.from_numpy(np.asarray(mixed_signal))
    if not isinstance(clean_signal, torch.Tensor):
        clean_signal = torch.from_numpy(np.asarray(clean_signal))
    
    per_snr = snr(preds=mixed_signal, target=clean_signal)
    return per_snr

def calculate_pesq(mixed_signal, clean_signal, sr):
    mixed_signal = mixed_signal.squeeze().cpu().numpy()
    clean_signal = clean_signal.squeeze().cpu().numpy()
    pesq_score = pesq(sr, clean_signal, mixed_signal, 'wb')
    return pesq_score

def cal_SISNR(ref_sig, out_sig, eps=1e-8):
    """Calcuate Scale-Invariant Source-to-Noise Ratio (SI-SNR)
    Args:
        ref_sig: numpy.ndarray, [T]
        out_sig: numpy.ndarray, [T]
    Returns:
        SISNR
    """
    assert len(ref_sig) == len(out_sig)
    ref_sig = ref_sig - np.mean(ref_sig)
    out_sig = out_sig - np.mean(out_sig)
    ref_energy = np.sum(ref_sig ** 2) + eps
    proj = np.sum(ref_sig * out_sig) * ref_sig / ref_energy
    noise = out_sig - proj
    ratio = np.sum(proj ** 2) / (np.sum(noise ** 2) + eps)
    sisnr = 10 * np.log(ratio + eps) / np.log(10.0)
    return sisnr