import os
import sys
import json
import torch
import soundfile as sf

# ===================== Input/Output Configuration =====================
INPUT_WAV  = "mixture_signal.wav"
OUTPUT_WAV = "processed_signal.wav"

if len(sys.argv) >= 3:
    INPUT_WAV  = sys.argv[1]
    OUTPUT_WAV = sys.argv[2]
# ============================================================

GPU_ID = 0
FS = 16000  # Hard-coded sampling rate

# ---- SpatialNet-DG Project Config ----
SP_ROOT = "spatialnet_DG"
SP_CKPT = "spatialnet_DG/ckpt/checkpoints/best_model_task1.tar"

# ---- KD-MP-SENet Project Config ----
MP_ROOT = "KD_MP_SENet"
MP_CKPT = "KD_MP_SENet/ckpt/g_best_task1"
MP_CONFIG_FALLBACK = "MP_SENet/config1.json"

# ============================================================

def purge_modules(prefixes=("models", "utils", "dataset", "env")):
    """Removes loaded modules to prevent naming conflicts between projects."""
    for k in list(sys.modules.keys()):
        for p in prefixes:
            if k == p or k.startswith(p + "."):
                sys.modules.pop(k, None)
                break

def load_mp_config(ckpt_path: str):
    """Loads configuration file for MP-SENet."""
    ckpt_dir = os.path.dirname(ckpt_path)
    for cand in ["config1.json", "config.json"]:
        p = os.path.join(ckpt_dir, cand)
        if os.path.isfile(p):
            with open(p, "r", encoding="utf-8") as f:
                return json.loads(f.read())
    if MP_CONFIG_FALLBACK is not None:
        with open(MP_CONFIG_FALLBACK, "r", encoding="utf-8") as f:
            return json.loads(f.read())
    raise FileNotFoundError("Could not find MP-SENet config file.")

def main():
    # ===== Device Selection =====
    if torch.cuda.is_available():
        device = torch.device(f"cuda:{GPU_ID}")
        torch.cuda.set_device(device)
    else:
        device = torch.device("cpu")

    # ================= SpatialNet Logic =================
    sys.path.insert(0, SP_ROOT)
    from SpatialNet import SpatialNet as SpatialNetModel
    from utils.stft import STFT
    from utils.norm import Norm

    # Model parameters are hard-coded here
    sp_model = SpatialNetModel(
        dim_input=4,
        dim_output=2,
        num_layers=16,
        encoder_kernel_size=5,
        dim_hidden=128,
        dim_ffn=192,
        num_heads=4,
        dropout=[0, 0, 0],
        kernel_size=(5, 3),
        conv_groups=(8, 8),
        norms=("LN", "LN", "GN", "LN", "LN", "LN"),
        dim_squeeze=8,
        num_freqs=257,
        full_share=0,
    ).to(device)

    stft = STFT(512, 256, 512)
    norm = Norm("frequency", online=False)

    print(f"[SpatialNet] Loading checkpoint: {SP_CKPT}")
    sp_state = torch.load(SP_CKPT, map_location=device)
    sp_model.load_state_dict(sp_state["model"])
    sp_model.eval()

    purge_modules(("models", "utils"))
    sys.path.remove(SP_ROOT)

    # ================= MP-SENet Logic =================
    sys.path.insert(0, MP_ROOT)
    from env import AttrDict
    from dataset import mag_pha_stft, mag_pha_istft
    from models.model import MPNet

    mp_cfg = AttrDict(load_mp_config(MP_CKPT))
    mp_model = MPNet(mp_cfg).to(device)

    print(f"[MP-SENet] Loading checkpoint: {MP_CKPT}")
    mp_state = torch.load(MP_CKPT, map_location=device)
    mp_model.load_state_dict(mp_state["generator"])
    mp_model.eval()

    # ================= Load wav =================
    print(f"Reading Input: {INPUT_WAV}")
    wav, sr = sf.read(INPUT_WAV)
    if sr != FS:
        raise ValueError(f"Sampling rate mismatch: Input={sr}, Expected={FS}")

    if wav.ndim == 1:
        wav = wav[None, :]   # [1, T]
    elif wav.ndim == 2:
        wav = wav.T          # [C, T]
    else:
        raise ValueError("Unexpected wav dimensions")
    wav = torch.from_numpy(wav).float().unsqueeze(0)  # [1, C, T]

    # ================= SpatialNet Inference =================
    with torch.inference_mode():
        _, stft_paras = stft.stft(wav)
        noisy_f, _ = stft.stft(wav)   # [1, C, F, T]
        noisy_f, (Xr_n, XrMM_n) = norm.norm(noisy_f, ref_channel=0)
        noisy_f = noisy_f.permute(0, 2, 3, 1)  # [1, F, T, C]
        noisy_f = torch.view_as_real(noisy_f).reshape(
            1, noisy_f.shape[1], noisy_f.shape[2], -1
        ).to(device)

        enhanced_f = sp_model(noisy_f)
        if not torch.is_complex(enhanced_f):
            enhanced_f = torch.view_as_complex(
                enhanced_f.float().reshape(
                    1, enhanced_f.shape[1], enhanced_f.shape[2], -1
                )
            )
        enhanced_f = enhanced_f.unsqueeze(-1).permute(0, 3, 1, 2)  # [1,1,F,T]
        enhanced_f = norm.inorm(enhanced_f, (Xr_n.to(device), XrMM_n.to(device)))
        enhanced_wav = stft.istft(enhanced_f, stft_paras)  # [1,1,T]

    # ================= MP-SENet Inference =================
    wav = enhanced_wav[0, 0]  # [T]
    norm_factor = torch.sqrt(wav.numel() / torch.sum(wav ** 2.0 + 1e-12))
    wav_in = (wav * norm_factor).unsqueeze(0)

    noisy_amp, noisy_pha, _ = mag_pha_stft(
        wav_in,
        mp_cfg.n_fft,
        mp_cfg.hop_size,
        mp_cfg.win_size,
        mp_cfg.compress_factor,
    )

    with torch.no_grad():
        amp_g, pha_g, _ = mp_model(noisy_amp, noisy_pha)

    audio_g = mag_pha_istft(
        amp_g,
        pha_g,
        mp_cfg.n_fft,
        mp_cfg.hop_size,
        mp_cfg.win_size,
        mp_cfg.compress_factor,
    )
    audio_g = audio_g / norm_factor

    # ================= Save Processed Audio =================
    out = audio_g.squeeze().detach().cpu().numpy()
    
    out_dir = os.path.dirname(OUTPUT_WAV)
    if out_dir != "":
        os.makedirs(out_dir, exist_ok=True)
    sf.write(OUTPUT_WAV, out, mp_cfg.sampling_rate, "PCM_16")
    
    print("\nSingle sample inference completed.")
    print("Input  :", INPUT_WAV)
    print("Output :", OUTPUT_WAV)

if __name__ == "__main__":
    main()