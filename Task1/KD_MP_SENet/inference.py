from __future__ import absolute_import, division, print_function, unicode_literals
import sys
sys.path.append("..")
import glob
import os
import argparse
import json
from re import S
import torch
import librosa
from env import AttrDict
from dataset import mag_pha_stft, mag_pha_istft
from models.model import MPNet
import soundfile as sf
from rich.progress import track

h = None
device = None
def calc_complexity(model, h):
    try:
        from thop import profile
    except ImportError:
        print("Error: 'thop' library is not installed. Please run 'pip install thop'")
        return

    print("\n--- Calculating Model Complexity ---")
    
    # 1. 构造 dummy 输入
    # 我们假设输入时长为 1 秒，计算对应的频点数和帧数
    sr = h.sampling_rate
    n_fft = h.n_fft
    hop_size = h.hop_size
    
    # 计算 STFT后的维度
    n_freq = n_fft // 2 + 1
    n_frames = int(sr / hop_size) + 1
    
    # 获取模型参数所在的设备和数据类型
    param = next(model.parameters())
    device = param.device
    dtype = param.dtype
    
    # 构造随机输入张量
    # 注意：根据你的 mag_pha_stft 用法，输入通常是 [Batch, Freq, Time] 或 [Batch, Channel, Freq, Time]
    # 如果报错维度不匹配，请尝试在中间添加 .unsqueeze(1) 变为 [1, 1, n_freq, n_frames]
    dummy_amp = torch.randn(1, n_freq, n_frames, device=device, dtype=dtype)
    dummy_pha = torch.randn(1, n_freq, n_frames, device=device, dtype=dtype)
    
    # 2. 使用 thop 进行分析
    # custom_ops 可以处理一些 thop 默认不支持的算子，如果报错可以查阅 thop 文档
    macs, params = profile(model, inputs=(dummy_amp, dummy_pha), verbose=False)
    
    # 3. 打印结果
    print(f"Input Shape (1 sec audio): {dummy_amp.shape}")
    print(f"MACs (Computational Cost): {macs / 1e9:.2f} G (Giga MACs)")
    print(f"Params (Model Size): {params / 1e6:.2f} M (Million)")
    print("------------------------------------\n")
def count_parameters(model):
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return total, trainable
def load_checkpoint(filepath, device):
    assert os.path.isfile(filepath)
    print("Loading '{}'".format(filepath))
    checkpoint_dict = torch.load(filepath, map_location=device)
    print("Complete.")
    return checkpoint_dict

def scan_checkpoint(cp_dir, prefix):
    pattern = os.path.join(cp_dir, prefix + '*')
    cp_list = glob.glob(pattern)
    if len(cp_list) == 0:
        return ''
    return sorted(cp_list)[-1]

def inference(a):
    model = MPNet(h).to(device)
    calc_complexity(model, h)
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Model Total Params: {total_params:,} ({total_params/1e6:.2f} M)")
    print(f"Model Trainable Params: {trainable_params:,} ({trainable_params/1e6:.2f} M)")
    state_dict = load_checkpoint(a.checkpoint_file, device)
    model.load_state_dict(state_dict['generator'])

    test_indexes = os.listdir(a.input_noisy_wavs_dir)

    os.makedirs(a.output_dir, exist_ok=True)

    model.eval()

    with torch.no_grad():
        for index in track(test_indexes):
            noisy_wav, _ = librosa.load(os.path.join(a.input_noisy_wavs_dir, index), sr=h.sampling_rate)
            noisy_wav = torch.FloatTensor(noisy_wav).to(device)
            norm_factor = torch.sqrt(len(noisy_wav) / torch.sum(noisy_wav ** 2.0)).to(device)
            noisy_wav = (noisy_wav * norm_factor).unsqueeze(0)
            noisy_amp, noisy_pha, noisy_com = mag_pha_stft(noisy_wav, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            amp_g, pha_g, com_g = model(noisy_amp, noisy_pha)
            audio_g = mag_pha_istft(amp_g, pha_g, h.n_fft, h.hop_size, h.win_size, h.compress_factor)
            audio_g = audio_g / norm_factor

            output_file = os.path.join(a.output_dir, index)

            sf.write(output_file, audio_g.squeeze().cpu().numpy(), h.sampling_rate, 'PCM_16')


def main():
    print('Initializing Inference Process..')

    parser = argparse.ArgumentParser()
    parser.add_argument('--input_noisy_wavs_dir', default='/home/iasp_guest1/tyx/spatialnet/fullembeddeddeeperchangedloss/ckpt/output0201')
    # parser.add_argument('--input_noisy_wavs_dir', default='/home/iasp_guest1/tzx/code/MP-SENet/dataset/testre/vad')
    parser.add_argument('--output_dir', default='/home/iasp_guest1/tyx/MP_SENet_tzx/outputtsk2')
    # parser.add_argument('--output_dir', default='/home/iasp_guest1/tzx/code/MP-SENet/dataset/testre/outputvad')
    parser.add_argument('--checkpoint_file',default='/home/iasp_guest1/tyx/MP_SENet_tzx/ckpt203/g_00492000')
    a = parser.parse_args()

    config_file = os.path.join(os.path.split(a.checkpoint_file)[0], 'config1.json')
    with open(config_file) as f:
        data = f.read()

    global h
    json_config = json.loads(data)
    h = AttrDict(json_config)

    torch.manual_seed(h.seed)
    global device
    if torch.cuda.is_available():
        torch.cuda.manual_seed(h.seed)
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    inference(a)


if __name__ == '__main__':
    main()
