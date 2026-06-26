from random import random
import soundfile as sf
import torch
import numpy as np
import random
import librosa

class SEDataset(torch.utils.data.Dataset):
    def __init__(
        self,
        clean_scp,
        noisy_scp,
        fs=16000,
        length=3,
        cut_audio=False
    ):

        self.file_pairs = self._load_scp_pairs(clean_scp, noisy_scp)

        self.fs = fs
        self.length = length
        self.L = int(length * fs)
        self.cut_audio = cut_audio
    
    def _load_scp_pairs(self, clean_scp, noisy_scp):
        with open(clean_scp, 'r') as f_clean, open(noisy_scp, 'r') as f_noisy:
            clean_lines = f_clean.readlines()
            noisy_lines = f_noisy.readlines()

        file_pairs = []
        for c_line, n_line in zip(clean_lines, noisy_lines):
            # 支持两种格式：有 utt_id 和没有 utt_id
            c_path = c_line.strip().split(maxsplit=1)[0]  # 如果只有路径，它就是 [0]，否则是 [1]
            n_path = n_line.strip().split(maxsplit=1)[0]
            file_pairs.append((c_path, n_path))

        return file_pairs
    
    def __len__(self):
        return len(self.file_pairs)

    
    def _read_audio(self, path):
        audio, fs = sf.read(path, dtype='float32')

        # resample
        if fs != self.fs:
            if audio.ndim == 1:  # mono
                audio = librosa.resample(audio, fs, self.fs).astype(np.float32)
            else:  # multi (T, C)
                audio = np.stack([
                    librosa.resample(audio[:, ch], fs, self.fs)
                    for ch in range(audio.shape[1])
                ], axis=0).astype(np.float32)  # shape (C, T)
                return audio  # 注意：多通道直接返回 (C, T)

        # --- 以下是 fs == self.fs 的情况 ---
        if audio.ndim == 1:
            # 单通道 shape=(T,)
            return audio
        else:
            # 多通道 (T, C) → (C, T)
            return audio.T


    def _crop_or_pad(self, audio):
        L = self.L

        if audio.ndim == 1:  # 单通道
            T = audio.shape[0]
            if T < L:
                return np.pad(audio, (0, L - T), mode="constant")
            else:
                start = random.randint(0, T - L)
                return audio[start:start+L]

        else:  # 多通道 (C,T)
            C, T = audio.shape
            if T < L:
                pad = np.zeros((C, L - T), dtype=np.float32)
                return np.concatenate([audio, pad], axis=1)
            else:
                start = random.randint(0, T - L)
                return audio[:, start:start+L]


    def __getitem__(self, idx):
        clean_path, noisy_path = self.file_pairs[idx]

        clean = self._read_audio(clean_path)
        noisy = self._read_audio(noisy_path)

        if self.cut_audio:
            clean = self._crop_or_pad(clean)
            noisy = self._crop_or_pad(noisy)

        file_name = clean_path.split('/')[-1]

        return torch.tensor(clean, dtype=torch.float32), torch.tensor(noisy, dtype=torch.float32), file_name

def collate_fn(batch):
    clean_list, noisy_list, names = zip(*batch)

    # 找最长长度
    max_len = max([c.shape[-1] if c.ndim == 2 else c.shape[0] for c in clean_list])

    def pad_tensor(t, max_len):
        if t.ndim == 1:  # (T,)
            return torch.nn.functional.pad(t, (0, max_len - t.size(0)))
        else:           # (C,T)
            return torch.nn.functional.pad(t, (0, max_len - t.size(1)))

    clean_padded = torch.stack([pad_tensor(c, max_len) for c in clean_list])
    noisy_padded = torch.stack([pad_tensor(n, max_len) for n in noisy_list])

    return clean_padded, noisy_padded, names


if __name__ == "__main__":
    import torch
    from torch.utils.data import DataLoader

    clean_scp = "/home/yujiezhu/code/SpatialNet/scp/test_clean.scp"
    noisy_scp = "/home/yujiezhu/code/SpatialNet/scp/test_noisy.scp"

    dataset = SEDataset(
        clean_scp=clean_scp,
        noisy_scp=noisy_scp,
        fs=16000,
        length=4,        # 4s
        cut_audio=True
    )

    loader = DataLoader(dataset, 
                        batch_size=2, 
                        shuffle=True,
                        collate_fn=collate_fn
                        )

    for noisy, clean, _ in loader:
        print("noisy shape:", noisy.shape)
        print("clean shape:", clean.shape)
        break
