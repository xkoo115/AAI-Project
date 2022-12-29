import numpy.fft
import torch
import torch.utils.data as data
import torchaudio
import numpy as np
from glob import glob


class AudioFlac(data.Dataset):
    def __init__(self, flac_dir):
        super(AudioFlac, self).__init__()
        self.flac_path = sorted(glob(flac_dir))

    def __getitem__(self, idx):
        flac_path = self.flac_path[idx]
        file_name = flac_path.split('/')[-1]
        flac, rate = torchaudio.load(flac_path)
        return flac, file_name

    def __len__(self):
        return len(self.flac_path)

class AudioNoise(data.Dataset):
    def __init__(self):
        super(AudioNoise, self).__init__()
        noise_dir = "LibriSpeech-SI/noise/*.wav"
        self.noise_path = glob(noise_dir)

    def __getitem__(self, idx):
        noise_path = self.noise_path[idx]
        noise, rate = torchaudio.load(noise_path)
        return noise

    def __len__(self):
        return len(self.noise_path)

class AudioSPK(data.Dataset):
    def __init__(self):
        super(AudioSPK, self).__init__()
        spk_dir = "LibriSpeech-SI/train/*/*.flac"
        self.spk_dir = sorted(glob(spk_dir))

    def __getitem__(self, idx):
        spk_path = self.spk_dir[idx]
        spk_name = spk_path.split('/')[-1].split('_')[0].split('k')[-1]
        spk_emb = np.zeros(250)
        spk_emb[int(spk_name)-1] = 1
        spk_emb = np.expand_dims(spk_emb, axis=0)
        spk_audio, rate = torchaudio.load(spk_path)
        return spk_audio, spk_name

    def __len__(self):
        return len(self.spk_dir)

class ReadData(data.Dataset):
    def __init__(self):
        super(ReadData, self).__init__()
        data_dir = "data/*/*.pt"
        self.data_dir = glob(data_dir)

    def __getitem__(self, idx):
        data_path = self.data_dir[idx]
        data_name = data_path.split('/')[-1].split('_')[0]
        data_emb = np.zeros(250)
        data_emb[int(data_name) - 1] = 1
        data_emb = np.expand_dims(data_emb, axis=0)
        data = torch.load(data_path)
        return data, data_emb

    def __len__(self):
        return len(self.data_dir)

# def get_noise(x):
#     out = [torch.zeros_like(x)]
#     noise_set = AudioNoise()
#     dataloader = data.DataLoader(noise_set, batch_size=1)
#     for i in dataloader:
#         temp = torch.repeat_interleave(i, repeats=x.shape[0], dim=0)
#         out.append(temp)
#     return out


