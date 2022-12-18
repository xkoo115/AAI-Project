import torch

from glob import glob
from dataloader import *
from torch.fft import fft
import numpy as np
import matplotlib.pyplot as plt

# noise_data = AudioNoise()
# for i in noise_data:
#     #print(i.shape)
#     x = range(400000)
#     plt.plot(x, i.squeeze())
#     plt.show()
#     break

# test_noise_dir = "LibriSpeech-SI/test-noisy/*.flac"
# test_noise = AudioFlac(test_noise_dir)
# for i in test_noise:
#     x = range(400000)
#     plt.plot(x, i.squeeze())
#     plt.show()
#     break

# spk_data = AudioSPK()
# print(len(spk_data))
#
# spk_shape = []
# for i, _ in spk_data:
#     # spk_shape.append(i.shape[1])
#     # print(_)
#     x = range(400000)
#     plt.plot(x, i.squeeze())
#     plt.show()
#     break

def test():
    test_path = "LibriSpeech-SI/test/*.flac"
    noisy_path = "LibriSpeech-SI/test-noisy/*.flac"

    test_set = AudioFlac(test_path)
    noisy_set = AudioFlac(noisy_path)

    test = do_fft(test_set)
    noisy = do_fft(noisy_set)

    model = torch.load("Recogniser.pt")

    out_test = predict(test, model.cpu())
    out_test_noisy = predict(noisy, model.cpu())

    np.savetxt("result_test.txt", out_test, fmt='>.3d')
    np.savetxt("result_test_noisy.txt", out_test_noisy, fmt='>.3d')

def do_fft(x):
    out = []
    for i in x:
        temp = torch.fft.fft(i, 400000).real
        temp = temp[:,:200000]
        out.append(temp)
    return out

def predict(x, model):
    out = []
    for i in x:
        temp = i.unsqueeze(dim=0)
        label = model(temp)
        _, pred = torch.max(label.data, dim=2)
        out.append(pred.item()+1)
    return np.array(out)

if __name__ == '__main__':
    test()