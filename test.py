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

    test, test_name = do_fft(test_set)
    noisy, noisy_name = do_fft(noisy_set)

    model = torch.load("Recogniser.pt")

    out_test = predict(test, model.cpu())
    out_test_noisy = predict(noisy, model.cpu())

    np.savetxt("result_test.txt", np.vstack((np.expand_dims(np.array(test_name), axis=0), np.expand_dims(out_test, axis=0))).T, fmt="%15s, %6s")
    np.savetxt("result_test_noisy.txt", np.vstack((np.expand_dims(np.array(noisy_name), axis=0), np.expand_dims(out_test_noisy, axis=0))).T, fmt="%15s, %6s")

def do_fft(x):
    out = []
    file_name = []
    for i, name in x:
        temp = torch.fft.fft(i, 400000).real
        temp = temp[:,:200000]
        out.append(temp)
        file_name.append(name)
    return out, file_name

def predict(x, model):
    out = []
    for i in x:
        temp = i.unsqueeze(dim=0)
        label = model(temp)
        _, pred = torch.max(label.data, dim=2)
        out.append(f"spk{pred.item()+1:0>3d}")
    return np.array(out)

if __name__ == '__main__':
    test()