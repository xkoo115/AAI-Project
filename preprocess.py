from dataloader import AudioNoise, AudioSPK
import torch
import random

noise_data = AudioNoise()
spk_data = AudioSPK()

for i, (audio, speaker) in enumerate(spk_data):
    audio = audio.cuda()
    audio_fft = torch.fft.fft(audio, 400000)
    noise_fft = torch.fft.fft(noise_data[random.randint(0,len(noise_data)-1)], 400000).cuda()
    audio_noise = audio_fft + noise_fft
    torch.save(audio_fft.cpu()[:,:200000], f"data/origin/{speaker}_{i:>5d}_o.pt")
    #torch.save(audio_noise.cpu()[:,:200000], f"data/noise/{speaker}_{i:>5d}_n.pt")