import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from model import Recogniser
from torch.utils.data import random_split, DataLoader
from dataloader import AudioFlac, ReadData
from tqdm import tqdm

def train():
    # Load Dataset
    train_size = 20000
    batch_size = 40
    dataset = ReadData()
    train_data, ver_data = random_split(dataset=dataset, lengths=[train_size, len(dataset)-train_size])
    dataloader = DataLoader(train_data, batch_size=batch_size, shuffle=True, num_workers=8)
    verloader = DataLoader(ver_data, batch_size=batch_size, shuffle=True, num_workers=8)

    # Establish Model
    recogniser = Recogniser().cuda()
    optimizer = torch.optim.Adam(recogniser.parameters(), lr=1e-4)
    #criterion = torch.nn.CrossEntropyLoss()
    criterion = torch.nn.BCELoss()
    #criterion = torch.nn.MSELoss()

    # Training Loop
    epoch = 20

    print("Start training...")
    loss_list = []

    for _ in range(epoch):
        print(f"Epoch{_+1:>4d}:")
        recogniser.train()
        pbar = tqdm(dataloader)
        for x, label in pbar:
            x = x.cuda().float()

            #noise = get_noise(x)
            label = label.cuda().float()
            # for n in noise:
            #     n = n.cuda().float()
            #     x_noise = x + n
            #     optimizer.zero_grad()
            #     out = recogniser(x_noise)
            #     loss = criterion(out, label)
            #     loss.backward()
            #     optimizer.step()
            #     pbar.set_description(f"Loss:{loss.item():.4f}")
            optimizer.zero_grad()
            out = recogniser(x)
            loss = criterion(out, label)
            loss.backward()
            optimizer.step()
            loss_list.append(loss.item())
            pbar.set_description(f"Loss:{loss.item():.4f}")

    # Saving Model
    print("Saving model...")
    torch.save(recogniser, "Recogniser.pt")
    np.savetxt("loss.txt", np.array(loss_list))

    # Verifying Loop
    print("Start verifying")
    total = 0
    correct = 0
    with torch.no_grad():
        for i, label in verloader:
            out = recogniser(i.cuda().float())
            _1, pred1 = torch.max(out.data, dim=2)
            _2, pred2 = torch.max(label.cuda().data, dim=2)
            correct += (pred1==pred2).sum().item()
            total += len(pred1)
    print(f"Accuracy:{correct/total:.4f}")


if __name__ == '__main__':
    train()