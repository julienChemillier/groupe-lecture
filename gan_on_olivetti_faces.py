import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.utils.data as data
import torchvision.utils as vutils
import matplotlib.pyplot as plt

# Configuration
cuda_usage = True
BATCH_SIZE = 20
Z_DIM = 100
G_HIDDEN = 64
D_HIDDEN = 64
X_DIM = 64
IMAGE_CHANNEL = 1
EPOCH_NUM = 301
REAL_LABEL = 1
FAKE_LABEL = 0
learning_rate = 2e-4

# Setting up CUDA for GPU usage if available
cuda_available = cuda_usage and torch.cuda.is_available()
print("PyTorch version: {}".format(torch.__version__))
if cuda_available:
    print("CUDA version: {}\n".format(torch.version.cuda))

if cuda_available:
    torch.cuda.manual_seed(seed)
device = torch.device("cuda:0" if cuda_available else "cpu")

# Loading Olivetti Faces dataset
faces = np.load("olivetti_faces.npy")  # Shape (400, 64, 64)
faces = (faces - 0.5) / 0.5  # Normalize to [-1, 1]
faces = np.expand_dims(faces, axis=1)  # Add channel dimension
dataset = data.TensorDataset(torch.tensor(faces, dtype=torch.float32))
dataloader = data.DataLoader(dataset, batch_size=BATCH_SIZE, shuffle=True)


# ---------------------------------------- Classes -------------------------------------------


class Generator(nn.Module):
    def __init__(self):
        super(Generator, self).__init__()
        self.main = nn.Sequential(
            nn.ConvTranspose2d(Z_DIM, G_HIDDEN * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 8),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_HIDDEN * 8, G_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 4),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_HIDDEN * 4, G_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN * 2),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_HIDDEN * 2, G_HIDDEN, 4, 2, 1, bias=False),
            nn.BatchNorm2d(G_HIDDEN),
            nn.ReLU(True),
            nn.ConvTranspose2d(G_HIDDEN, IMAGE_CHANNEL, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    def forward(self, input):
        return self.main(input)


class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()
        self.main = nn.Sequential(
            nn.Conv2d(IMAGE_CHANNEL, D_HIDDEN, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(D_HIDDEN, D_HIDDEN * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 2),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(D_HIDDEN * 2, D_HIDDEN * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 4),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(D_HIDDEN * 4, D_HIDDEN * 8, 4, 2, 1, bias=False),
            nn.BatchNorm2d(D_HIDDEN * 8),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(D_HIDDEN * 8, 1, 4, 1, 0, bias=False),
            nn.Sigmoid()
        )
    def forward(self, input):
        return self.main(input).view(-1, 1).squeeze(1)

# ---------------------------------------- Training -------------------------------------------


# Initialize networks
netG = Generator().to(device)
netD = Discriminator().to(device)
criterion = nn.BCELoss()
optimizerD = optim.Adam(netD.parameters(), lr=learning_rate, betas=(0.5, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=learning_rate, betas=(0.5, 0.999))


# Vectors to see the evolution
vector_see_evolution = torch.randn(BATCH_SIZE, Z_DIM, 1, 1, device=device)
G_losses = []
D_losses = []

for epoch in range(EPOCH_NUM):
    for i, data in enumerate(dataloader, 0):
        # Discriminator with real data
        netD.zero_grad()
        real_cpu = data[0].to(device)
        b_size = real_cpu.size(0)
        label = torch.full((b_size,), REAL_LABEL, dtype=torch.float, device=device)
        output = netD(real_cpu).view(-1)
        errD_real = criterion(output, label)
        errD_real.backward()

        # Discriminator with fake data
        noise = torch.randn(b_size, Z_DIM, 1, 1, device=device)
        fake = netG(noise)
        label.fill_(FAKE_LABEL)
        output = netD(fake.detach()).view(-1)
        errD_fake = criterion(output, label)
        errD_fake.backward()
        optimizerD.step()

        # Generator with fake data
        netG.zero_grad()
        label.fill_(REAL_LABEL)
        output = netD(fake).view(-1)
        errG = criterion(output, label)
        errG.backward()
        optimizerG.step()

    G_losses.append(errG.item())
    D_losses.append(errD_real.item() + errD_fake.item())

    # Display images every 25 epochs during training
    if epoch % 25 == 0:
        with torch.no_grad():
            fake = netG(vector_see_evolution).detach().cpu()
        img_grid = vutils.make_grid(fake, padding=2, normalize=True)
        plt.figure(figsize=(10, 10))
        plt.imshow(np.transpose(img_grid, (1, 2, 0)))
        plt.title(f'Epoch {epoch}')
        plt.axis('off')
        plt.show()
