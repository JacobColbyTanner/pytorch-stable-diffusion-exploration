import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
from diffusion import Diffusion
from ddpm import DDPMSampler
from pipeline import get_time_embedding
from tqdm import tqdm
import time
import matplotlib.pyplot as plt
import numpy as np

# Hyperparameters
learning_rate = 1e-4
num_epochs = 1
label_select = 0

seed = None
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Dataset and DataLoader
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])

train_dataset = datasets.MNIST(root='./data', train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

# Initialize model, loss function, and optimizer
model = Diffusion()
#load saved model
model.load_state_dict(torch.load('saved_models/diffusion_model.pth'))
print("Model loaded successfully.")

#print number of parameters in model 
num_param = sum(p.numel() for p in model.parameters())
print("Number of parameters in model:", num_param/1e6, "M")


generator = torch.Generator(device=device)
if seed is None:
    generator.seed()
sampler = DDPMSampler(generator)

# Testing loop

model.eval()

label = torch.tensor(label_select, device=device)

# Sample random timesteps
random_data = torch.rand((1, 1, 28, 28), device=device)
timesteps = sampler.timesteps
time_embeddings = []
timestep = timesteps[0]
time_embedding = get_time_embedding(timestep)
    
labels_edit = torch.ones_like(random_data) * label.view(-1, 1, 1, 1)

# Add noise to the images
noisy_data, noise = sampler.add_noise(random_data, sampler.timesteps[0])


#print("noisy_data shape:", noisy_data.shape)
predicted_noise = model(noisy_data, labels_edit, time_embedding)

unnoised_data = sampler.step(timestep, noisy_data, predicted_noise)


#plot the noised image, and the unnoised image


noisy_data = noisy_data.squeeze().cpu().detach().numpy()
unnoised_data = unnoised_data.squeeze().cpu().detach().numpy()

fig, ax = plt.subplots(1, 2)
ax[0].imshow(noisy_data, cmap='gray')
ax[0].set_title("Noised Image")
ax[0].axis('off')
ax[1].imshow(unnoised_data, cmap='gray')
ax[1].set_title("Unnoised Image")
ax[1].axis('off')

plt.show()



