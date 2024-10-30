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
import numpy as np
import matplotlib.pyplot as plt

# Hyperparameters
batch_size = 32
learning_rate = 1e-4
num_epochs = 1
num_timesteps = 1000
n_inference_steps = 50
loss_to_break = 0.01
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
model = Diffusion().to(device)
#print number of parameters in model 
num_param = sum(p.numel() for p in model.parameters())
print("Number of parameters in model:", num_param/1e6, "M")

criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
generator = torch.Generator(device=device)
if seed is None:
    generator.seed()
sampler = DDPMSampler(generator)
sampler.set_inference_timesteps(num_inference_steps=n_inference_steps)




# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data, labels) in enumerate(train_loader):
        start_time = time.time()
        data = data.to(device)
        batch_size = data.size(0)
       
            
        labels_edit = torch.ones_like(data) * labels.view(-1, 1, 1, 1)

        # Sample random timesteps uniformly
        timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
        
        time_embedding = get_time_embedding(timesteps)
        # Add noise to the images
        noisy_data, noise = sampler.add_noise(data, timesteps)
       
        # Predict the noise
        optimizer.zero_grad()
        #print("noisy_data shape:", noisy_data.shape)
        predicted_noise = model(noisy_data, labels_edit, time_embedding)
        
        # Compute loss
        loss = criterion(predicted_noise, noise)

        if loss < loss_to_break:
            print("Loss is less than ", loss_to_break, ". Breaking.")
            break
        
        # Backpropagation and optimization
        loss.backward()
        optimizer.step()

        end_time = time.time()
        time_taken = end_time - start_time

        print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}, Time: {time_taken:.4f}')
        if batch_idx % 100 == 0:
            print(f'Epoch [{epoch+1}/{num_epochs}], Step [{batch_idx}/{len(train_loader)}], Loss: {loss.item():.4f}')
            
            # Initialize the noisy image
            print("noisy_data shape:", noisy_data.shape)
            current_noisy_data = noisy_data[0].clone().unsqueeze(0)
            print("current_noisy_data shape:", current_noisy_data.shape)

            #set model to eval for inference
            model.eval()
            # Run the reverse diffusion process
            num_steps_to_denoise = int(torch.round(timesteps[0]/n_inference_steps))

            
            for t in reversed(range(num_steps_to_denoise)):
                print("inference denoising time step: ", t)
                timesteps_inf = torch.tensor([np.round(t*n_inference_steps)], device=device)
                time_embedding = get_time_embedding(timesteps_inf)
                predicted_noise = model(current_noisy_data, labels_edit[0].clone().unsqueeze(0), time_embedding)
                current_noisy_data = sampler.step(timesteps_inf, current_noisy_data, predicted_noise)
            
            unnoised_data = current_noisy_data

            # Plot the noised image and the unnoised image
            noisy_data_np = noisy_data[0].squeeze().cpu().detach().numpy()
            unnoised_data_np = unnoised_data.squeeze().cpu().detach().numpy()

            fig, ax = plt.subplots(1, 3)

            ax[0].imshow(noisy_data_np, cmap='gray')
            ax[0].set_title("Noised Image")
            ax[0].axis('off')
            ax[1].imshow(unnoised_data_np, cmap='gray')
            ax[1].set_title("Unnoised Image")
            ax[1].axis('off')
            ax[2].imshow(data[0].squeeze().cpu().detach().numpy(), cmap='gray')
            ax[2].set_title("Original Image")
            ax[2].axis('off')
            # Add overall plot title with the label from labels
            fig.suptitle(f"Label: {labels[0]}")

            plt.draw()
            plt.pause(0.1)

            #set model back to train
            model.train()

    if loss < loss_to_break:
        break

#save model
torch.save(model.state_dict(), 'saved_models/diffusion_model.pth')

print("Training complete. Model saved.")