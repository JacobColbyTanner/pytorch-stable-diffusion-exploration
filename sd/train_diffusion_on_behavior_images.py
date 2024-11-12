import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset
from torchvision import datasets, transforms
from diffusion import Diffusion
from ddpm import DDPMSampler
from pipeline import get_time_embedding
from tqdm import tqdm
import time
import numpy as np
import matplotlib.pyplot as plt
from RL_agent.RL_train_cartpole import GymAgent


#Diffusion hyperparameters
batch_size = 32
learning_rate = 1e-4
num_epochs = 10
num_timesteps = 1000
n_inference_steps = 50
loss_to_break = 0.001
seed = None
norm_mean = 0
norm_std = 1
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
inference_interval = 20
use_inference_steps = False

#Agent hyperparameters
max_runs_reward = 400 #maximize the number of steps during training simulation (equivalent to the maximum reward because you can get at most 1 for reward per time step)
image_size_rows = 50 #max_runs_reward
num_images = 1000
num_random_images = 200
environment = 'CartPole-v1'
file_path_RL = 'saved_models/cartpole_policy_net.pth'
collect_data = False

#initialize agent class
agent = GymAgent(environment)
if collect_data:    
    print("Collecting data...")
    agent.collect_data(file_path_RL,num_images,image_size_rows)
    agent.collect_random_data(num_random_images,image_size_rows)






class CustomDataset(Dataset):
    def __init__(self, file_path, transform=None):
        self.data = np.load(file_path)
        self.transform = transform
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        image = self.data[idx]
        

        #image[:,2] = np.log(image[:,2]+0.5)
        
        if self.transform:
            image = self.transform(image)
        #image[:,:,2] = torch.log(image[:,:,2] + 1e-6)
        

        return image
    
def calculate_column_stats(dataset):
    # Stack all images to calculate mean and std for each column
    all_images = np.stack([dataset[i] for i in range(len(dataset))])
    mean = np.mean(all_images, axis=(0, 2))
    std = np.std(all_images, axis=(0, 2))
    return mean, std

class ColumnNormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # Normalize each column independently
        mean = torch.tensor(self.mean).view(1, -1, 1)
        std = torch.tensor(self.std).view(1, -1, 1)
        return (img - mean) / std

class ColumnUnnormalize:
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, img):
        # Unnormalize each column independently
        mean = torch.tensor(self.mean).view(1, -1, 1)
        std = torch.tensor(self.std).view(1, -1, 1)
        return img * std + mean



# Dataset and DataLoader with normalization
transform = transforms.Compose([
    transforms.ToTensor(),
    #ColumnNormalize(norm_mean, norm_std)
])

file_path_data = "/Users/jacobtanner/pytorch-stable-diffusion/saved_state_action_rewards/cartpole_states_actions_rewards.npy"
train_dataset = CustomDataset(file_path_data, transform=transform)



file_path_data_random = "/Users/jacobtanner/pytorch-stable-diffusion/saved_state_action_rewards/cartpole_states_actions_rewards_random.npy"
train_dataset_random = CustomDataset(file_path_data_random, transform=transform)
train_dataset.data = np.concatenate([train_dataset.data, train_dataset_random.data], axis=0)
mean, std = calculate_column_stats(train_dataset.data)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
print("train_loader length:", len(train_loader))

#to add back original mean and std
reverse_transform = ColumnUnnormalize(mean, std)
# Initialize model, loss function, and optimizer
model = Diffusion().to(device)
#print number of parameters in model 
num_param = sum(p.numel() for p in model.parameters())
print("Number of parameters in model:", num_param/1e6, "M")

criterion = nn.MSELoss()
criterion_infer = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=learning_rate)
generator = torch.Generator(device=device)
if seed is None:
    generator.seed()
sampler = DDPMSampler(generator)
sampler.set_inference_timesteps(num_inference_steps=n_inference_steps)

# here are all the unique characters that occur in the alphabet
text = "abcdefghijklmnopqrstuvwxyz"
chars = sorted(list(set(text)))
vocab_size = len(chars)
print('Vocabulary size:', vocab_size)
# create a mapping from characters to integers
stoi = { ch:i for i,ch in enumerate(chars) }
itos = { i:ch for i,ch in enumerate(chars) }
encode = lambda s: [stoi[c] for c in s] # encoder: take a string, output a list of integers
decode = lambda l: ''.join([itos[i] for i in l]) # decoder: take a list of integers, output a string



# Training loop
for epoch in range(num_epochs):
    for batch_idx, (data) in enumerate(train_loader):
        
        start_time = time.time()
        #change data to torch type float32
        data = data.float()
        data = data.to(device)
        
        batch_size = data.size(0)

        labels_text = "cartpole"
        #tokenize the labels
        labels_edit = torch.tensor([encode(labels_text)]).to(device)
        #copy the labels for batch size
        labels_edit = labels_edit.repeat(batch_size, 1)
        
        
        #send labels edit to the device
        labels_edit = labels_edit.to(device)

        # Sample random timesteps uniformly
        #timesteps = torch.randint(0, num_timesteps, (batch_size,), device=device).long()
        #sample from half normal so as to sample more time steps with less noise (to make training easier)
        timesteps = torch.abs(torch.randn(batch_size, device=device)) * (num_timesteps / 4)
        timesteps = timesteps.long().clamp(0, num_timesteps - 1)
        

        #initialize time embedding
        time_embedding = torch.zeros((batch_size, 320), device=device)
        for b in range(batch_size):
            time_embedding[b] = get_time_embedding(timesteps[b],device)
            
    
        # Add noise to the images
        noisy_data, noise = sampler.add_noise(data, timesteps)
       
        # Predict the noise
        optimizer.zero_grad()
        #print("noisy_data shape:", noisy_data.shape)
        
        predicted_noise = model(noisy_data, labels_edit, time_embedding)
        
        #print("predicted_noise shape:", predicted_noise.shape)
        #print("noise shape:", noise.shape)
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

        #############################THEN DO INFERENCE TO CHECK IF IT IS WORKING ##############################
        if batch_idx % inference_interval == 0:   
            
            print("starting inference test...")
            
            # Initialize the noisy image
            print("noisy_data shape:", noisy_data.shape)
            current_noisy_data = noisy_data[0].clone().unsqueeze(0)
            print("current_noisy_data shape:", current_noisy_data.shape)

            #set model to eval for inference
            model.eval()
            # Run the reverse diffusion process
            if use_inference_steps:
                
                num_steps_to_denoise = int(torch.round(timesteps[0]/n_inference_steps))
                
                inference_loss = []
                for t in reversed(range(num_steps_to_denoise)):
                    print("inference denoising time step: ", t)
                    inference_loss.append(criterion_infer(current_noisy_data, data[0].clone().unsqueeze(0)).detach().numpy())
                    timesteps_inf = torch.tensor([np.round(t*n_inference_steps)], device=device)
                    time_embedding = get_time_embedding(timesteps_inf, device)
                    predicted_noise = model(current_noisy_data, labels_edit[0].clone().unsqueeze(0), time_embedding)
                    current_noisy_data = sampler.step(timesteps_inf, current_noisy_data, predicted_noise)
            else:

                num_steps_to_denoise = int(timesteps[0])
                
                inference_loss = []
                for t in reversed(range(num_steps_to_denoise)):
                    print("inference denoising time step: ", t)
                    inference_loss.append(criterion_infer(current_noisy_data, data[0].clone().unsqueeze(0)).detach().numpy())
                    timesteps_inf = torch.tensor([t], device=device)
                    time_embedding = get_time_embedding(timesteps_inf, device)
                    predicted_noise = model(current_noisy_data, labels_edit[0].clone().unsqueeze(0), time_embedding)
                    current_noisy_data = sampler.step(timesteps_inf, current_noisy_data, predicted_noise)



            inference_loss = np.array(inference_loss)
            
            unnoised_data = current_noisy_data

            # Plot the noised image and the unnoised image
            noisy_data_np = noisy_data[0].squeeze().cpu().detach().numpy()
            unnoised_data_np = unnoised_data.squeeze().cpu().detach().numpy()

            fig, ax = plt.subplots(1, 5)

            cax = ax[0].imshow(noisy_data_np[0:10,:])
            # Add a colorbar to the first axis
            fig.colorbar(cax, ax=ax[0]) 
            ax[0].set_title("Noised Image")
            ax[0].axis('off')
            cax2 = ax[1].imshow(unnoised_data_np[0:10,:])
            fig.colorbar(cax2, ax=ax[1]) 
            ax[1].set_title("Unnoised Image")
            ax[1].axis('off')
            D = data[0].squeeze().cpu().detach().numpy()
            cax3 = ax[2].imshow(D[0:10,:])
            fig.colorbar(cax3, ax=ax[2]) 
            ax[2].set_title("Original Image")
            ax[2].axis('off')
            # Add overall plot title with the label from labels
            ax[3].plot(inference_loss)
            ax[3].set_title("Inference Loss")
            ax[3].set_ylabel("MSE Loss")
            ax[3].set_xlabel("Denoising time step")
            #force plot to be square
            ax[3].set_aspect('equal', 'box')
            cax4 = ax[4].imshow(predicted_noise.squeeze().cpu().detach().numpy()[0:10,:])
            fig.colorbar(cax4, ax=ax[4]) 
            ax[4].set_title("Predicted Noise")
            ax[4].axis('off')
            fig.suptitle(f"Label: {labels_text} - noise from t={timesteps[0]} forward diffusion steps")


            # Save the plot
            plt.savefig(f"images/diffusion_{epoch}_{batch_idx}.png")

            #set model back to train
            model.train()

    if loss < loss_to_break:
        break

#save model
torch.save(model.state_dict(), 'saved_models/diffusion_model_to_behavior_with_random.pth')

print("Training complete. Model saved.")