import torch
import torch.nn as nn
from torch.nn import functional as F
from transformer_model import estimate_loss, get_action_batch, transformer_model
from transformers import GPT2Tokenizer
import numpy as np
from scipy.io import loadmat, savemat
import h5py
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import find_peaks
import time
import os
from RL_agent.RL_train_cartpole import GymAgent
import gym

# hyperparameters transformer model
batch_size = 32 # how many independent sequences will we process in parallel?
block_size = 50 # what is the maximum context length for predictions?
max_iters = 10000
eval_interval = 200
learning_rate = 1e-3
device = 'cuda' if torch.cuda.is_available() else 'cpu'
eval_iters = 200
num_state_action_reward = 6 
n_embd = 100 #embedding dimension
n_head = 2
n_layer = 1
dropout = 0.1 #dropout probability
# ------------

#Agent hyperparameters
max_runs_reward = 400 #maximize the number of steps during training simulation (equivalent to the maximum reward because you can get at most 1 for reward per time step)
image_size_rows = 100 #max_runs_reward
num_images = 5000
environment = 'CartPole-v1'
file_path_RL = 'saved_models/cartpole_policy_net.pth'
collect_data = False

#initialize agent class
agent = GymAgent(environment)
if collect_data:    
    print("Collecting data...")
    agent.collect_data(file_path_RL,num_images,image_size_rows)
    




file_path_data = "/Users/jacobtanner/pytorch-stable-diffusion/saved_state_action_rewards/cartpole_states_actions_rewards.npy"

data = np.load(file_path_data, allow_pickle=True)



batch, target_ts_batch = get_action_batch(data,batch_size, block_size, num_images, train_test='train')

print("batch shape: ", batch.shape)
print("target_ts_batch shape: ", target_ts_batch.shape)



model = transformer_model(num_state_action_reward, n_embd, n_head, dropout, block_size, batch_size, n_layer, device)
model = model.to(device)
# print the number of parameters in the model
print(sum(p.numel() for p in model.parameters())/1e6, 'M parameters')

# create a PyTorch optimizer
optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)

for iter in range(max_iters):

    # every once in a while evaluate the loss on train and val sets
    if iter % eval_interval == 0 or iter == max_iters - 1:
        losses = estimate_loss(model, data, eval_iters, block_size, batch_size, num_images)
        print(f"step {iter}: train loss {losses['train']:.4f}, val loss {losses['test']:.4f}")


    # sample a batch of data
    batch, target_ts_batch = get_action_batch(data,batch_size, block_size, num_images, train_test='train')

    # evaluate the loss
    logits, loss = model(batch, targets = target_ts_batch)
    optimizer.zero_grad(set_to_none=True)
    loss.backward()
    optimizer.step()

# generate from the model
model.eval()
context = batch[0,0:block_size,:].squeeze(0)
predicted_actions = model.generate(context, max_new_tokens=10, keep_starting_tokens=True)

plt.figure()
plt.imshow(predicted_actions.detach().numpy())
plt.show()

def visualize(env_name, model):
    env = gym.make(env_name, render_mode='human')

    

    state,_ = env.reset()
    done = False
    total_reward = 0
    
    while not done:
        env.render()
        state_action_reward = model.generate(context, max_new_tokens=1)
        action = state_action_reward[0,-2]
        if action > 0.5:
            action = 1
        else:
            action = 0
        state, reward, done, _, _ = env.step(action)
        total_reward += reward
    

    env.close()
    print(f"Total Reward: {total_reward}")


visualize(environment, model)


