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
import gym
from tqdm import tqdm


#hyperparameters
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
environment = 'CartPole-v1'
file_path_RL = 'saved_models/cartpole_policy_net.pth'
N_steps = 40 #number of steps to predict into the future
test_interval = 10 #number of steps to test the model-based agent
training_steps = 2000 #number of training steps for the model-based agent
batch_size = 1 #number of possible predicted futures to consider given current state and expectation of future rewards
num_inference_steps = 1

# Instantiate the model
diffusion_model = Diffusion().to(device)

# Load the saved model state dictionary
diffusion_model.load_state_dict(torch.load('saved_models/diffusion_model_to_behavior_with_random.pth'))



#open model-free RL agent
#initialize agent class
model_free_agent = GymAgent(environment)
model_free_agent.load_model_weights(file_path_RL)


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



# Create class for select action, train, visualize and so on
class GymDiffusionAgent:
    def __init__(self, environment, diffusion_model, N_steps, num_inference_steps, learning_rate=1e-5):
        self.env_name = environment
        self.env = gym.make(environment)
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = 1
        self.reward_dim = 1
        self.N_steps = N_steps
        self.diffusion_model = diffusion_model
        # Set the model to evaluation mode
        self.diffusion_model.eval()
        generator = torch.Generator(device=device)
        self.sampler = DDPMSampler(generator)
        self.criterion = nn.MSELoss()
        self.optimizer = optim.Adam(self.diffusion_model.parameters(), lr=learning_rate)
        self.num_inference_steps = num_inference_steps

    def select_action_from_inference(self,predicted_state_action_reward, step_idx):
        
        #action is the next step in the predicted state action reward (in the action dimension)
        if predicted_state_action_reward[step_idx+1, self.state_dim:self.state_dim+self.action_dim] > 0.5:
            action = 1
        else:
            action = 0

        return action

    def compute_returns(self, rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns
    
    def inference(self, starting_state, state_time_indices, N_steps, batch_size):


        labels_text = "cartpole"
        #tokenize the labels
        labels_edit = torch.tensor([encode(labels_text)]).to(device)
        
        #send labels edit to the device
        labels_edit = labels_edit.to(device)
        labels_edit = labels_edit.repeat(batch_size, 1)

        num_columns = self.state_dim+self.action_dim+self.reward_dim
        #create random noise image that is size N_steps x len(input_tensor)
        current_noisy_data = torch.randn(N_steps, num_columns, device=device)*0.5
        current_noisy_data[state_time_indices,0:self.state_dim] = torch.from_numpy(starting_state) #condition the noise image on the current state of the environment
        #add reward of 1 to the reward column in order to predict that the agent got rewards
        current_noisy_data[:,-1] = 1
        #add two initial dimensions for batch size and channel  
        current_noisy_data = current_noisy_data.unsqueeze(0)
        #repeat this across num_batches
        current_noisy_data = current_noisy_data.repeat(batch_size, 1, 1, 1)
        
        for t in tqdm(reversed(range(self.num_inference_steps)), desc="Inference Progress"):
            #print("inference denoising time step: ", t)
            timesteps_inf = torch.tensor([t], device=device)
            time_embedding = get_time_embedding(timesteps_inf, device)
            time_embedding = time_embedding.repeat(batch_size, 1)
            predicted_noise = diffusion_model(current_noisy_data, labels_edit.clone().unsqueeze(0), time_embedding)
            current_noisy_data = self.sampler.step(timesteps_inf, current_noisy_data, predicted_noise)
            
            #make sure that starting state and full rewards are consistent across inference
            current_noisy_data[0,0,state_time_indices,0:self.state_dim] = torch.from_numpy(starting_state) #condition the noise image on the current state of the environment
            #add reward of 1 to the reward column in order to predict that the agent got rewards
            current_noisy_data[:,:,:,-1] = 1

        
        
        #Now we have the denoised image
        denoised_action_image = current_noisy_data #.squeeze(0).squeeze(0)
        #check if nans in the denoised image
        #print("nans in denoised image: ", torch.isnan(denoised_action_image).any())
        #plot the image
        #plt.imshow(denoised_action_image.clone().detach().cpu().numpy())
        #plt.colorbar()
        #plt.show()

        return denoised_action_image
    
    def select_batch_by_distance(self, predicted_batch, observed):
        #calculate the distance between the predicted batch states and the observed state
        distance = torch.sum(torch.abs(predicted_batch-observed), dim=1)
        #select the batch with the minimum distance
        min_distance_index = torch.argmin(distance)

        return min_distance_index

    def test(self, N_steps_action):
        env = gym.make(self.env_name)
        batch_size = 1

        state,_ = env.reset()
        #predicted state action reward for N steps
        predicted_state_action_reward = self.inference(state, 0, N_steps_action, batch_size)

        done = False
        total_reward = 0
        step_idx = 0
        while not done:
            action = self.select_action_from_inference(predicted_state_action_reward[0].squeeze(0).squeeze(0), step_idx)
            step_idx += 1
            state, reward, done, _, _ = env.step(action)

            total_reward += reward

        env.close()
        print(f"___________________________________________________Total Reward: {total_reward} ____________________________")
        return total_reward
    
    def visualize(self, N_steps_action, batch_size):
        env = gym.make(self.env_name, render_mode='human')

        starting_state,_ = env.reset()
        #predicted state action reward for N steps
        predicted_state_action_reward = self.inference(starting_state, 0, N_steps_action, batch_size)

        done = False
        total_reward = 0
        step_idx = 0
        batch_select = 0 #initialize the batch select
        all_distances = []
        observed_states_so_far = []
        observed_states_so_far.append(starting_state)


        while not done:
            env.render()
            action = self.select_action_from_inference(predicted_state_action_reward[batch_select].squeeze(0).squeeze(0), step_idx)
            step_idx += 1
            state, reward, done, _, _ = env.step(action)
            observed_states_so_far.append(state)
            predicted_batch = predicted_state_action_reward[:,:,step_idx,0:self.state_dim].squeeze(1)
            observed = torch.from_numpy(state).to(device)
            #select whichever batch is closest to use
            batch_select = self.select_batch_by_distance(predicted_batch, observed)
            #calculate the distance of the best possible state (from batch selection above) to the observed state
            distance = self.calculate_state_distance(predicted_batch[batch_select].squeeze(0).squeeze(0), observed)
            if distance > 1.45:
                print("Distance is high. Performing new inference...")
                predicted_state_action_reward = self.inference(np.array(observed_states_so_far), np.arange(0,step_idx+1) ,N_steps_inference, batch_size)
                batch_select = self.select_batch_by_distance(predicted_batch, observed)
            all_distances.append(distance.detach().numpy())
            predicted_state_action_reward[batch_select].squeeze(0).squeeze(0)
            total_reward += reward

        env.close()
        print(f"___________________________________________________Total Reward: {total_reward} ____________________________")

        
        all_distances = np.array(all_distances)

        return all_distances
        


    def calculate_state_distance(self, predicted_state, observed_state):
        #calculate the distance between the predicted state and the observed state
        distance = torch.sum(torch.abs(predicted_state-observed_state))

        return distance
    

    def train_model_based_RL(self, num_training_steps, batch_size):
        
        #set diffusion model to train
        self.diffusion_model.train()
        
        
        all_loss = []
        all_trail_lengths = []
        for i in range(num_training_steps):
            start_time = time.time()
            #initialize the environment
            state,_ = self.env.reset()
            #train the model-based agent
            #predicted state action reward for N steps
            predicted_state_action_reward = self.inference(state, 0, self.N_steps, batch_size)
            loss, t = self.train_loop(state, predicted_state_action_reward)
            #store loss
            #all_trail_lengths.append(t)
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            total_time = time.time()-start_time
            all_loss.append(loss.clone().detach().numpy())
            
            print(f"Step: {i}, Loss: {loss.item()}, Time: {total_time}")
            if i % test_interval == 0:
                print("testing... max reward possible is 200")
                all_trail_lengths.append(self.test(200))
                #all_loss.append(loss.clone().detach().numpy())
                #plot the loss
                plt.figure()
                #plot subplots
                plt.subplot(2,1,1)
                plt.plot(all_loss)
                plt.xlabel("Training Steps")
                plt.ylabel("Loss")
                plt.subplot(2,1,2)
                plt.plot(all_trail_lengths)
                plt.xlabel("Training Steps")
                plt.ylabel("Trail Length/Total Reward")
                plt.draw()
                plt.pause(0.1)

                torch.save(self.diffusion_model.state_dict(), 'saved_models/diffusion_model_to_behavior_after_RL.pth')

        self.diffusion_model.eval()

    def train_loop(self, state, predicted_state_action_reward):
        
        print("pred state act rew shape: ", predicted_state_action_reward.shape)
        state_action_reward = torch.zeros_like(predicted_state_action_reward.squeeze(0).squeeze(0))
        print("state action reward shape: ", state_action_reward.shape)
        state_action_reward[0,:] = predicted_state_action_reward[0,0,0,:] #initialize the state action reward with the first predicted state action reward  
        for t in range(self.N_steps-1):
            #select action
            action = self.select_action_from_inference(predicted_state_action_reward[0].squeeze(0).squeeze(0), t)
            #take action
            next_state, reward, done, _, _ = self.env.step(action)
            state = torch.from_numpy(state).to(device)
            action = torch.tensor([action], device=device)
            reward = torch.tensor([reward], device=device)
            state_action_reward[t+1,:] = torch.concat([state, action, reward])
            #update the state
            state = next_state
            #check if done
            if done:
                
                total_t_to_consider = t+1
                #remove parts of action images where action leads to failure. Agent should be predicting that it is successful.
                state_action_reward = state_action_reward[0:total_t_to_consider]
                predicted_state_action_reward = predicted_state_action_reward[0:total_t_to_consider]
            
                print("Done at time step: ", t)
                break
        
        print("state action reward shape: ", state_action_reward.shape)
        print("predicted state action reward shape: ", predicted_state_action_reward.shape)

        MSE_loss = self.criterion(state_action_reward,predicted_state_action_reward[0].squeeze(0).squeeze(0))
        #calculate distance of average predicted reward from 1
        avg_reward_distance = torch.sum(torch.abs(1-predicted_state_action_reward[:,:,:,-1]))
        steps_failed = self.N_steps - (t+1)
        loss = MSE_loss #+ avg_reward_distance #+ steps_failed  #this should punish the model for not predicting that it gets reward, so it shouldnt start predicting that it doesn't get reward
        
        '''
        plt.figure()
        plt.plot(predicted_state_action_reward[:,-1].clone().detach().cpu().numpy())
        plt.xlabel("time points")
        #select xticks to be 0 to self.N_steps
        plt.xticks(np.arange(0,self.N_steps))
        plt.ylabel("reward")
        plt.draw()
        plt.pause(0.1)
        '''

        return loss, t
        



#initialize the model-based agent
model_based_agent = GymDiffusionAgent(environment, diffusion_model, N_steps, num_inference_steps)

#visualize the model-based agent
model_based_agent.train_model_based_RL(training_steps, batch_size)

'''
all_distances = []
for i in range(5):
    all_distances.append(model_based_agent.visualize(200,batch_size))

plt.figure()
for i in range(len(all_distances)):
    plt.plot(all_distances[i])
plt.xlabel("Time Steps")
plt.ylabel("Distance")
plt.show()
'''

