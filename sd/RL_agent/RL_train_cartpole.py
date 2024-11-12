import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import gym
import matplotlib.pyplot as plt



#note for collecting data, the image size needs to be divisible by 2
#hyperparameters

select_to_do = "collect_data" #train, visualize, collect_data
max_runs_reward = 400 #maximize the number of steps during training simulation (equivalent to the maximum reward because you can get at most 1 for reward per time step)
image_size_rows = max_runs_reward
num_images = 10
environment = 'CartPole-v1'
file_path = 'saved_models/cartpole_policy_net.pth'

# Define the policy network
class PolicyNet(nn.Module):
    def __init__(self, state_dim, action_dim):
        super(PolicyNet, self).__init__()
        self.fc1 = nn.Linear(state_dim, 128)
        self.fc2 = nn.Linear(128, action_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.softmax(self.fc2(x), dim=-1)
        return x
    
# Create class for select action, train, visualize and so on
class GymAgent:
    def __init__(self, environment, learning_rate=1e-3):
        self.env_name = environment
        self.env = gym.make(environment)
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n
        self.policy_net = PolicyNet(state_dim, action_dim)
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)

    def load_model_weights(self, file_path):
        self.policy_net.load_state_dict(torch.load(file_path))

    def select_action(self, state, policy_net):
        state_tensor = torch.from_numpy(state).float().unsqueeze(0)
        action_probs = policy_net(state_tensor)
        action = np.random.choice(len(action_probs[0]), p=action_probs.detach().numpy()[0])
        return action

    def compute_returns(self, rewards, gamma=0.99):
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
        return returns

    def train(self, max_runs_reward):
        
        
        num_episodes = 1000
        rewards_list = []
        for episode in range(num_episodes):
            state,_ = self.env.reset()
            states = []
            actions = []
            rewards = []

            for t in range(max_runs_reward):
                action = self.select_action(state, self.policy_net)
                next_state, reward, done,_,_ = self.env.step(action)
                

                states.append(state)
                actions.append(action)
                rewards.append(reward)

                if done:
                    break
                state = next_state

            returns = self.compute_returns(rewards)
            returns = torch.tensor(returns, dtype=torch.float32)
            states = torch.tensor(states, dtype=torch.float32)
            actions = torch.tensor(actions, dtype=torch.long)

            # Compute loss
            self.optimizer.zero_grad()
            for i in range(len(states)):
                state_tensor = states[i].unsqueeze(0)
                action_tensor = torch.tensor([actions[i]], dtype=torch.long)
                action_probs = self.policy_net(state_tensor)
                log_prob = torch.log(action_probs.squeeze(0)[action_tensor])
                loss = -log_prob * returns[i]
                loss.backward()
            self.optimizer.step()

            print(f"Episode {episode}, Total Reward: {sum(rewards)}")
            #collect rewards for plotting
            rewards_list.append(sum(rewards))
            
        #save the model
        torch.save(self.policy_net.state_dict(), 'saved_models/cartpole_policy_net.pth')

        self.env.close()
        plt.figure()
        plt.plot(rewards_list)
        plt.show()

    def visualize(self, file_path):
        env = gym.make(self.env_name, render_mode='human')
        state_dim = env.observation_space.shape[0]
        action_dim = env.action_space.n

        policy_net = PolicyNet(state_dim, action_dim)
        policy_net.load_state_dict(torch.load(file_path))
        policy_net.eval()

        state,_ = env.reset()
        done = False
        total_reward = 0
        all_states = []
        all_actions = []
        all_rewards = []
        while not done:
            env.render()
            all_states.append(state)
            action = self.select_action(state, policy_net)
            state, reward, done, _, _ = env.step(action)
            total_reward += reward
            all_actions.append(action)
            all_rewards.append(reward)

        env.close()
        print(f"Total Reward: {total_reward}")


        

            
    def collect_data(self,file_path,num_images,image_size_rows):
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n

        policy_net = PolicyNet(state_dim, action_dim)
        policy_net.load_state_dict(torch.load(file_path))
        policy_net.eval()

        states_actions_rewards = []
        ii = 0
        while ii < num_images:
            print(f"Attempting to collect image {ii}")
            state,_ = self.env.reset()
            done = False
            all_states = []
            all_actions = []
            all_rewards = []
            steps = 0
            while not done:
                steps += 1
                all_states.append(state)
                action = self.select_action(state, policy_net)
                state, reward, done, _, _ = self.env.step(action)
                all_actions.append(action)
                all_rewards.append(reward)
                if steps >= image_size_rows+10:
                    done = True
                if done:
                    #erase N most recent states, actions, and rewards
                    N = 10
                    all_states = all_states[:-N]
                    all_actions = all_actions[:-N]
                    all_rewards = all_rewards[:-N]
                

            self.env.close()
            if steps >= image_size_rows+10: #images need to be the same size
                print("Successfully collected image")
                #concatenate all states, actions, and rewards
                #cart position, cart velocity, pole angle, pole angular velocity 
                all_states = np.array(all_states)
                all_actions = np.array(all_actions)
                all_rewards = np.array(all_rewards)
                
                states_actions_rewards.append(np.concatenate([all_states, all_actions.reshape(-1,1), all_rewards.reshape(-1,1)], axis=1))
                print(states_actions_rewards[ii].shape)
                ii += 1 #stored the image and then go on to the next image
            else:
                print("Failed to collect image... retrying:")

        states_actions_rewards = np.array(states_actions_rewards)   
        print(states_actions_rewards.shape)
        #save array
        np.save('saved_state_action_rewards/cartpole_states_actions_rewards.npy', states_actions_rewards)
        print("Saved states, actions, and rewards to file")
    

    
    def collect_random_data(self,num_images,image_size_rows):
        
        state_dim = self.env.observation_space.shape[0]
        action_dim = self.env.action_space.n


        states_actions_rewards = []
        ii = 0
        while ii < num_images:
            print(f"Attempting to collect image {ii}")
            state,_ = self.env.reset()
            done = False
            all_states = []
            all_actions = []
            all_rewards = []
            steps = 0
            for i in range(image_size_rows):
                steps += 1
                all_states.append(state)
                action = np.random.choice(action_dim)
                state, reward, done, _, _ = self.env.step(action)
                all_actions.append(action)
                all_rewards.append(reward)

                if done:
                    all_rewards[-5:] = [0] * min(5, len(all_rewards)) #set last 5 rewards to 0
                    state,_ = self.env.reset()
                
            ii += 1
            self.env.close()
            
            print("Successfully collected random action image")
            #concatenate all states, actions, and rewards
            #cart position, cart velocity, pole angle, pole angular velocity 
            all_states = np.array(all_states)
            all_actions = np.array(all_actions)
            all_rewards = np.array(all_rewards)
            
            states_actions_rewards.append(np.concatenate([all_states, all_actions.reshape(-1,1), all_rewards.reshape(-1,1)], axis=1))
        
        states_actions_rewards = np.array(states_actions_rewards)   
        print(states_actions_rewards.shape)
        #save array
        np.save('saved_state_action_rewards/cartpole_states_actions_rewards_random.npy', states_actions_rewards)
        print("Saved states, actions, and rewards to file")
        




    
# Create the agent
#agent = GymAgent(environment)
#agent.collect_data(file_path,num_images,image_size_rows)