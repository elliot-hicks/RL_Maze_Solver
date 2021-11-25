#import packages for maze environments
from maze_maker_package import maze_maker as m
from agent_package import Agent
import gym
import LeNet as LN
import numpy as np

"""
from gym_maze_package import gym_maze
Need to resolve issues with maze_gym before we can import these:
import gym_maze 
env = gym.make('maze-v0')
"""

#import pytorch libraries
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torchvision.transforms as T


def calculate_values(trajectory_rewards, discount_factor):
    for i in range(len(trajectory_rewards)):
        
        if (i == 0):
            continue
        else:
            # disount the future rewards and add them to current value
            trajectory_rewards[-i-1]  += discount_factor*trajectory_rewards[-i]
        
        return trajectory_rewards
    
def loss_fn(model,states, action_labels, values):
    loss = 0
    for i in range(len(values)):
        probability_i = model(states[i])[action_labels[i]]
        loss -= values[i]*np.log(probability_i)
    return(loss)
        

def train(maze_env, model,number_of_episodes, discount_rate, optimiser):
    # discount rate is tha gamma in most RL formulas, used to encourage efficiency
    # exploration factor gives the proportion of episodes where the agent explores
    exploration_period = 0.1*number_of_episodes
    agent = Agent(maze_env.maze,starting_epsilon = 0.9,memory_buffer_size = 1000)
    last_episode = []
    for episode_number in range(number_of_episodes):
        steps = 0
        total_transitions = 0
        episode_done = False
        while not episode_done:
            action_probabilities = model(maze_env.state)
            action_probabilities = agent.test_actions(action_probabilities,maze_env.state) # set invalid action probs to zero, renomalise
            action, action_label = agent.choose_action(action_probabilities)
            agent.position += action
            agent.update_epsilon(number_of_episodes, episode_number)
            
            
            #should exploration probabilities be 1/3 or the actual prob? should equal prob from NN 
            state_before, action, state_after, reward, episode_done = maze_env.step(action)
            agent.replay_buffer.add([maze_env.state,action_label,state_after,reward])
            
            
            
            steps +=1
            total_transitions+=1
            if (episode_done):
                
                last_episode = agent.replay_buffer[-steps:,1] #list of actions in ep
                
                #backprop and calculate state values for trajectory
                trajectory_values = calculate_values(agent.replay_buffer[-steps:,3])
                agent.replay_buffer[-steps:,-3] = trajectory_values

                
        if ((total_transitions>exploration_period) and (episode_number % 10 == 0 )):
            #train CNN after every 10 episodes
            
            training_batch = agent.memory_buffer.r_sample()
            states = training_batch[:,0]
            action_labels = training_batch[:,1]
            values = training_batch[:,2]
            loss = loss_fn(states,action_labels,values)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
                
            
    return last_episode
            
            
def maze_solver():
    maze_env = gym.make('maze-v0')
    starting_position = [1,1]
    memory_size = 1000
    learning_rate = 1e-3
    elite_memory_size = 0.1*memory_size
    agent = Agent(maze_env, starting_position, memory_size, elite_memory_size)
    model = LN.LeNetCNN()
    ADAM = torch.optim.ADAM(model.params, lr  = learning_rate)
    
    final_episode = train(maze_env,model, number_of_episodes=1000, discount_rate = 0.95, optimiser = ADAM)
    agent.replay(final_episode)# animate the actions of the agent in final episode
