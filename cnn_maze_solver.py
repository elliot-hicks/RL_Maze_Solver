#import packages for maze environments
from maze_maker_package import maze_maker as m
from agent_package import agent 
import gym
import LeNet as LN

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


"""
TRAINING ALGORITHM PSEUDOCODE

def train(maze_env, model,number_episodes, discount_rate, optimiser):
    # discount rate is tha gamma in most RL formulas, used to encourage efficiency
    # exploration factor gives the proportion of episodes where the agent explores
    last_episode = []
    for episode in range(number_episodes):
        steps = 0
        total_transitions = 0
        episode_done = false
        while not episode_done:
            net_action_index, action_probabilties = argmax(model(state)), model(state)
            agent_action_index = agent.choose_step(action)
            action_probability = probabilities[agent_action_index] 
            #should exploration probabilities be 1/3 or the actual prob?
            state_before, action, state_after, reward, episode_done = agent.take_step(action)
            reward *= (discount_rate**steps)
            agent.memory_buffer.add([state,action,state_after,reward,probability])
            steps +=1
            total_transitions+=1
            
            if (episode_done):
                last_episode = memory_buffer[-steps:,3]
                total_reward = sum(memory_buffer[-steps:,3])
                agent.memory_buffer[-steps:,3] = total_reward
            if ((total_transitions-eploration_period)%100 == 0):
                training_batch = agent.memory_buffer.r_sample(batch_size)
                episode_rewards, probabilities = training_batch[3], training_bactch[4]
                loss = loss_fn(episode_rewards, probabilities)
                #we need an optimiser that works on final episode reward and probabilities in episode
                optimiser.zero_grad()
                loss.backward()
                optimiser.step()
                
                
                
                
                
                
            else:
                pass
    return last_episode
            
            
def maze_solver():
    maze_env = gym.make('maze-v0')
    starting_position = [1,1]
    memory_size = 1000
    learning_rate = 1e-3
    elite_memory_size = 0.1*memory_size
    agent = Agent(maze_env, starting_position, memory_size, elite_memory_size)
    model = LN.LeNetCNN()
    optimiser = torch.optim.ADAM(model.params, lr  = learing_rate)
    
    final_episode = tain(maze_env,model, number_episodes, discount_rate, optimiser)
    agent.replay(final_episode)# animate the actions of the agent in final episode




"""



