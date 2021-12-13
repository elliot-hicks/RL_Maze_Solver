#import packages for maze environments
from agent_package import agent as a
from maze_maker_package import maze_maker
from LeNet_package import LeNet_CNN as LN
from CNN10 import CNN10 as CN
import gym
from gym_maze_package import gym_maze
env = gym.make('maze-v0')

import torch
import numpy as np
import matplotlib.pyplot as plt
#import pytorch libraries

def calculate_values(trajectory_rewards, discount_factor):
    i = 1  
    while i < len(trajectory_rewards):       
        # disount the future rewards and add them to current value
        trajectory_rewards[-i-1]  += discount_factor*trajectory_rewards[-i]
        i+=1
    print("Rewards mean:", np.mean(trajectory_rewards))
    return trajectory_rewards
    
def loss_fn(model,states, action_labels, values):
    
    loss = 0
    for i in range(len(values)):
        probability_i = model(states[i])
        #loss -= torch.mul(torch.log(probability_i[0][action_labels[i]]),values[i])
        loss -= torch.mul(torch.log(probability_i[action_labels[i]]),values[i])
    print(loss)
    return(loss)    

def train(maze_env, model,number_of_episodes, discount_factor, optimiser):
    # discount rate is tha gamma in most RL formulas, used to encourage efficiency
    # exploration factor gives the proportion of episodes where the agent explores
    buffer_size = 3000
    exploration_period = 200
    agent = a.Agent(maze_env.original_maze,exploration_period, starting_epsilon = 1, buffer_size = buffer_size)
    last_episode = []
    episode_number = 0
    
    while episode_number <number_of_episodes :
        print(episode_number)
        maze_env.reset() # restart maze
        agent.position = np.array([1,1])
        
        total_transitions = 0
        episode_done = False
        while not episode_done:
            action_probabilities = model(maze_env.state).detach().numpy()#[0]
            action_probabilities = agent.test_actions(action_probabilities) # set invalid action probs to zero, renomalise
            action, action_label = agent.choose_action(action_probabilities)
            agent.position += action
            state, action, reward, episode_done = maze_env.step(action) 
            
            agent.replay_buffer.add([state,action_label,reward, episode_done])
            total_transitions+=1
            if (episode_done):
                
                print("Episode: ", episode_number)
                print("Epsilon:", agent.epsilon)
                print("Steps: ", maze_env.step_count)
                episode_steps.append(maze_env.step_count)
                #last_episode = agent.replay_buffer[-steps:,1] #list of actions in ep
                #backprop and calculate state values for trajectory
                
                
                trajectory_values = calculate_values(agent.replay_buffer[-maze_env.step_count:,2], discount_factor)
                episode_av_reward.append(np.mean(trajectory_values))              
                agent.replay_buffer.update_values(maze_env.step_count,trajectory_values)
                
                if (maze_env.step_count <np.percentile(episode_steps,10)):
    
                    print(maze_env.step_count)
                    agent.update_elite_buffer(maze_env.step_count)
                    print(agent.elite_experience_buffer.size)
                
                
                
        agent.update_epsilon(episode_number,number_of_episodes)
        exploration_over = episode_number>=exploration_period
        if (exploration_over and (episode_number % 5 == 0 )):
            #train CNN after every 10 episodes
            training_batch = agent.replay_buffer.random_batch()
            states = training_batch[:,0]
            action_labels = training_batch[:,1]
            values = training_batch[:,2]
            loss = loss_fn(model,states,action_labels,values)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        if (agent.elite_experience_buffer.is_full() and exploration_over and (episode_number % 10 == 0 )):
            #train CNN after every 10 episodes
            training_batch = agent.elite_experience_buffer.random_batch(400)
            states = training_batch[:,0]
            action_labels = training_batch[:,1]
            values = training_batch[:,2]
            loss = loss_fn(model,states,action_labels,values)
            optimiser.zero_grad()
            loss.backward()
            optimiser.step()
        #maze_env.reset()
        episode_number+=1
    
    return(episode_steps, episode_av_reward)
   #
            
def maze_solver():
    
    maze_env = gym.make('maze-v0')
    learning_rate = 1e-4
    model = CN.ECNN10(1,4).float()
    ADAM = torch.optim.Adam(model.parameters(), lr  = learning_rate)
    steps = train(maze_env,model, number_of_episodes=2000, discount_factor = 0.95, optimiser = ADAM)
    #print(final_episode)
    #agent.replay(final_episode)# animate the actions of the agent in final episode
    return(steps)
episode_steps = []
episode_av_reward = []
steps, rewards = maze_solver()
#


#COPIED FROM ONLINE REMOVE OR REF
def moving_average(a, n=20) :
    ret = np.cumsum(a, dtype=float)
    ret[n:] = ret[n:] - ret[:-n]
    return ret[n - 1:] / n


plt.plot(moving_average(rewards,1))