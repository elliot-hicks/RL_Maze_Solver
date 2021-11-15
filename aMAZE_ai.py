import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import namedtuple
import random

#import packages for maze environments
from maze_maker_package import maze_maker as m
import gym
"""
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

Experience = namedtuple('Experience','old_state action new_state initial_reward final_reward')
# use of final_reward is for the elite buffer, it may be removed later


class ExperienceBuffer():
    """
    Experience buffer is bascially a short-term memory for a netural network
    from which it can select random samples to learn from and stabalize learning
    Deques are used here such that when we hit capacity, the network 
    'forgets' the older experiences, and hopefully become filled with
    experiences of 'good steps'.
    """
    def __init__(self, capacity):
        """
        Parameters
        ----------
        capacity : int
            The capacity of the networks 'memory' 
        
        """
        self.capacity = capacity
        self.memory_buffer = collections.deque(maxlen = capacity)
        self.size = len(self.memory_buffer) # current size of the memory buffer
    
    def add(self, experience):
        self.memory_buffer.append(experience) #add to right of buffer
        
    def random_sample_batch(self, batch_size = 100):
        """
        Parameters
        ----------
        batch_size : int,
            Hyperparameter: how many experiences we should look at during the 
            batch training, introduced to stabalize training.The default is 100.
        Returns
        -------
        batch: list of Experiences
        """
        
        batch = random.sample(self.memory_buffer,batch_size)
        return batch
            

class Agent:
    def __init__(self, maze,starting_epsilon,buffer_size, elite_buffer_size = 100):
        self.epsilon = starting_epsilon
        self.experience_buffer = ExperienceBuffer(elite_buffer_size)
        self.elite_experience_buffer = ExperienceBuffer(elite_buffer_size)
        # we use deques for more efficient appends and size capping
        self.actions = {"up":[-1,0],"right":[0,1],"down":[1,0],"left":[0,-1]}
        
        
        #initialise nets
        
    def choose_exploration(self):
        #use epsilon-greedy for eploration vs exploitation trade off
        return True if np.random.uniform(0,1)<self.epsilon else False
             
    def apply_action(self, action):
        #new_state = agent.apply_action(action)
        #apply action to state
        #actions are 1x2 list so we can add them to the state
        return (self.state + np.asarray(action)) 
    
    def update_state(self, current_state):
        old_state = self.state
        self.state= current_state
        print(current_state)
        self.environment[old_state[0],old_state[1]] = 0
        self.environment[current_state[0],current_state[1]] = 2
        
    def is_finished(self):
        print("GOAL:",self.goal_state)
        print("State:",self.state)
        return True if np.array_equiv(self.state,self.goal_state) else False
    
    def show_state(self):
        plt.axis("off")
        plt.imshow(self.environment)
        
    """ 
    def initialise_main_network():
        # code for main net 
        #self.main net = q
        
    def intitialsie_target_network():
        #code for tgt net
        # self.target_net = Q
    
    def update_target_network():
        # target_network weights -> main net weights
    
    def update_main_network():
        #update the main network using random sampled batches
        # from experience buffer. do not delete buffer after update
        
    """
    
maze = m.build_maze()
Agent = Agent(maze,0.9,10)
new_state = Agent.apply_action(np.array([1,0]))
Agent.update_state(new_state)
Agent.show_state()
new_state = Agent.apply_action(np.array([1,0]))
Agent.update_state(new_state)
Agent.show_state()

"""

pseudo code for Agent solving maze using memory buffer and two NNs 
- possible alt is Q-table if this is too hard

def run_maze(Agent):
    N=10000
    episodes = range(1,N)
    for episode in epsiodes:
        current_state = Agent.state
        if choose_exploration(Agent.epsilon):
            # new_sate,reward = Agent.apply_action(random_choice(actions))
        else:
            # current_state  = Agent.state
            # best_action = MAXa(Q_main_NN(current_state,a))
            # new_state, reward = Agent.apply_action(best_action)
        # finished = Agent.is_done()
        # experience = [Agent.environment,action,reward,finished,new_state]
        # Agent.experience_buffer.append(experience)
        
        
        if buffer filled: #start learning
        
             mini_batch = random.sample(Agent.experience_buffer)
            for exp in mini_batch:
                if experience.finished == True:
                    Q_value(old_state,action) = exp.reward # this will back prop through the Qfunc
                else:
                    Q_value(old_state,action) = exp.reward + gamma*MAXAa'(Q_target_NN(new_state,a'))
                fit Q_main_NN(s,a) to Q_value(s,a) using MSE or whatever
            
            if (episode%100 == 0):
                Q_target_NN_params = Q_main_nn_params
                #update the target NN to the value of the main NN
            
        
        
#TO DO LIST:
    
    when does an episode end?
    look at how we optimize the NNs?
    how should we penalise the moves out of the maze?
    how should we choose the params of the NNs?
    input the state as a list to NN
    
    
    """
        
        
        
        
        
        
            
