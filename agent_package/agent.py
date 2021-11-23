import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import namedtuple
import random

Experience = namedtuple('Experience','old_state action new_state initial_reward final_reward action_probability')
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
        
    def r_batch(self, batch_size = 100):
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
        self.action_space = [[-1,0],[0,1],[1,0],[0,-1]]
        
        #initialise net
    """
    def choose_step(self, action_index):
        # we give the agent the action, we let it choose whether to follow it
        
        if np.random.uniform(0,1)<self.epsilon:
            randmom_index = random.randint(0,3)
            action = self.action_space.remove(self.action_space[action_index])[random_index]
            # choose an action at random from the action space with the predicted action exlcuded
            probability = probabilities.remove(probabilities[action_index])[random_index]
            # find the corresponding probability for that action
        return action, probability
    
    def update_epsilon(number_epsiodes, episode):
        #epsilon should start high and then start to decrease 
        expoloration_period= 0.1*number_episodes
        if (episode < exploration_period):
            self.epsilon = self.epsilon 
            # keep epsilon constant for first 10% of runs to encourage exploration early on
        elif (self.epsilon < self.epsilon_lower_bound):
            self.epsilon = self.epsilon_lower_bound
        elif (episode-exploration_period % 100 == 0):
            self.epsilon *= 0.95**((episode-exploration_period)/100)
            # reduce epsilon by 5% every 100 episodes


To do list:
    Design the algorithm for training with just batches, no elite buffers
    Learn the syntax and structres of pytorch networks and what they ouput
    implement ADAM optimizer
    decide if these should be in the agent class or in their own file.     



"""
    
    
        
        
        
        
        
            
