import numpy as np
import matplotlib.pyplot as plt
import collections
from collections import namedtuple
import random

Experience = namedtuple('Experience','old_state action_label new_state values ')
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
        self.position = np.array([1,1])
        self.environment = maze
        self.epsilon = starting_epsilon
        self.replay_buffer = ExperienceBuffer(buffer_size)
        self.elite_experience_buffer = ExperienceBuffer(elite_buffer_size)
        # we use deques for more efficient appends and size capping
        self.action_space = np.array([[-1,0],[0,1],[1,0],[0,-1]])
        self.action_space_labels = np.array([0,1,2,3])
        
        #initialise net
    """
    
    
    def test_actions(action_probabilities):
        #remove possibility to step in to wall
        for (action in range(4)):
            test_position = agent.position + action_space[action,:]
            if (maze[test_position[0],test_position_[1]] == -1):
                     action_probabilities[action] = 0   
        #renormalise
        normed_probabilities = probabilities/(sum(probabilities**2))**0.5
            
        return (normed_probabilities)
                     
    
    def choose_action(self, probabilities):
        # implement epsilon-greedy,explotration prob of self.epsilon
        policy_action_label = np.random.choice(action_space_labels,probabilities)
        
        if np.random.uniform(0,1)<self.epsilon:
            reduced_action_space_labels = action_space_labels.remove(policy_action_label)
            reduced_probabilities = probabilities.remove(probabilities[policy_action_label])  
            random_action_label = np.random.choice(reduced_action_space_lables, reduced_probabilities)
            random_action = action_space[random_action_label]
            return random_action, random_action_label
        else:    
            return policy_action, policy_action_label
    
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
    
    
        
        
        
        
        
            
