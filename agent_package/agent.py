"""
I, Elliot Hicks, have read and understood the School's Academic Integrity
Policy, as well as guidance relating to this module, and confirm that this
submission complies with the policy. The content of this file is my own
original work, with any significant material copied or adapted from other
sources clearly indicated and attributed.
Author: Elliot Hicks
Project Title: RL_CNN_maze_solver
Date: 13/12/2021
"""

import numpy as np


def normalise(vector):
    return(vector / (sum(vector)))


class ExperienceBuffer():

    def __init__(self, capacity):
        self.capacity = capacity
        self.memory_buffer = np.ndarray((0, 4))
        self.size = len(self.memory_buffer)  # current memory buffer size

    def __getitem__(self, index):
        # Override index operator to return slice of memory_buffer
        return self.memory_buffer[index]

    def is_full(self):
        return self.size >= self.capacity

    def add(self, experience):
        # Mimic deque functionality, ndarray with a capacity.
        if (self.size == self.capacity):
            # if at capacity, remove first entry
            self.memory_buffer = np.delete(self.memory_buffer, (0), axis=0)
        else:
            self.size += 1
        self.memory_buffer = np.vstack((self.memory_buffer, experience))

    def update_values(self, episode_steps, values):
        """
        Before backpropagation, rewards are updated for the last episode using:

            R_{t'} = SUM_{t = t'}^{T}(r_t*discount_factor^{t'-t})

        This function updates the memory buffer's rewards to these values for
        the corresponding episode:
        """
        self.memory_buffer[-episode_steps:, 3] = values

    def random_batch(self, batch_size=600):
        """
        Parameters
        ----------
        batch_size : int,
            Hyperparameter: how many transitions used in each batch training of
            CNN, introduced to stabalize training.
        Returns
        -------
        batch: list of Experiences
        """

        rand_indices = np.random.choice(self.size,
                                        size=batch_size,
                                        replace=False)
        batch = self.memory_buffer[rand_indices, :]
        return batch


class Agent:
    def __init__(self, maze,exploration_period, starting_epsilon,buffer_size, elite_buffer_size = 600):
        self.position = np.array([1,1])
        self.environment = maze
        self.exploration_period = exploration_period
        self.epsilon = starting_epsilon
        self.epsilon_lower_bound = starting_epsilon/100
        self.replay_buffer = ExperienceBuffer(buffer_size)
        self.elite_experience_buffer = ExperienceBuffer(elite_buffer_size)
        self.elite_steps = []
        # we use deques for more efficient appends and size capping
        self.action_space = np.array([[-1,0],[0,1],[1,0],[0,-1]])
        self.action_space_labels = np.array([0,1,2,3])
        
    def test_actions(self,action_probabilities):
        #remove possibility to step in to wall
        for action in range(4):
            test_position = self.position + self.action_space[action,:]
            #test if new position is a wall (val = -1)
            if (self.environment[test_position[0],test_position[1]] == -1):
                action_probabilities[action] = 0   
        #renormalise
        normed_probabilities = normalise(action_probabilities)
        return (normed_probabilities)              
    
    def choose_action(self, probabilities):
        # implement epsilon-greedy,explotration prob of self.epsilon
        policy_action_label = np.random.choice(self.action_space_labels, p = probabilities)
        policy_action = self.action_space[policy_action_label]
        if np.random.uniform(0,1)<self.epsilon:
            probabilities[policy_action_label]=0 # agent must choose alt step
            probabilities = normalise(probabilities)
            try:
                random_action_label = np.random.choice(self.action_space_labels, p = probabilities)
                random_action = self.action_space[random_action_label]
                return random_action, random_action_label
            except ValueError:
                # probabilities are NANs if all zeros after removing policy's choice
                # this happens when only one action is valid
                return policy_action, policy_action_label
        else:    
            return policy_action, policy_action_label
    
    def update_epsilon(self,episode, number_of_episodes):
        #epsilon should start high and then start to decrease 
        if (episode < self.exploration_period):
            pass
            # keep epsilon constant for first 10% of runs to encourage exploration early on
        elif (self.epsilon < self.epsilon_lower_bound):
            self.epsilon = self.epsilon_lower_bound
        elif ((episode-self.exploration_period) % 50 == 0):
            print("epsilon reduced")
            self.epsilon *= 0.9#(number_of_episodes-episode)/number_of_episodes
            # reduce epsilon by 5% every 100 episodes
    
    def update_elite_buffer(self,n_steps):
        #add the last n steps (n = steps), to the elite buffer from the memory buffer
        if self.elite_experience_buffer.is_full():
            if n_steps<max(self.elite_steps):
                add_new_elite_memory = True
            else:
                add_new_elite_memory = False
        else:
            add_new_elite_memory = True
        
        if add_new_elite_memory:
            print("Elite memory found")
            self.elite_steps.append(n_steps)
            for i in range(1,n_steps+1):
                self.elite_experience_buffer.add(self.replay_buffer.memory_buffer[-i,:])
            