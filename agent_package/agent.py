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


class EliteExperienceBuffer(ExperienceBuffer):
    """
    The elite memory class intuituvely inherits a lot of ExperienceBuffer
    properties, so it is inherited. Needs access to the agent that owns it's
    ExperienceBuffer to access memories so is passed a reference to the agent
    in __innit__.
    """

    def __init__(self, agent, capacity):
        super().__init__(capacity)
        self.agent = agent

    def add_last_episode(self, n_steps):
        # Add the last n steps (n = steps), to the elite buffer
        if self.is_full():
            if n_steps < max(self.agent.elite_steps):
                add_new_elite_memory = True
            else:
                add_new_elite_memory = False
        else:
            add_new_elite_memory = True

        if add_new_elite_memory:
            print("##### SAVED AS ELITE MEMORY ##### ")
            self.agent.elite_steps.append(n_steps)
            for i in range(1, n_steps + 1):
                self.add(self.agent.replay_buffer.memory_buffer[-i, :])


class Agent:

    def __init__(self, maze, exploration_period,
                 starting_epsilon,
                 buffer_size,
                 elite_buffer_size=600):
        self.position = np.array([1, 1])
        self.environment = maze
        self.exploration_period = exploration_period
        self.epsilon = starting_epsilon
        self.epsilon_lower_bound = starting_epsilon / 100
        self.replay_buffer = ExperienceBuffer(buffer_size)
        self.elite_experience_buffer = EliteExperienceBuffer(self, elite_buffer_size)
        self.elite_steps = []
        # we use deques for more efficient appends and size capping
        self.action_space = np.array([[-1, 0], [0, 1], [1, 0], [0, -1]])
        self.action_space_labels = np.array([0, 1, 2, 3])

    def test_actions(self, action_probabilities):
        """
        Removed possibility of agent stepping in to walls, speeds up training
        as the CNN has one less thing to learn and the rewards function
        can be made simply, with no tuning of rewards for wall steps needed.
        """
        for action in range(4):
            test_position = self.position + self.action_space[action, :]
            # Test if new position is a wall (val = -1):
            if (self.environment[test_position[0], test_position[1]] == -1):
                action_probabilities[action] = 0
        normed_probabilities = normalise(action_probabilities)  # Renormalize
        return (normed_probabilities)

    def choose_action(self, probabilities):
        """
        Parameters
        ----------
        probabilities : NumPy array (floats)
            Porbabilties perscribed by policy function corresponding to the
            agent actions. Implemented after action testing, exmploys
            epsilon-greedy approach.
        Returns
        -------
        random_(policy_)action: action chosen by exploration(explotation)
        random_(policy_)action_label: encoding corresponding to action:
            0,1,2,3 = up,right,down,left

        """
        # Get policy action:
        policy_action_label = np.random.choice(self.action_space_labels,
                                               p=probabilities)
        policy_action = self.action_space[policy_action_label]
        if np.random.uniform(0, 1) < self.epsilon:
            probabilities[policy_action_label] = 0  # Policy not followed
            probabilities = normalise(probabilities)  # Renormalize
            try:
                random_action_label = np.random.choice(self.action_space_labels,
                                                       p=probabilities)
                random_action = self.action_space[random_action_label]
                return random_action, random_action_label
            except ValueError:
                """
                If only one action valid, the action space will be the policy
                action, after removing policy action in epslion greedy step,
                all the probabilties would be zero, causing a 0 division error.
                """
                return policy_action, policy_action_label
        else:
            return policy_action, policy_action_label

    def update_epsilon(self, episode, number_of_episodes):
        # Epsilon should stay at 1 for exploration period:
        if (episode < self.exploration_period):
            pass
        # Epsilon should not go below lower bound of 1%
        elif (self.epsilon < self.epsilon_lower_bound):
            self.epsilon = self.epsilon_lower_bound
        # Decrease epsilon by 10% every 10 eps past exploration period:
        elif ((episode-self.exploration_period) % 10 == 0):
            print("\nEpsilon reduced.")
            self.epsilon *= 0.9
