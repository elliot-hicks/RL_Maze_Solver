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
import gym
from maze_maker_package import maze_maker as mm
import numpy as np


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}

    def __init__(self):
        # use of np.copy is key to making shallow copies
        self.agent_position = np.asarray([1, 1])
        maze = mm.build_maze()
        # Insert agent position and goal position.
        # Can be modified for different RL tasks by changing positions:
        maze[1, 1], maze[-2, -2] = 3, 4
        # Create shallow copies:
        self.original_maze = np.copy(maze)
        # Create target state, agent at goal position:
        self.goal_state = np.copy(maze)
        self.goal_state[1, 1], self.goal_state[-2, -2] = 0, 3
        self.old_state = np.copy(maze)
        self.state = np.copy(maze)
        # Limit steps, speeds up training by stopping very long trajectories
        self.max_duration = 300
        self.done = False  # Done flag

    def is_ep_finished(self):
        # if agent performance is too poor or agent solved maze, end episode
        done, success = False, False

        if (self.step_count >= self.max_duration):
            print("\nTIMED OUT:")
            done, success = True, False  # Finshed but not successful
        elif (np.array_equal(self.state, self.goal_state)):
            print("\n###### SUCCESS ###### ")
            done, success = True, True  # Finished and successful
        else:
            done, success = False, False

        return done, success

    def step(self, agent_action):
        """

        Parameters
        ----------
        agent_action : NumPy array
            Vector corresponding to action: e.g. up = [-1,0].

        Returns
        -------
        list
        Returns a TRANSITION, a tuple (mathematical, not literal tuple type
            (state,action,reward,done). Used in training, stored in agent's
            memory buffer.
        """

        self.step_count += 1
        self.old_state = np.copy(self.state)
        # Update agent position and state:
        self.state[self.agent_position[0], self.agent_position[1]] = 0
        self.agent_position += np.asarray(agent_action)
        self.state[self.agent_position[0], self.agent_position[1]] = 3
        self.done, successful = self.is_ep_finished()
        reward = +10 if successful else -1

        return ([self.old_state, agent_action, reward, self.done])

    def reset(self):
        # reset the environment for next episode:
        self.state = np.copy(self.original_maze)
        self.agent_position = np.asarray([1, 1])
        self.step_count = 0
        self.done = False

    def render(self, mode='human', close=False):
        # Print out the maze using matplot.imshow
        mm.show(self.state, self.agent_position)
