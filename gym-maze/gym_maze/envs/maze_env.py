
import gym # I had to use pip install instead of conda to get gym, do you know why?
from gym import error, spaces, utils
from gym.utils import seeding
import maze_maker as mm
import numpy as np


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    def __init__(self):
        self.agent_positon = np.asarray([1,1])
        self.minimum_threshold = -1
        maze = mm.build_maze(5,5)
        self.rewards_map = maze # maze without the agents position marked
        maze[1,1], maze [-2,-2] = 3, 4 # insert agent position and goal state
        self.original_state = maze # never edited
        self.old_state = maze # updated constantly
        self.state = maze
        goal_state = maze
        goal_state[1,1], goal_state[-1,-1] = 0, 3 
        # copy of maze with agent position at end of maze
        self.goal_state = goal_state
        self.done = False
        
    def step(self, agent_action, agent_sore):
        # calc get rewards etc for state transitions
        self.old_state = self.state
        self.state[self.agent_positon[0],self.agent_position[1]] = 0  # erase old agent position 3->0
        self.agent_position += np.asarray(agent_action) # actions written as vectors
        self.state[self.agent_positon[0],self.agent_position[1]] = 3  # flag new agent position x->3
        self.done = is_ep_finished(agent_score)
        reward = calc_reward(agent_position)
        
        return ([self.old_state, action, self.state, reward])
        
    def reset(self):
        # reset the env
        self.state = self.old_state = original_maze 
        self.agent_position = np.asarray([1,1])
        self.done = False
        
    def render(self, mode='human', close=False):
        mm.show(self.state) # print out the maze using matplot.imshow
        
    def is_ep_finished(self, agent_score):
        # if agent performance is too poor or agent solved maze, end episode
        if ((agent_score <= self.minimum_threshold) or (self.state == self.goal_state)):
            return True
        else:
            return False
    
    def calc_reward(self, agent_position):
        #use rewards map to tell us what the reward for the step was:
            # wall = -1, free = 0, goal state = +1
        reward = self.rewards_map[agent_position[0], agent_position[1]]
        if (reward != +1):
            reward -= 0.05 # penalise steps - encourage efficiency
            return reward
        else:
            return reward
            
        