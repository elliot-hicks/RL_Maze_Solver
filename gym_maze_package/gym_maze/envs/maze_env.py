
import gym 
from gym import error, spaces, utils
from gym.utils import seeding
from maze_maker_package import maze_maker as mm
import numpy as np


class MazeEnv(gym.Env):
    metadata = {'render.modes': ['human']}
    
    def __init__(self):
        # use of np.copy is key to avoiding the differnent state vars being updated
        self.agent_position = np.asarray([1,1])
        maze = mm.build_maze(5,5)
        maze[1,1], maze [-2,-2] = 3, 4 # insert agent position and goal position
        self.original_maze = np.copy(maze) # never edited
        self.old_state = maze # updated constantly
        self.state = maze
        self.goal_state = np.copy(maze)
        self.goal_state[1,1], self.goal_state[-2,-2] = 0, 3 
        # copy of maze with agent position at end of maze
        self.max_duration = 100 # max number of steps in one episode, avoids infinite loops
        self.step_count = 0
        self.done = False
        
    def is_ep_finished(self):
        # if agent performance is too poor or agent solved maze, end episode
        if ((self.steps > self.max_duration) or (np.array_equal(self.state,self.goal_state))):
            return True
        else:
            return False
        
    def calc_reward(self):
        # penalise every step that doesnt end in goal state with -1
        return +1 if np.array_equal(self.state,self.goal_state) else -1
        
    def step(self, agent_action):
        # calc get rewards etc for state transitions
        self.old_state = self.state
        self.state[self.agent_position[0],self.agent_position[1]] = 0  # erase old agent position 3->0
        self.agent_position += np.asarray(agent_action) # actions written as vectors
        self.state[self.agent_position[0],self.agent_position[1]] = 3  # add new agent position x->3
        self.done = self.is_ep_finished()
        reward = self.calc_reward() 
        self.step_count+=1
        return ([self.old_state, agent_action, self.state, reward])
        
    def reset(self):
        # reset the env
        self.state = self.old_state = self.original_maze 
        self.state[self.agent_position[0],self.agent_position[1]] = 0  # erase old agent position 3->0
        self.agent_position = np.asarray([1,1])
        self.step_count = 0
        self.done = False
        
    def render(self, mode='human', close=False):
        mm.show(self.state, self.agent_position) # print out the maze using matplot.imshow
        
        
