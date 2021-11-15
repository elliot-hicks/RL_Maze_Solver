[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6094273&assignment_repo_type=AssignmentRepo)
# Scientific Programming in Python – submission 2

## Project title: RL_Maze_Solver

## Student name: Elliot Hicks 
## ID: 20321974
  
## Table of Contents
1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Roadmap](#roadmap)
5. [Similar Work](#similar-work)
6. [References](#references)
7. [Contact](#contact)

  
# Description

RL_Maze_Solver will demonstrate the ability of reinforcement learning (specifically Q-learning) by randomly generating 2D mazes and tasking an agent to find its most efficient solution. Mazes are made with the Maze_maker.py package I wrote and allow for 2D mazes with multiple routes to be made. I'm currently trying to create a strong grounding in supervised learning algorithms from a few books listed in the references. 

# Installation
Information on packages/ environments will be made available here:

Required Packages:

* **NumPy**
* **Random**
* **Matplotlib.pyplot**
* **maze_maker**:
* **gym_maze_package**
* **collections**
* **gym**
* **Pytorch**

# Usage
# Maze_maker.py

This is a basic example of the mazes made by ```Maze_maker.py```, a 20x30 maze where the agent will start at [0,0] (top-left) and aims to find the exit at [-1,-1] (bottom-right).
Note: ```Maze_maker.py``` was updated from a random walk generator to a recursive maze generator, this makes far more difficult mazes.

![Maze_maker.py original design](https://github.com/mpags-python/coursework2021-sub2-elliot-hicks/blob/main/plot_example_20x30.png)
![Maze_maker.py new design](https://github.com/mpags-python/coursework2021-sub2-elliot-hicks/blob/main/new_plot_example_20x30.png)

## Maze production
```Maze_maker.py``` is a module made to create 2D mazes of any shape. 
The recursive algorithm that inspired this method can be found here: [wiki recursive](https://en.wikipedia.org/wiki/Maze_generation_algorithm#Recursive_implementation).

An exmple of the recursive process is shown below:

![recursive mase generation](https://upload.wikimedia.org/wikipedia/commons/9/90/Recursive_maze.gif)


Key functions:

1.```recursive_maze(maze)```:
   
* Parameters:

  * maze (2D NumPy array):
  * maze is spit in four quadrants using one horizontal wall and one vertical wall.
  * walls have gates added at random
  * each quadrant then is treated as a maze and fed back in to function this recursively builds up the maze

* Returns:
    * maze NumPy array that descibes the maze as a matrix of ints:
    * 0 = steppable space
    * 1 = wall
    * 2 = vertical gate (hole in vertical wall)
    * 3 = horizontal gate (hole in horizontal wall)
    * Gates are highlighted so we can remove any obstructions made during the maze generation

2.```finalise_maze(maze)```:

* Parameters

  * maze (2D NumPy array): Sometimes added walls obstruct gates, so we use the encoding
    from the recursive maze generator to find gates and clear space 
    around them as necessary. Final prodcut is a maze where all points
    are accessible.

* Returns

  * maze (2D NumPy array): returns finalised array
  * Code snipppet below shows the process of removing obstructions around gates
    * gates in horizontal walls (value = 2) must have the entrance cleared above, below and inside gate.
    * gates in vertical walls (value = 3) must have the entrance cleared left, right and inside gate.
  * this final step made the code much more readable as introducing obstruction clearing in the maze generator made the code undreadable.
```python
for row in range(len(maze[:,0])):
    for col in range(len(maze[0,:])):
        if(maze[row,col] == 3):
            maze[row+1,col], maze[row,col], maze[row-1,col] = 0,0,0
        elif(maze[row,col] == 2):
            maze[row,col-1], maze[row,col], maze[row,col+1] = 0,0,0
        else:
            pass
return maze
```
## Maze Visualisation:

Mazes are currently visualised using [matplotlib.pyplot.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html) from the matplotlib library. Values for pixels are:
* Tunnell: 0
* Wall: 1
* Start ([0,0], top left): 3 
* Goal ([-1,-1], bottom right): 4
```python
def show(maze):
    #formatting image
    maze[-1,-1] = 4 
    maze[0,0] = 3 
    plt.axis('off')
    plt.imshow(maze)    
```

![Maze_maker.py new design](https://github.com/mpags-python/coursework2021-sub2-elliot-hicks/blob/main/new_plot_example_20x30.png)

## maze_maker Status:
The maze_maker file is now finished, I dont predict any changed needing to be made however some useful extensions are
anticipated.
All possible improvements are listed below:
1. Add pickle or start using seeds, veering towards use of pickle so i can also save NN parameters.
2. Add exception handling for stupid maze inputs


# aMAZE_ai.py:

## ```Agent```:
* **```aMAZE_ai.py```** will contain an Agent class that will solve a given maze from the **```maze_maker```** library using Q-learning techniques.
* The agent class was revised following the introduction of the ```Gym``` library:

## ```ExperienceBuffer```:
In order to train a CNN to solve these mazes, I want to give it a short-term memory, this comes in the form of the 
ExperienceBuffer class. This is essentially a deque from *collections* which is an array with a fixed capacity. 
Any transitions made by the Agent class will be recorded in the Experience buffer. To fill the buffer with experiences
a *namedtuple* (also from *collections*) has been introduced:
```python 
Experience = namedtuple('Experience','old_state action new_state initial_reward final_reward')
```
Here we add where the agent was, its action, the resulting state, the intial reward from the action and the final reward.
the final reward corresponds to the score of the agent at the end of the episode, its key to creating [elite buffers](#elite buffers).

The ExperienceBuffer class has (currently) three main attributes:
1. capacity (int): How many experiences the deque can hold, this is analogous to how short or long the networks memory is.
2. memory_buffer (deque): the actual deque for experiences with maxlen = capacity,
3. size (int): the length of the memory_buffer, once size = capacity, training can start.

The ExperienceBuffer has three main methods:
1. ```__innit__```:
```python 
def __init__(self, capacity):
      self.capacity = capacity
      self.memory_buffer = collections.deque(maxlen = capacity)
      self.size = len(self.memory_buffer) # current size of the memory buffer 
```
2. ```add```:
```python 
def add(self, experience):
      self.memory_buffer.append(experience) #add to right of buffer
```
3. ```random_sample_batch```
```python 
def random_sample_batch(self, batch_size = 100):
        batch = random.sample(self.memory_buffer,batch_size)
        return batch
```
## Elite buffers:
In order to improve training of the CNN, many training regimens employ an 'elite buffer', a buffer of transitons/experiences
that were part of particularly good episodes (say the top 10% of all final scores). The memories in this buffer are
protected, their existence in the CNNs memory extended, so that they can reinforce its behaviour for a longer amount of
time. 

I havent decided yet if I will implement this. It is of course a great addition, however I want to solve employing the 
normal batch learning before introducing more complex models like this.

## Use of Pytorch:

I plan to use a CNN from the pytorch library to solve the maze, a single CNN will be used. The current aMAZEai package 
shows a pseudocode for the implementation of two NNs, a main and a target NN, while this would produce more stable training, it seemed
overcomplicated. 

### aMAZE_ai status:
With most of the maze environment building complete, bar some installation bugs, the aMAZE_ai file was able to take form.
aMAZEai now contains the Experience and ExpereienceBuffer objects which can be used in the training of the CNN. 
With the maze environment taking on most of the functionality of the Agent, we can now focus on the training algorithm, 
the most exciting part.

# openAI/Gym custom gym environment: 'maze_env':

In order to make the generated maze from maze_maker a gym-environment for the pytorch library I had to learn to create my own custom 
environments for OpenAi's Gym package. I did this using A. Poddars article [3]. Following this i have attempted
to create one with the following file structure, however im still new to making packages so I expect some issues, particularly
with the use of ```maze_maker.py``` as it is several directories above the maze_env file, the current file structure is:
``` 
maze_maker_package:
  > __innit__
  > maze_maker.py
gym_maze_package:
  >README.md
  >setup.py
  >gym_maze:
    >__init__.py
    >envs:
      >maze_env.py
      >__innit__.py
```

This is mostly resolved, however I'm now struggling to install the custom environment to the gym package. My reference
guide [3] suggests using :
```python 
pip install -e .
```
within the gym_maze file but it isnt working.

## maze_env:

```maze_env.py``` requires four functions to be compatible with the pytorch training, specifically:
1. ```__innit__```:
```python
def __init__(self):
        # use of np.copy is key to avoiding the differnent state vars being updated
        self.agent_position = np.asarray([1,1])
        self.minimum_threshold = -1
        maze = mm.build_maze(5,5)
        # rewards map: maze without the agents position marked, shallow copy to stop alterations
        self.rewards_map = np.copy(maze) 
        maze[1,1], maze [-2,-2] = 3, 4 # insert agent position and goal position
        self.original_maze = np.copy(maze) # never edited
        self.old_state = maze # updated constantly
        self.state = maze
        goal_state = np.copy(maze)
        goal_state[1,1], goal_state[-2,-2] = 0, 3 
        # copy of maze with agent position at end of maze
        self.goal_state = goal_state
        self.done = False
```
2. ```step```:
```python
    def step(self, agent_action, agent_score):
        # calc get rewards etc for state transitions
        self.old_state = self.state
        self.state[self.agent_position[0],self.agent_position[1]] = 0  # erase old agent position 3->0
        self.agent_position += np.asarray(agent_action) # actions written as vectors
        self.state[self.agent_position[0],self.agent_position[1]] = 3  # flag new agent position x->3
        self.done = self.is_ep_finished(agent_score)
        reward = self.calc_reward(self.agent_position) 
        return ([self.old_state, agent_action, self.state, reward])
```
3. ```reset```:
```python
    def reset(self):
        # reset the env
        self.state = self.old_state = self.original_maze 
        self.state[self.agent_position[0],self.agent_position[1]] = 0  # erase old agent position 3->0
        self.agent_position = np.asarray([1,1])
        self.done = False
```
4. ```render```:
```python
    def render(self, mode='human', close=False):
        mm.show(self.state, self.agent_position) # print out the maze using matplot.imshow
```

Note here that there are some helper functions, ``` is_ep_finsihed``` and ``` calc_reward``` which simply tell us if the
episode is over and the reward for a given action respectively. I anticipate the env will also need some changes once the 
aMAZE_ai file is finished, Given the large overlap of the environment methods and  those discussed in the early Agent 
designs, the Agent class must be redesigned.

# gym_maze_package Status:
Improvements for the gym_maze_package are mostly at a standstill until I figure out how to get the custom gym environment
installed to the gym package. But the codes in the gym_env.py are all working and have been tested. The issues with
importing the maze_maker package from sever directories above the gym_env.py were solved! It was an overall quite fun 
experience to learn how to configure this environment by solving that issue.

## Roadmap
- [x] Design Maze_maker package (:exclamation: introduce pickle to save mazes),
- [x] Visualise Maze,
- [X] Finish maze_env (:exclamation: installation bug)
- [x] Build aMAZE_ai package (now with the new pytorch involvment, this should be made fairly quickly),
- [ ] Build CNN and batch learning algorithm
- [ ] Testing:,
  - [ ] Testing is now expected to comprise curriculum learning 
- [ ] Find optimal params for a given maze/ investigate learning rate effects on efficiency.


## Similar Work
This is a very basic application of RL and so has been done many times.
One example I saw used the simpleai A* algorithm so solve mazes: [simpleai](https://simpleai.readthedocs.io/en/latest/).

## References

1. T. Mitchell, 'Machine Learning, International Student Edition', 1997, p.367-387 [Mitchell](http://www.cs.cmu.edu/~tom/mlbook.html)
2. P. Wilmott, 'Machine Learning: An Applied Mathematics Introduction, 2020, p.173-215 [Willmott](https://www-tandfonline-com.nottingham.idm.oclc.org/doi/full/10.1080/14697688.2020.1725610) 
3. A. Poddar, 'Making a custom environment in gym', 2019, [Poddar](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)

## Contact

:email: Email: [ppxeh1@nottingham.ac.uk](mailto:ppxeh1@nottingham.ac.uk)
