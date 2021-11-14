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
* **Maze_maker**: link for code: [Maze_maker.py](https://github.com/elliot-hicks/RL_Maze_Solver/blob/main/maze_maker.py)
* **gym**
* **Pytorch**

# Usage
## ```Maze_maker.py```

This is a basic example of the mazes made by ```Maze_maker.py```, a 20x30 maze where the agent will start at [0,0] (top-left) and aims to find the exit at [-1,-1] (bottom-right).
Note: ```Maze_maker.py``` was updated from a random walk generator to a recursive maze generator, this makes far more difficult mazes.

![Maze_maker.py original design](https://github.com/mpags-python/coursework2021-sub2-elliot-hicks/blob/main/plot_example_20x30.png)
![Maze_maker.py new design](https://github.com/mpags-python/coursework2021-sub2-elliot-hicks/blob/main/new_plot_example_20x30.png)

### Maze production
```Maze_maker.py``` is a module made to create 2D mazes of any shape. 
The recursive algorithm that inspired this method can be found here: [wiki recursive](https://en.wikipedia.org/wiki/Maze_generation_algorithm#Recursive_implementation).

An exmple of the recursive process is shown below:

![recursive mase generation](https://upload.wikimedia.org/wikipedia/commons/9/90/Recursive_maze.gif)


Key functions:

###```recursive_maze(maze)```:
   
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

### ```finalise_maze(maze)```:

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
### Maze Visualisation:

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

### Status
All possible improvements are listed below:
1. To aid the training, walls need to be added around the maze to block in the agent.
2. Add pickle or start using seeds, veering towards use of pickle so i can also save NN parameters.
3. Add exception handling for stupid maze inputs
## **```aMAZE_ai.py```**
### Agent
* **```aMAZE_ai.py```** will contain an Agent class that will solve a given maze from the **```Maze_maker```** library using Q-learning techniques.
* The agent class was revised following the introduction of the ```Gym``` library:
### ```openAI/Gym```:
The plan is to use a CNN from the pytorch library to solve the mazes, to do this we need a gym environment, so i have attempted
to create one with the following file structure, however im still new to making packages so I expect some issues, particularly
with the use of ```maze_maker.py``` as it is several directories above the maze_env file, the current file structure is:
``` 
maze_maker:
  > __innit__
  > maze_maker.py
aMAZE_ai
  > gym-maze
    >README.md
    >gym_maze:
      >__init__.py
      >envs:
        >maze_env.py
        >__innit__.py
```

## ```maze_env.py```:
```maze_env.py``` requires four functions to be compatible with the pytorch training, specifically:
1. ```__innit__```:
```python
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
```
2. ```step```:
```python
def step(self, agent_action, agent_sore):
        # calc get rewards etc for state transitions
        self.old_state = self.state
        self.state[self.agent_positon[0],self.agent_position[1]] = 0  # erase old agent position 3->0
        self.agent_position += np.asarray(agent_action) # actions written as vectors
        self.state[self.agent_positon[0],self.agent_position[1]] = 3  # flag new agent position x->3
        self.done = is_ep_finished(agent_score)
        reward = calc_reward(agent_position)
        
        return ([self.old_state, action, self.state, reward])
```
3. ```reset```:
```python
    def reset(self):
        # reset the env
        self.state = self.old_state = original_maze 
        self.agent_position = np.asarray([1,1])
        self.done = False
```
4. ```render```:
```python
    def render(self, mode='human', close=False):
        mm.show(self.state) # print out the maze using matplot.imshow
```

Given the large overlap of the environment methods and those discussed in the early Agent designs, 
the Agent class must be redesigned.
## ```Pytorch```:

I plan to use a CNN from the pytorch library to solve the maze, a single CNN will be used. The current aMAZEai package 
shows a pseudocode for the implementation of two NNs, a main and a target NN, while this would produce more stable training, it seemed
overcomplicated. However, the idea of a memory buffer is excellent,and I will be applying it to the CNN code. I have also
seen interesting methods of applying an 'elite' batch which retains the most profitable episodes for a longer time period,
I would like to apply a similar method, time permitting.

### Status:
aMAZE_ai (name is up for debate) is yet to be written, with the gym environment now taking precedence, hence the roadmap has also been revised.
I decided that designing the mazes should take precedence over designing the AI because the maze design will heavily influence the behavior of the Agent.
I have also been reading Tom Mitchell's *'Machine Learning'* and Paul Wilmott's *'Machine Learning: An Applied Mathematics Introduction'*. 
I aim to learn how to write a Q-learning algorithm from these texts. Links to both texts can be found below in [References](#references).

## Roadmap
- [x] Design Maze_maker package,
- [x] Visualise Maze,
- [ ] Finish maze_env
- [ ] Build aMAZE_ai package (with the new pytorch involvment, this should be made fairly quickly),
- [ ] Build CNN and batch learning algorithm
- [ ] Testing:,
  - [ ] Testing is now expected to comprise of curriculum learning 
- [ ] Find optimal params for a given maze/ investigate learning rate effects on efficiency.


## Similar Work
This is a very basic application of RL and so has been done many times. One example I saw used the simpleai A* algorithm so solve mazes: [simpleai](https://simpleai.readthedocs.io/en/latest/).

## References

1. T. Mitchell, 'Machine Learning, International Student Edition', 1997, p.367-387 [Mitchell](http://www.cs.cmu.edu/~tom/mlbook.html)
2. P. Wilmott, 'Machine Learning: An Applied Mathematics Introduction, 2020, p.173-215 [Willmott](https://www-tandfonline-com.nottingham.idm.oclc.org/doi/full/10.1080/14697688.2020.1725610) 


## Contact

:email: Email: [ppxeh1@nottingham.ac.uk](mailto:ppxeh1@nottingham.ac.uk)
