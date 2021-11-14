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

### ```recursive_maze(maze)```:
   
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
![alt text](https://github.com/mpags-python/coursework2021-sub2-elliot-hicks/blob/main/new_plot_example_20x30.png)

### Status
All possible improvements are listed below:
1. To aid the training, walls need to be added around the maze to block in the agent.
2. Add pickle or start using seeds, verring towards use of pickle so i can also save NN parameters.
## **```aMAZE_ai.py```**
### Agent
**```aMAZE_ai.py```** will contain an Agent class that will solve a given maze from the **```Maze_maker```** library using Q-learning techniques.

Agent main **attributes**:
1. **```learning_rate```** (float): This will affect how quickly the agent will take in new information for updating Q-values.2
2. **```position```** (list): Gives the position of the agent inside the maze. This can be used to show the movement of the agent in animations later on. 
3. penalties/rewards: Not technically an attribute to the Agent object but will affect the agents behaviour, likely will be encoded in to the environment the agent percieves by using the maze matrix (see make environment).

Agent main **methods**:
1. **```create_environment(maze)```**: This will return a 2D NumPy array with shape (width+2, height+2) as the maze it is asked to solve. The matrix will contain the values of the immediate rewards/penalties the agent would recieve for moving to a position in the maze. The environment is larger than the maze so that attempting to step out of the maze can be heavily penalised. 
2. **```explore()```**: This will tell the agent to *explore* its environment as part of its Q-learning.
3. **```pick_action()```**: Returns int. Use Q-table to choose next action based on immediate reward and future state.
4. **```check_position()```**: Returns bool value. Check if Agent has found end of maze.
5. **```save_sol()```**: Saves the Q-table values for the maze, possibly in form of actions i.e. 1,2,1,4,3 where each corresponds to an action. Or saves a *map* which draws out the route in a 2D array. I beleive the first option would be more efficient to use with some sort of ```take_action(action)``` function. 
6. **```save_stats()```**: Writes to a text file. All attempts to solve the maze will have stats recored e.g. time to solve, total length of solution route. This will allow for performance of the AI to be monitored over time. Possibly saved as image, with plot of maze and the stats underneath.

### Status:
aMAZE_ai (name is up for debate) is yet to be written. I decided that desinging the mazes should take precedence over designing the AI because the maze design will heavily influence the behavior of the Agent. I have also been reading Tom Mitchell's *'Machine Learning'* and Paul Wilmott's *'Machine Learning: An Applied Mathematics Introduction'*. I aim to learn how to write a Q-learning algorithm from these texts. Links to both texts can be found below in [References](#references).

## Roadmap
- [x] Design Maze_maker package,
- [x] Visualise Maze,
- [ ] Build aMAZE_ai package (expected to take two weeks to get basic version working),
- [ ] Design Q-learning Algorithm,
- [ ] Testing:,
  - [ ] Test agent in empty Maze,
  - [ ] Test agent in simple Maze (5x5),
  - [ ] Test agent in complex mazes (10x10), (20x30), (100x100),
- [ ] Find optimal params for a given maze/ investigate learning rate effects on efficiency.


## Similar Work
This is a very basic application of RL and so has been done many times. One example I saw used the simpleai A* algorithm so solve mazes: [simpleai](https://simpleai.readthedocs.io/en/latest/).

References:

1. T. Mitchell, 'Machine Learning, International Student Edition', 1997, p.367-387 [Mitchell](http://www.cs.cmu.edu/~tom/mlbook.html)
2. P. Wilmott, 'Machine Learning: An Applied Mathematics Introduction, 2020, p.173-215 [Willmott](https://www-tandfonline-com.nottingham.idm.oclc.org/doi/full/10.1080/14697688.2020.1725610) 


## Contact

:email: Email: [ppxeh1@nottingham.ac.uk](mailto:ppxeh1@nottingham.ac.uk)

