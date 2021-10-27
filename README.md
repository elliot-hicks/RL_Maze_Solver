[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6094273&assignment_repo_type=AssignmentRepo)
# Scientific Programming in Python – submission 1

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

![alt text](https://github.com/mpags-python/coursework2021-sub1-elliot-hicks/blob/main/plot_example_20x30.png)

### Maze production
```Maze_maker.py``` is a module made to create 2D mazes of any shape. ```Maze_maker.py``` has three main functions for regular use:
1. **```create_path(height, width)```**:  Returns 2d NumpY Array. Used to create a *random* walk through the maze as a solution. Involves several biasing variables to help the random walk reach [-1,-1] without too many steps. This actually creates more difficult mazes as the solution is less easy to stumble upon.
```
def create_path(height, width):
    # note we wont worry about looping paths as it test optimisation of Q learning
    # always start at [0,0], and end at [-1,-1]
    
    maze_frame = np.ones((height,width)) # matrix describing the random walk
    position = [0,0]
    test_position = [0,0]
    
    while not end_found:
        valid_position = False
        while not valid_position:   
            # bias random walk to adjust to different shape mazes
            direction_bias_level = height/(height+width)
            step_bias = 0.7 #prob step right = 70%
            if (random.uniform(0, 1)<direction_bias_level): # way to implement random vars
                axis_for_movement = 0
            else:
                axis_for_movement = 1
            if (random.uniform(0, 1)<step_bias):
                step_value = 1
            else:
                step_value = -1
            {
            ... #psuedo code to save space, see Maze_maker.py for full code
            
            CHECK IF NEW POSITION IS VALID:
                  IF SO, UPDATE POSITION
                  ELSE START AGAIN
            REPEAT UNTIL FINAL POSITION FOUND
            ...
            }                   
    return maze_frame
```
2. **```fill_maze_walls(stacked_maze_frame, number_of_mazes)```**: Returns 2D NumPy array. Uses another function ```stack_mazes()``` which returns s a 3D matrix of mazes. ```fill_maze_walls()``` then adds walls at random everywhere except where the paths are to fill out the maze.
Key parameters:
   * ```stacked_maze_frame```: 3D NumPy array with dimensions (number_of_mazes, height, width). Created in ```stack_mazes()``` which is not shown.
   * ```number_of_mazes```: The number of maze routes to be combined, same as ```n_routes``` (see below).
 
3. **```create_maze(size, n_routes)```**:  Returns 2D NumPy array. Calls all other maze functions, will provide a user input option. Key parameters:
    * **```shape```** (tuple , DEFUALT = (20,30)): size can take any tuple of two ints (height, width where height and width > 0). Note, the shape affects the bias of the random walk in create_path, vertical movements occur with probabilit (height/(height+width)). This seems to be the most general way of guiding the random walk through mazes of varying dimensions. 
    * **```n_routes```** (int, DEFAULT = 1): All mazes are produced with several possible routes as solutions. This was introduced as an possible extension to test the AI's efficiency in choosing the most optimal path. All tests will intitially be started with n_routes of 1 and then increased if the agent performs well.
Mazes can also be displayed using show()


### Maze Visualisation:

Mazes are currently visualised using [matplotlib.pyplot.imshow](https://matplotlib.org/stable/api/_as_gen/matplotlib.pyplot.imshow.html) from the matplotlib library. Values for pixels are:
* Tunnell: 0
* Wall: 1
* Start ([0,0], top left): 3 
* Goal ([-1,-1], bottom right): 4
```
def show(maze):
    #formatting image
    maze[-1,-1] = 4 
    maze[0,0] = 3 
    plt.axis('off')
    plt.imshow(maze)    
```
![alt text](https://github.com/mpags-python/coursework2021-sub1-elliot-hicks/blob/main/plot_example_20x30.png)

### Status
All possible improvements are listed below:
1. A new function to save the mazes will be needed, the mazes are randomly generated so it makes sense to save some to test the AIs performance over time.
1. Exception handling may be itroduced on maze creation for the shape variable.
2. As it stands, mazes will be kept as 2D NumPy arrays and fed to an Agent object to explore, the Agent will have an attribute Q-table to score positions for Q-learning. However, it may be more efficient to have mazes carry their own Q-tables but it may be more difficult because the maze will require agent information such as learning rate and penalties.
3. The higher dimensions used in the creation of arrays lead to unsighlty indentation which may be hard to follow, possible use of flattening/reshape to make the maze could make the code more aesthetic but harm readability. 

## **```aMAZE_ai.py```**
### Agent
**```aMAZE_ai.py```** will contain an Agent class that will solve a given maze from the **```Maze_maker```** library using Q-learning techniques.

Agent main **attributes**:
1. **```learning_rate```** (float): This will affect how quickly the agent will take in new information for updating Q-values.
2. **```Q_table```** (3D NumPy array): A table the Agent will queery to make decisions during the exploration of the maze.
3. **```position```** (list): Gives the position of the agent inside the maze. This can be used to show the movement of the agent in animations later on. 
4. penalties/rewards: Not technically an attribute to the Agent object but will affect the agents behaviour, likely will be encoded in to the environment the agent percieves by using the maze matrix (see make environment).

Agent main **methods**:
1. **```create_environment(maze)```**: This will return a 2D NumPy array with shape (width+2, height+2) as the maze it is asked to solve. The matrix will contain the values of the immediate rewards/penalties the agent would recieve for moving to a position in the maze. The environment is larger than the maze so that attempting to step out of the maze can be heavily penalised. 
2. **```explore()```**: This will tell the agent to *explore* its environment as part of its Q-learning.
3. **```update_Q()```**: Update Q-table.
4. **```pick_action()```**: Returns int. Use Q-table to choose next action based on immediate reward and future state.
5. **```check_position()```**: Returns bool value. Check if Agent has found end of maze.
6. **```save_sol()```**: Saves the Q-table values for the maze, possibly in form of actions i.e. 1,2,1,4,3 where each corresponds to an action. Or saves a *map* which draws out the route in a 2D array. I beleive the first option would be more efficient to use with some sort of ```take_action(action)``` function. 
7. **```save_stats()```**: Writes to a text file. All attempts to solve the maze will have stats recored e.g. time to solve, total length of solution route. This will allow for performance of the AI to be monitored over time. Possibly saved as image, with plot of maze and the stats underneath.

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

## References

1. T. Mitchell, 'Machine Learning, International Student Edition', 1997, p.367-387 [Mitchell](http://www.cs.cmu.edu/~tom/mlbook.html)
2. P. Wilmott, 'Machine Learning: An Applied Mathematics Introduction, 2020, p.173-215 [Willmott](https://www-tandfonline-com.nottingham.idm.oclc.org/doi/full/10.1080/14697688.2020.1725610) 


## Contact

:email: Email: [ppxeh1@nottingham.ac.uk](mailto:ppxeh1@nottingham.ac.uk)
