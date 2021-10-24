[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6094273&assignment_repo_type=AssignmentRepo)
# Scientific Programming in Python – submission 1

## Project title: RL_Maze_Solver

## Student name: Elliot Hicks
  
## Table of Contents
1. [Description](#description)
2. [Installation](#installation)
3. [Usage](#usage)
4. [Workflow](#workflow)
5. [Similar Work](#similar-work)
6. [References](#references)

  
# Description

Give some brief background and describe what you intend your code to do, together with a rough outline (e.g., classes, functions, snippets, comments, pseudocode).

RL_Maze_Solver will demonstrate the ability of reinforcement learning (specifically Q-learning) by randomly generating 2D mazes and tasking an agent to find its most efficient solution.

# Installation
Information on packages/ environments will be made available here.

# Usage
## ```Maze_maker.py```
![alt text](https://github.com/mpags-python/coursework2021-sub1-elliot-hicks/blob/main/plot_example_20x30.png)

### Maze production
```Maze_maker.py``` is a module made to create 2D mazes of any shape. maze_maker has three main functions for regualar use:
1. **```make_path()```**:
2. **```fill_walls()```**:
3. **```create_maze()```**: key vars: 
  1. **```shape```** (tuple , DEFUALT = (20,30)): size can take any tuple of ints height, width where height and width > 0. Note shape affects the bias of the random walk in make_path, vertical movements are favoured more by a factor of eqn(hieght/(height+width)). This seems to be the most general way of guiding the random walk through mazes of varying dimensions. 
  2. **```n_routes```** (int, DEFAULT = 1): All mazes are produced with several possible routes as solutions. This was introduced as an possible extension to test the AI's efficiency in choosing the most optimal path. All tests will intitially be started with n_routes of 1 and then increased if the agent performs well.
Mazes can also be displayed using show()
### Status
All possible improvements are listed below:
1. Exception handling may be itroduced on maze creation for the shape variable.
2. As it stands, mazes will be kept as 2D NumPy arrays and fed to an Agent object to explore, the Agent will have an attribute Q-table to score positions for Q-learning. However, it may be more efficient to have mazes carry their own Q-tables but it may be more difficult because the maze will require agent information such as learning rate and penalties.
3. The higher dimensions used in the creation of arrays lead to unsighlty indentation which may be hard to follow, possible use of flattening/reshape to make the maze could make the code more aesthetic but harm readability. 

## **```aMAZE_ai.py```**
### Agent
**```aMAZE_ai.py```** will create an Agent that will solve a given maze from the **```Maze_maker```** library using Q-learning techniques.
Expected main attributes:
1. **```learning_rate```** (float):
2. **```Q_table```** (nd NumPy array): a table the Agent will queery to make decisions during the exploration of the maze.
3. **```position```** (list): Gives the position of the agent inside the maze. This can be used to show the movement of the agent in animations later on. 
4. penalties/rewards: Not technically an attribute to the Agent object but will affect the agents behaviour, likely will be encoded in to the environment the agent percieves by using the maze matrix (see make environment)

Expected main methods:
1. **```create_environment(maze)```**: 
    This will return a 2D NumPy array with shape (width+2, height+2) as the maze it is asked to solve. The matrix will contain the values of the immediate rewards/penalties the agent would recieve for moving to a position in the maze. The environment is larger than the maze so that attempting to step out of the maze can be heavily penalised. 

## Workflow

## Similar Work
This is a very basic application of RL and so has been done many times. One popular example I have seen is the (LIBRARY) simpleai A* maze solver: \LINK

## References
BOOKS USED FOR MATHS





[See here for a guide to writing markdown.](https://guides.github.com/features/mastering-markdown/)
