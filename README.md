[![Open in Visual Studio Code](https://classroom.github.com/assets/open-in-vscode-f059dc9a6f8d3a56e377f745f24479a46679e63a5d9fe6f495e02850cd0d8118.svg)](https://classroom.github.com/online_ide?assignment_repo_id=6094273&assignment_repo_type=AssignmentRepo)
# Scientific Programming in Python – submission 3

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

RL_Maze_Solver will demonstrate the ability of reinforcement learning (specifically Q-learning)
by randomly generating 2D mazes and tasking an agent to find its most efficient solution. 
Mazes are made with the maze_maker.py package I wrote and allow for 2D mazes with multiple routes to be made. The design
of the project was made with generality in mind, this code could easily be tweaked to have solve problems where the goal
and starting states are varied randomly. Unfortunately, convergence to an optimal policy was not achieved. The reasons 
for this will me mentioned in the conclusion section.


As the project has grown in complexity, restructuring has been vital for organisation.
List of main components and their purposes:
### maze_maker_package:

This package generates recursive mazes.

### gym_maze_package:
This is the package that converts the maze produced by maze_maker to a suitable gym environment for pytorch implementation

### CNN10:
This is a package designed to make a CNN (ECNN10) which can take in a 10x10 single channel maze, designed to be trained to solve 
the maze by learning the optimal policy.

### agent_package: 
This package contains the design for the Agent class. This class' purpose is to navigate the maze using the network
for its policy. It also contains the ExperienceBuffer and EliteExperienceBuffer classes.

### cnn_maze_solver.py:
This is where all the packages are combined:
  * the previous packages are used to create the setup for the RL task.
  * A training algorithm is employed to train the ECNN10 to produce an optimal policy function.
  * The best solution found is animated using matplotlib animate
# Installation
Information on packages/ environments will be made available here:

Required Packages:

* **NumPy**
* **Random**
* **Matplotlib.pyplot**
* **maze_maker**:
* **gym_maze_package**
* **CNN10**
* **agent_package**
* **collections**
* **gym**
* **Pytorch**

In order to install maze_gym to create the custom gym environment, maze_maker must first be installed using pip install
in the maze_maker_package directory:
```shell
...\maze_maker_package>pip install -e . maze_maker
```
Following this, the gym-maze package can be installed using the following in the gym_maze_package directory:
```shell
...\gym_maze_package>pip install -e .gym_maze
```

# Usage
# maze_maker_package

This is a basic example of the mazes made by ```Maze_maker.py```, a 20x30 maze where the agent will start at [0,0] (top-left) and aims to find the exit at [-1,-1] (bottom-right).
Note: ```Maze_maker.py``` was updated from a random walk generator to a recursive maze generator, this makes far more difficult mazes.


![Maze_maker.py original design](https://github.com/mpags-python/coursework2021-sub3-elliot-hicks/blob/main/plot_example_20x30.png)
![Maze_maker.py new design](https://github.com/mpags-python/coursework2021-sub3-elliot-hicks/blob/main/new_plot_example_20x30.png)

## Maze production
```Maze_maker.py``` is a module made to create 2D mazes of any shape. 
The recursive algorithm that inspired this method can be found here: [wiki recursive](https://en.wikipedia.org/wiki/Maze_generation_algorithm#Recursive_implementation).

An exmple of the recursive process is shown below:
<p align="center">
  <img width="450" height="450" src="https://upload.wikimedia.org/wikipedia/commons/9/90/Recursive_maze.gif">
</p>

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
* Wall: -1
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
<p align="center">
  <img width="600" height="400" src="https://github.com/mpags-python/coursework2021-sub3-elliot-hicks/blob/main/new_plot_example_20x30.png">
</p>


## maze_maker Status:
The maze_maker file is now finished. Unfortunately, an appication of pickle was not completed due to comlpications in 
other parts of the project.
All possible improvements are listed below:
1. Add pickle or start using seeds, veering towards use of pickle so we could also save NN parameters.
2. Add exception handling for invalid maze inputs

# agent_package:

## ```agent.py```:
This will contain an Agent class that will solve a given maze from the **```maze_maker```** library using Q-learning techniques.
* The agent class was revised following the introduction of the OpenAI and Gym libraries 
* [openAI/Gym custom gym environment: 'maze_env'](#openAI/Gym-custom-gym-environment:-'maze_env':).

## ```ExperienceBuffer```:
In order to train a CNN to solve these mazes, I want to give it a short-term memory, this comes in the form of the 
ExperienceBuffer class. Originally, a deque object from *collections* was used, but this class had very limited slicing
features, so a NumPy ndarray was used instead. Any transitions made by the Agent class will be recorded in the
experience buffer. 
The buffer contains transitions, tuples of (s,a,r,d), where:
* s = state (before action)
* a = action taken in state s
* r = reward from action a in state s
* d = done flag, a bool telling us whether the trajectory is finished.

The (s,a,r) tuples are key to the training process, the done flag is used to isolate individual trajectories/ episodes.

The ExperienceBuffer class has three main attributes:
1. capacity (int): How many experiences the deque can hold, this is analogous to how short or long the networks memory is.
2. memory_buffer (NumPy ndarray): the actual memory store with maxlen = capacity,
3. size (int): the length of the memory_buffer, once size = capacity, training can start.

The ExperienceBuffer has three useful methods:
1. ```__init__```:
```python 
def __init__(self, capacity):
      self.capacity = capacity
      self.memory_buffer = collections.deque(maxlen = capacity)
      self.size = len(self.memory_buffer) # current size of the memory buffer 
```
2. ```add```:
```python 
def add(self, experience):
        # Mimic deque functionality, ndarray with a capacity.
        if (self.size == self.capacity):
            # if at capacity, remove first entry
            self.memory_buffer = np.delete(self.memory_buffer, (0), axis=0)
        else:
            self.size += 1
        self.memory_buffer = np.vstack((self.memory_buffer, experience))
```
3. ```random_sample_batch```
```python 
def random_sample_batch(self, batch_size = 100):
        batch = random.sample(self.memory_buffer,batch_size)
        return batch
```
There is also an ```python update_values``` function which rewrites the rewards column of the experience buffer for the
last N steps, where N is the length of the last episode.

## ```EliteExperienceBuffer```:
In order to improve training of the CNN, many training regimens employ an 'elite buffer', a buffer of transitons/experiences
that were part of particularly good episodes (say the top 10% of all final scores). The memories in this buffer are
protected, their existence in the CNNs memory extended, so that they can reinforce its behaviour for a longer amount of
time. The experience buffer was finally implemented in an attempt to improve learning, it initially takes trajectories
that are in the top 10% and checks if they are better than any other trajectories stored in the EliteBuffer. 
The addition of a new elite memory does not remove the poorest of the elite memories to free
up space, instead any elite memory is removed, including possibly the best, from the elite buffer. This stops trajectories
being used too much.

The EliteExperienceBuffer is a type of ExperienceBuffer with harsher terms for adding memories, due to this 
relationship, the EliteExperienceBuffer is inherited from the ExperienceBuffer class. Agents will therefore contain
an ExperienceBuffer and EliteExperienceBuffer which work in tandem. In order to give the elite buffer access to the
normal buffers trajectories, a reference of the agent is passed in the EliteExperienceBuffers ```python init```.
```python
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
            self.agent.elite_steps.append(n_steps)
            for i in range(1, n_steps + 1):
                self.add(self.agent.replay_buffer.memory_buffer[-i, :])
```

## ```Agent```:
The agent class is designed to take in action probabilities from the CNN and to take actions according to an epsilon 
greedy aproach. The epsilon value will be an attribute of the Agent class and decreases as training continues,
with exploration initially being large such that the agent explores and gives the network new data. After some 
exploration period, the agent's epsilon value begins decreasing by 10% every 10 episodes. Epsilon is
given a small lower bound of 0.01.This is so that the agent becomes more reliant on the policy as time goes on 
and the policy improves, but exploration is still possible. 

So that the CNN did not learn to not step in to walls through penalisation, moves were tested, and the actions that
put the agent in an invalid state had their corresponding probabilities set to zero. The addition of maze boundaries
also made this easier as stepping out of the maze still gave an invalid state without the code crashing.
This was done in the ```test_actions``` function:
```python    
        def test_actions(self, action_probabilities):
        for action in range(4):
            test_position = self.position + self.action_space[action, :]
            # Test if new position is a wall (val = -1):
            if (self.environment[test_position[0], test_position[1]] == -1):
                action_probabilities[action] = 0
        normed_probabilities = normalise(action_probabilities)  # Renormalize
        return (normed_probabilities)
```
Note ```normalise``` is a function made so that probabilties are rescaled to still ad to 1 for use in np.random.choice.
The epsilon-greedy was implemented in the ```choose_action``` function. Here the probabilities are used to select an 
action using ```np.random.choice``` where they are given as an argument to give the p.m.f of each action. In the 
cases that exploration is chosed, the policy's action is removed from the possible choices and another action chosen,
again this is done by setting the P(policy action) to zero and renormalising.

```
    def choose_action(self, probabilities):
        # Get policy action:
        policy_action_label = np.random.choice(self.action_space_labels, p=probabilities)
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
                return policy_action, policy_action_label
        else:
            return policy_action, policy_action_label
```
The exception handling here is used when only one action is possible, therefore the probabilities become 
(0,0,0,P(policy action)), in this case, if exploration is chosen, the new probabilities would be (0,0,0,0), which
throws an error in the norm function because of the 0 division, so it is caught.

The epsilon scheme is simple:
* If in the exploration period epsilon = 1
* Then decrease epsilon by 10% every 10 episodes
* When epsilon reaches 0.01, it stays there.
### agent.py status:
The agent class is complete.

# gym_maze_package:

In order to make the generated maze from maze_maker a gym-environment for the pytorch library I had to learn to create my own custom 
environments for OpenAi's Gym package. I did this using A. Poddars article [3]. Following this, and with some help, I 
was able to get the maze-env produced, but it took a considerable amount of time. 

## maze_env:

```maze_env.py``` requires four functions to be compatible with the pytorch training, specifically:
1. ```__innit__```:
```python
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
```
2. ```step```:
```python
    def step(self, agent_action):
        self.step_count += 1
        self.old_state = np.copy(self.state)
        # Update agent position and state:
        self.state[self.agent_position[0], self.agent_position[1]] = 0
        self.agent_position += np.asarray(agent_action)
        self.state[self.agent_position[0], self.agent_position[1]] = 3
        self.done, successful = self.is_ep_finished()
        reward = +10 if successful else -1
        return ([self.old_state, agent_action, reward, self.done])
```
Shallow copies were invaluable as states are often updated in training, so using np.copy was common.
Note, a +10 reward was given to agents that found the solution, this is because some trajectories time-out (hit the mx
step count). But a trajectory that gets to the goal at the max-1, is far better than the one that gets timed out at
max, but was nowhere near the goal. This was used to encourage the agents to win.
3. ```reset```:
```python
    def reset(self):
        # reset the environment for next episode:
        self.state = np.copy(self.original_maze)
        self.agent_position = np.asarray([1, 1])
        self.step_count = 0
        self.done = False
```
4. ```render```:
```python
    def render(self, mode='human', close=False):
        mm.show(self.state, self.agent_position) # print out the maze using matplot.imshow
```

A helper function is_ep_finished is also used, it tells us if the trajectory is over and if it was a successful one, 
returning two bools: ```done``` and ```success```. Print statements are also used to inform the user of episode results.
Given the large overlap of the environment methods and  those discussed in the early Agent designs, 
the Agent class was heavily redesigned.

### gym_maze_package status:
The gym_maze_package is complete. A further improvement would be introducing an animation to the render function, 
I looked in to having it show the agents actions in real time, but it got complex and required threading.

## CNN10
The goal for a vairable sized CNN architecture was short lived, initially a 32x32 LeNet architecture was used, but
to keep the task simple, the CNN was made in to a 10x01 single channel architecture. The design here is by far the
most difficult part, tuning hyperparamters such as kernal size, stride and the overall architecture is very difficult. 
Often these are found using trial and error, but given the time constraints I was unable to find the working ones.
The ECNN10 is inherited from the torch.nn.Module class the architecture is as follows:
```python
        self.convolutional_1 = nn.Conv2d(1, 1, kernel_size=2, padding=1)
        self.ReLU_1 = nn.ReLU()
        self.convolutional_2 = nn.Conv2d(1, 1, kernel_size=2, padding=1)
        self.ReLU_2 = nn.ReLU()
        self.fully_connected_in = nn.Linear(in_features=196, out_features=500)
        self.ReLU_3 = nn.ReLU()
        self.fully_connected_out = nn.Linear(
            in_features=500, out_features=num_actions)
        self.softmax = nn.Softmax(dim=1)
```
Softmax was used to convert the outputs to a probability distribution.
Note that maxpooling was removed, this was becauseit was anticipated that decreasing resolution of the maze would 
cause too much information loss, making this a Fully Convolutional net. Unfortunately, this is where the most
doubt lies, as is there is no correct answer for the CNN architecure, only ones that work, and ones that dont.

## cnn_maze_solver.py
This file contains the training algorithm for the project and code for making an animation of the best route achieved
in the training process. The training process is inspired by the work done in the original 
[REINFORCE](https://link.springer.com/article/10.1023/A:1022672621406) paper. The rewards function is -1 for 
any step not resulting in the goal state, and +10 for a step that does. The training is as follows:
* Initialise maze environment using maze_maker and the gym_maze_packages to make the env.
* Use CNN10 to initialise the CNN that approximates the policy function
* Set optimisers as ADAM and learning rate as 1e-4
* train the CNN using ```train()```
The loss function used in this model is:
<p align="center">
  <img width="200" height="400" src="https://github.com/mpags-python/coursework2021-sub3-elliot-hicks/blob/main/CodeCogsEqn.gif">
</p>
(Above created using codecogs).

This corresponds to a zero-bias value approximation as seen in the REINFORCE paper. Here i indexs over the random batch
taken from the agent memory buffers. This method of random sampling is supposed to improve stability in training.
The loss is back-propagated using the usual PyTorch functions: optimiser.zero_grad, loss.backward, optimser.step.
The training code is too long to be shown here but is available in full in the cnn_maze_solver.py file. 
The CNN is trained every 5 episodes using the main memory buffer, and every 10 episodes with the elite buffer, once the
exploration period has concluded. The number of episodes ran can be changed in the maze_solverfunction, it is
recommended that the memory buffer size is less than exploration_period*max_steps, such that the exploration period
is fully utilised, this can be tuned either way but I recommend setting the exploration period by hand depending on 
the maze size. The same applies for the max_steps as more difficult mazes will require more steps.

The ```train``` function prints statements continuously in the following form:
```console
|Episode = 10 | Steps = 50 | Epsilon = 1.00 |
```
Where we see the current episode number, the number of steps taken in that episode to complete the trajectory, and
the epsilon value for that episode.  Print statements are also made when successful runs are found, i.e. 
the agent completes the maze: SUCCESS, and when it fails: TIMED OUT.

Several diagnostic lists are made, including 
* episode_steps : Number of steps in each episode during the training
* episode_av_reward: Average reward along each episode
* shortest_length: The length of the best route, found using min(episdode_steps)
* best_episode: The episode which corresponds to the best solution, found using np.argmin(episode_steps)

These can be used to determine if the CNN is improving:
* episode_steps trending downwards suggests improvements
* episode_av_reward trending upwards suggests improvements
* best_episode must be as small as possible, min value for any 10x10 maze is 19. 
### cnn_maze_solver.py status:
The cnn_maze_solver.py file is comlpete.

## CNN_maze_solver performance:

**The cnn_maze_solver has been loaded with hyperparams that make training faster for you when running it.**


The cnn_maze_solver.py appears to work occasionally, training is very unstable, but when forced in to a very simple situation
the agent does learn, however it is far too slow to be useful considering we are only working with 10x10 mazes. I believe
its likely due to the hyperparameters and architecture of the CNN being used. Given more time I would be able to 
further test different combinations of the parameters such as:
* ECNN10 architecture
* number of episodes
* epsilon scheme
* exploration period
* rewards scheme
* buffer sizes (Elite and normal)
 There was evidence of learning in some cases as shown by the following plot of steps vs episode. 
The steps were rolling averaged over 50 values:
<p align="center">
  <img width="600" height="400" src="https://github.com/mpags-python/coursework2021-sub3-elliot-hicks/blob/main/Possible_improvements.png">
</p>
Another example of learning is shown by:
<p align="center">
  <img width="600" height="400" src="https://github.com/mpags-python/coursework2021-sub3-elliot-hicks/blob/main/Possible_improvements_steps.png">
</p>
Here a general trend is shown with the steps decreasing, suggesting the agent learning. However, inspection of the maze 
shows it was a very simple situation. However, there is again evidence of learning, just very slow learning.
<p align="center">
  <img width="600" height="400" src="https://github.com/mpags-python/coursework2021-sub3-elliot-hicks/blob/main/example_very_simple_learning.png">
</p>
The steps functions are often very noisey and red herrings are common, often after 500 episodes, the CNN would stop finding the solution.
Again the main issue here is instability. It was attempted to use an empty maze as a form of curriculum learning, however,
the training was still unstable and sometimes the agent would stop solving the maze at all. Given the algorithms failure
to optimise the paths, the variables created were called best_episode rather than optimal. In order to still be able
to show a visual representation of the work, the states in the best episode were animated and plotted in the 
 ```animate_imshow``` function. This shows the best episode seen in the training process,but it likely will be found
using exploration more than exploitation.


I was very disappointed to see this project fail to solve these tasks, I will continue working on it in my spare time though. 

# Roadmap
- [x] Design Maze_maker package,
- [x] Visualise Maze,
- [X] Finish maze_env,
- [x] Build Agent package,
- [x] Build CNN package,
- [x] Design batch learning algorithm,
- [x] Testing:
  - [x] Testing is now expected to comprise curriculum learning,
- [x] Improve training with the addition of 'elite batches'

# Incompleted Parts
There were plans to add pickle so that users can save/load CNN parameters and mazes. However, the issues with 
getting the gym environment, learning the theory behind RL and learning to use PyTorch took a lot of time. Most
importantly, trying to fix the instability in the training took a lot of time at the end of the project. Given more time
the user functions would be completed. 

# Similar Work

This is a very basic application of RL and so has been done many times.
One example I saw used the simpleai A* algorithm so solve mazes: [simpleai](https://simpleai.readthedocs.io/en/latest/).

# References

1. T. Mitchell, 'Machine Learning, International Student Edition', 1997, p.367-387 [Mitchell](http://www.cs.cmu.edu/~tom/mlbook.html)
2. P. Wilmott, 'Machine Learning: An Applied Mathematics Introduction, 2020, p.173-215 [Willmott](https://www-tandfonline-com.nottingham.idm.oclc.org/doi/full/10.1080/14697688.2020.1725610) 
3. A. Poddar, 'Making a custom environment in gym', 2019, [Poddar](https://medium.com/@apoddar573/making-your-own-custom-environment-in-gym-c3b65ff8cdaa)
4. R.J.Williams, 'Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning', 1992, [REINFORCE](https://link.springer.com/article/10.1023/A:1022672621406)
# Comments/ Questions:

1. Given the size of this README.md growing so fast, should I start splitting this in to smaller readme files within the correspoding packages and retain this as a broader overview of the project? I want to make sure I'm following the 'style' used in github.
2. I know one of the criterion is good use of github, could you let me know what you think of my use so far? Im enjoying it, the more I use it the faster I can build up this project. I did have some issues where I lost file histories because I was forced to change their names. 

## Contact

:email: Email: [ppxeh1@nottingham.ac.uk](mailto:ppxeh1@nottingham.ac.uk)
