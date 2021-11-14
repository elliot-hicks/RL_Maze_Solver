# DATE SINCE LAST UPDATE: 25/10/2021
# Author: Elliot Hicks
# Contact: ppxeh1@nottingham.ac.uk

import numpy as np
import matplotlib.pyplot as plt
import random


def position_check(position, height, width):
    if ((position[0]>=0 and position[0]<=height-1 ) and
        (position[1]>=0 and position[1]<=width-1)):
        return True
    else:
        return False

# what type should the maze be? 
def create_path(height, width): #WORKS
    # note we wont worry about looping paths as it test optimisation of Q learning
    # may bias directions but testing fully random walk now
    # always start at [0,0], and end at [-1,-1]
    # time randint vs randuni for direction changes
    
    maze_frame = np.ones((height,width))
    maze_frame[0,0] = 0
    end_found = False
    
    position = [0,0]
    test_position = [0,0]
    while not end_found:
        valid_position = False
        while not valid_position:   
            direction_bias_level = height/(height+width)
            step_bias = 0.7 # think about how to find a justified value
                            # higher bias makes harder maze!!
            if (random.uniform(0, 1)<direction_bias_level):
                axis_for_movement = 0
            else:
                axis_for_movement = 1
            if (random.uniform(0, 1)<step_bias):
                step_value = 1
            else:
                step_value = -1
            #-1 left/up, 1 right/down (biased because we want to get to [-1,-1])
            test_position[axis_for_movement] = position[axis_for_movement] + step_value
            valid_position = position_check(test_position, height, width)
            if (valid_position):
                position[axis_for_movement] = position[axis_for_movement] + step_value
                maze_frame[position[0],position[1]] = 0
                if (position == [height-1,width-1]):
                    end_found = True
            else:
                pass
        
    return maze_frame

def stack_mazes(size, number_of_mazes):
    # creates a 3d np array of mazes
    stacked_maze_frame = np.zeros((size[0],size[1],number_of_mazes)) 
    for iteration in range(number_of_mazes):
        stacked_maze_frame[:,:,iteration] = create_path(size[0], size[1])
    return stacked_maze_frame

def fill_maze_walls(stacked_maze_frame, number_of_mazes):
    # we take all the mazes in the stacked maze 3d array and combine them
    # if a point is part of any maze's route, it is added to the full maze's route
    maze_frame = stacked_maze_frame[:,:,0]
    for row in range(len(maze_frame[:,0])):
        for column in range(len(maze_frame[0,:])):
            for maze in range(number_of_mazes):
                maze_frame[row,column] *= stacked_maze_frame[row,column,maze]
            maze_frame[row,column] *= random.randint(0,1) #
    return maze_frame

def show(maze):
    #formatting image
    maze[-1,-1] = 4 
    maze[0,0] = 3 
    plt.axis('off')
    plt.imshow(maze)    
            

def create_maze(size = (20,30),n_routes = 1):
    
    stacked_maze_frame = stack_mazes( size, n_routes)
    maze = fill_maze_walls(stacked_maze_frame, n_routes)
    return maze
    
maze = create_maze((20,30),1)
show(maze)  

