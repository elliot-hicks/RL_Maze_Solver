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
import numpy as np
import matplotlib.pyplot as plt
import random

height = 10
width = 10
maze_frame = np.zeros((height, width))


def recursive_maze(maze=maze_frame):
    """
    Parameters
    ----------
    maze : ARRAY
        2D array that descibes the maze as a matrix of ints.
        0 = steppable space
        1 = wall
        2 = vertical gate (hole in vertical wall)
        3 = horizontal gate (hole in horizontal wall)

    Returns
    -------
    maze : Array
        maze is spit in four quadrants using one h. wall and one v. wall.
        walls have gates added at random
        each quadrant then is treated as a maze and fed back in to function
        this recursively builds up a maze.
    """
    maze_height = maze.shape[0]
    maze_width = maze.shape[1]
    if maze_height <= 2 or maze_width <= 2:
        return maze
    else:
        row = random.randint(1, maze_height - 2)  # row of new horizontal wall
        horizontal_wall = np.ones(maze_width)  # intitially full wall
        col = random.randint(1, maze_width - 2)  # col of new vertical wall
        vertical_wall = np.ones(maze_height)
        h_gate_ind_1 = random.randint(0, col - 1)  # col of horizontal gate
        h_gate_ind_2 = random.randint(col + 1, maze_width - 1)
        #  we need to not place vertical gate in the new horizontal wall
        valid_vertical_gate = False
        while not valid_vertical_gate:
            v_gate_ind = random.randint(0, maze_height - 1)
            if v_gate_ind != row:  # dont place gate inside new wall
                valid_vertical_gate = True
        # encode different gate for later processing
        vertical_wall[v_gate_ind] = 2  # add vertical gate to v wall
        horizontal_wall[h_gate_ind_1] = 3  # add horizontal gate to h wall
        horizontal_wall[h_gate_ind_2] = 3
        maze[row, :] = horizontal_wall  # add walls
        maze[:, col] = vertical_wall
        # the two walls split the maze in to 4 chambers
        # index limits of the full maze that define the chambers
        chamber_11 = [[0, row], [0, col]]
        chamber_12 = [[0, row], [col + 1, maze_width]]
        chamber_21 = [[row + 1, maze_height], [0, col]]
        chamber_22 = [[row + 1, maze_height], [col + 1, maze_width]]
        chambers = [chamber_11, chamber_12, chamber_21, chamber_22]

        for chamber in chambers:
            # Define the maze chamber:
            maze_chamber = maze[
                chamber[0][0]:chamber[0][1], chamber[1][0]:chamber[1][1]
            ]
            # Fill maze chamber:
            maze_chamber = recursive_maze(maze_chamber)
            # Add chamber to maze
            maze[
                chamber[0][0]:chamber[0][1], chamber[1][0]:chamber[1][1]
            ] = maze_chamber
        return maze


def finalise_maze(maze):
    """
    Parameters
    ----------
    maze : Array
        Sometimes added walls obstruct gates, so we use the encoding
        from the recursive maze generator to find gates and clear space
        around them as necessary. Final prodcut is a maze where all points
        are accessible.

    Returns
    -------
    maze : Array
        returns finalised array.
    """
    maze_height = len(maze[:, 0])
    maze_width = len(maze[0, :])
    for row in range(maze_height):
        for col in range(maze_width):
            if maze[row, col] == 3:
                maze[row + 1, col], maze[row, col], maze[row - 1, col] = 0, 0, 0
            elif maze[row, col] == 2:
                maze[row, col - 1], maze[row, col], maze[row, col + 1] = 0, 0, 0
            else:
                pass
    # Create canvas for maze, produces boundary walls:
    maze_with_borders = np.ones((maze_height + 2, maze_width + 2))
    # Add maze to canvas:
    maze_with_borders[1:maze_height + 1, 1:maze_width + 1] = maze
    return maze_with_borders


def show(maze, agent_position):
    # Plots array with start and ending points highlighted,
    # encodes agent as 3, goal state as 4:
    maze[-2, -2] = 4
    maze[agent_position[0], agent_position[1]] = 3
    plt.axis("off")
    plt.imshow(maze)


def build_maze(width=10, height=10):
    maze_frame = np.zeros((height, width))
    maze_init = recursive_maze(maze_frame)
    maze = -1 * finalise_maze(maze_init)
    maze[-2, -2] = +1  # insert rewards for being in goal state
    return maze
