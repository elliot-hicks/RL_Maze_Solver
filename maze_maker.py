import numpy as np
import matplotlib.pyplot as plt
import random
height = 20
width = 30

maze_frame = np.zeros((height,width))

def recursive_maze(maze):
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
    if (maze_height <=2 or maze_width<=2):
        
        return maze
    else:
        
        row = random.randint(1,maze_height-2)  # row of new horizontal wall
        horizontal_wall = np.ones(maze_width) # intitially full wall
        col = random.randint(1,maze_width-2) # col of new vertical wall
        vertical_wall = np.ones(maze_height) 
        h_gate_ind_1 = random.randint(0,col-1) # pos of gate through horiz wall
        h_gate_ind_2 = random.randint(col+1,maze_width-1)
    
        #  we need to not place vertical gate in the new horizontal wall
        valid_vertical_gate = False
        while not valid_vertical_gate:
            v_gate_ind = random.randint(0,maze_height-1)
            if v_gate_ind != row: # dont place gate inside new wall
                valid_vertical_gate = True
                
        # encode different gate for later processing
        vertical_wall[v_gate_ind] = 2 # add vertical gate to v wall
        horizontal_wall[h_gate_ind_1] = 3 # add horizontal gate to h wall
        horizontal_wall[h_gate_ind_2] = 3 
        maze[row,:] = horizontal_wall   # add walls
        maze[:,col] = vertical_wall        
        
        # the two walls split the maze in to 4 chambers
        
        # index limits of the full maze that define the chambers
        chamber_11 = [[0,row],[0,col]] 
        chamber_12= [[0,row], [col+1, maze_width]]
        chamber_21= [[row+1,maze_height],[0,col]]
        chamber_22= [[row+1,maze_height],[col+1,maze_width]]
        chambers = [chamber_11,chamber_12,chamber_21,chamber_22]
        
        for chamber in chambers:
            maze_chamber = maze[chamber[0][0]:chamber[0][1],
                                chamber[1][0]:chamber[1][1]]
            maze_chamber = recursive_maze(maze_chamber)
            maze[chamber[0][0]:chamber[0][1],
                 chamber[1][0]:chamber[1][1]] = maze_chamber
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
    maze_height = len(maze[:,0])
    maze_width = len(maze[0,:])
    for row in range(maze_height):
        for col in range(maze_width):
            if(maze[row,col] == 3):
                maze[row+1,col], maze[row,col], maze[row-1,col] = 0,0,0
            elif(maze[row,col] == 2):
                maze[row,col-1], maze[row,col], maze[row,col+1] = 0,0,0
            else:
                pass
    # add walls surrounding the maze
    maze_with_borders = np.ones((maze_height+2,maze_width+2))
    maze_with_borders[1:maze_height+1,1:maze_width+1] = maze # fill centre with maze
    
    
    return maze_with_borders

def show(maze):
    """
    Plots array with start and ending points highlighted.

    """
    maze[-2,-2] = 4 
    maze[1,1] = 3 
    plt.axis('off')
    plt.imshow(maze) 
    
maze_init = recursive_maze(maze_frame)
maze = finalise_maze(maze_init)
show(maze)
            
