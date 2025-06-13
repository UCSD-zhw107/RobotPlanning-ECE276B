# ECE276B PR2 Spring 2025

## Overview
In this assignment, you will implement and compare the performance of search-based and sampling-based motion planning algorithms on several 3-D environments.

### 1. main.py
This file contains examples of how to load and display the environments and how to call a motion planner and plot the planned path. Feel free to modify this file to fit your specific goals for the project. In particular, you should certainly replace Line 104 with a call to a function which checks whether the planned path intersects the boundary or any of the blocks in the environment.

### 2. Planner.py
This file contains an implementation of a baseline planner. The baseline planner gets stuck in complex environments and is not very careful with collision checking. Modify this file in any way necessary for your own implementation.

### 3. astar.py
This file contains a class defining a node for the A* algorithm as well as an incomplete implementation of A*. Feel free to continue implementing the A* algorithm here or start over with your own approach.

### 4. maps
This folder contains 7 test environments described via a rectangular outer boundary and a list of rectangular obstacles. The start and goal points for each environment are specified in main.py.


## Usage

### main.py

To use either **A Star** or **RRT Star**, please modify the main.py (if __name__ == "__main__" part), you can also sepcify different param for each planner. **You would need to install OMPL to run RRT Star**.


### utils.py

It contains implementation of collision checking and boundary checking for the part 1 of this project

### astar.py

It contains the implementation of the A star algorithm. The results and path images are in /astar folder

### rrt.py

It contains the RRT star algorithm which applies RRTstar from OMPL. The results and path images are in /rrt folder



