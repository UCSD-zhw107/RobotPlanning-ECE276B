# UCSD ECE276B PR3

## Overview
In this assignment, you will implement a controller for a differential-drive car robot to track a trajectory.

## Dependencies
This starter code was tested with: python 3.10, matplotlib 3.9.0, and numpy 1.26.4.
MuJoCo and dm_control are required if you choose to use MuJoCo as your simulator.

## Starter code
### 1. main.py
This file contains examples of how to generate control inputs from a simple P controller and apply the control on a car model. This simple controller does not work well. Your task is to replace the P controller with your own controller using CEC and GPI as described in the project statement.

### 2. utils.py
This file contains code to visualize the desired trajectory, robot's trajectory, and obstacles.

### 3. cec.py
This file provides skeleton code for the CEC algorithm (Part 1 of the project).

### 4. gpi.py
This file provides skeleton code for the GPI algorithm (Part 2 of the project).

### 5. value_function.py
This file provides skeleton code for the value function used by the GPI algorithm (Part 2 of the project).

### 6. mujoco_car.py
This file provides an interface for the MuJoCo simulator.


## Submission

## CEC
To run the cec, simply use `python run_cec.py`

## GPI
- To run the gpi, simply sue `python run_gpi.py`. This is the determinisitc GPI, which only make transition to next state = g(t,et, ut,0). It would work with noise in car dynamic model. 
- In testing, we observed that GPI often performs reasonably well under moderate noise levels (e.g., final errors around 80). However, in some cases—especially under larger noise I guess. Please run the evaluation multiple times to account for variability. If the initial result appears unusually poor, it may due to a high-noise.
- In addition, if you want to try with car dynamic with `noise=Fale`, please change the `q: float=1.0` to `q: float=2.0` in the `GPIConfig` in `gpi,py`. After this modification, you would need to
delete the precompute `statge_costs.npy` and `policy.npy` if you have done so, since the param is changed which needs to recompute stage cost and policy.
- In testing, `q=2.0` would work better for car dynamic model with `noise=Fale`, and `q=1.0` would work better for car dynamic model with `noise=Fale`. 
- I also include the stochatic version (make transition to neighbors with transition probablity) in `stochastic.py`, but will not work unfortunately.


## Result
- I also included some of the results in the `/result` folder which has all the figures and gif I included in the report.
- The gif would be output to `/fig`

