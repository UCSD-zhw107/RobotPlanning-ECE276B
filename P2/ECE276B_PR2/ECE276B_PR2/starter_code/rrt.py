import numpy as np
from ompl import base as ob
from ompl import geometric as og
from utils import is_in_boundary, is_in_collision, check_all_blocks


class MyMotionValidator(ob.MotionValidator):
    def __init__(self, si, check_function, blocks):
        super(MyMotionValidator, self).__init__(si)
        self.check_function = check_function
        self.blocks = blocks
        
    def checkMotion(self, s1, s2):
        from_point = [s1[i] for i in range(3)]  
        to_point = [s2[i] for i in range(3)]
        
        return not self.check_function(from_point, to_point, self.blocks)




class RRT(object):
    def __init__(self, start, goal, blocks, boundary, time_limit=60.0):
        self.start = tuple(round(coord, 1) for coord in start)
        self.goal = tuple(round(coord, 1) for coord in goal)
        self.blocks = blocks
        self.boundary = boundary
        self.time_limit = time_limit
    

    def is_state_valid(self, state):
        s = [state[i] for i in range(3)]
        # check within boundary
        if not is_in_boundary(s, self.boundary):
            return False

        # check within obstable
        if is_in_collision(s, self.blocks):
            return False

        return True
    
    def is_motion_valid(self, state_1, state_2):
        # check if line segment collision free
        return not check_all_blocks(state_1, state_2, self.blocks)
    


    def init_rrt(self):
        # define space
        space = ob.RealVectorStateSpace(3)
        bounds = ob.RealVectorBounds(3)

        # set bounds from boundary
        xmin, ymin, zmin, xmax, ymax, zmax = self.boundary[0][:6]
        bounds.setLow(0, float(xmin))
        bounds.setLow(1, float(ymin))
        bounds.setLow(2, float(zmin))
        bounds.setHigh(0, float(xmax))
        bounds.setHigh(1, float(ymax))
        bounds.setHigh(2, float(zmax))
        space.setBounds(bounds)

        # create space information
        si = ob.SpaceInformation(space)

        # state validity checker
        si.setStateValidityChecker(ob.StateValidityCheckerFn(self.is_state_valid))

        # motion validity checker
        motion_validator = MyMotionValidator(si, check_all_blocks, self.blocks)
        si.setMotionValidator(motion_validator)

        # set start and goal
        start_state = ob.State(space)
        goal_state = ob.State(space)
        for i in range(3):
            start_state[i] = self.start[i]
            goal_state[i] = self.goal[i]

        # set problem definition
        pdef = ob.ProblemDefinition(si)
        pdef.setStartAndGoalStates(start_state, goal_state)

        # define planner
        planner = og.RRT(si)
        planner.setProblemDefinition(pdef)
        planner.setup()

        return planner, pdef, si
    
    
    def extract_path_points(self, path):
        states = path.getStates()
        points = []
        for state in states:
            # extract each point
            point = [state[i] for i in range(3)]
            points.append(point)
        points = np.array(points)
        return points


    def plan(self):
        planner, pdef, si = self.init_rrt()

        # Solve
        solved = planner.solve(self.time_limit)
        if solved:
            path = pdef.getSolutionPath()
            path.interpolate(100)  # for smoother path
            return self.extract_path_points(path)
        else:
            print("[RRT] No solution found.")
            return None
