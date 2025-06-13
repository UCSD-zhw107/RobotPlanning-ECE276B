from utils import *
from minigrid.core.world_object import Wall, Key, Door
import numpy as np

MF = 0  # Move Forward
TL = 1  # Turn Left
TR = 2  # Turn Right
PK = 3  # Pickup Key
UD = 4  # Unlock Door

# state index
TIME = 0 # Time Step
POS = 1 # Position(x,y), x col, y row
DIR = 2 # Direction
ISKEY = 3 # IF agent carry key
ISDOOR = 4 # If door is locked

# cost
MF_COST = 1.0
TL_COST = 1.0
TR_COST = 1.0
PK_COST = 1.0
UD_COST = 1.0

# Possible Directions
ORI = np.array([[1,0],[0,1],[0,-1],[-1,0]])
RIGHT = (1,0)
LEFT = (-1,0)
UP = (0,-1)
DOWN = (0,1)

# TL map
TL_MAP = {
    RIGHT:UP,
    UP:LEFT,
    LEFT:DOWN,
    DOWN:RIGHT
}
# TR map
TR_MAP = {
    RIGHT:DOWN,
    DOWN:LEFT,
    LEFT:UP,
    UP:RIGHT
}

def make_node(t, pos, dir, key, door):
    return (t, tuple(pos), tuple(dir), key, door)

def make_state(pos, dir, key, door):
    return (tuple(pos), tuple(dir), key, door)

class KnownEnv(object):
    def __init__(self, env,info):
        self.env = env
        self.info = info
        self.t = 0
        self.value = {}
        self.state = []
        self.policy = {}
        self.__init_agent()
        self.__init_state_space()
        self.__init_value()
        plot_env(self.env)

    def __init_agent(self):
        # initalize initial state
        agent_pos = self.env.agent_pos
        agent_dir = self.env.dir_vec
        door = self.env.grid.get(self.info["door_pos"][0], self.info["door_pos"][1])
        is_open = door.is_open
        is_carrying = self.env.carrying is not None

        self.init_state = make_state(agent_pos, agent_dir, is_carrying, is_open)
        self.init_node = make_node(self.t, agent_pos, agent_dir, is_open, is_open)


    def __init_value(self):
        # initialize value
        # initial state
        agent_pos = self.init_node[POS]
        agent_dir = self.init_node[DIR]
        is_carrying = self.init_node[ISKEY]
        is_open = self.init_node[ISDOOR]
        # initial node
        for i in range(self.t, self.T):
            node = make_node(i,agent_pos,agent_dir,is_carrying,is_open)
            self.value[node] = 0.0
        # other node
        for state in self.state:
            if state == self.init_state:
                continue
            # at t=0 is inf
            node = make_node(self.t, state[0], state[1], state[2], state[3])
            self.value[node] = np.inf
            # at t=1 invalid transition is inf
            node = make_node(self.t+1, state[0], state[1], state[2], state[3])
            self.value[node] = np.inf
        # valid transition at t=1
        for u in [MF,TL,TR,PK,UD]:
            next_node, cost = self.transition(self.init_node, u)
            if cost < np.inf:
                if self.value.get(next_node, np.inf) > cost:
                    self.value[next_node] = cost
                    self.policy[next_node] = u

    def __init_state_space(self):
        # initialize state space (doesn't include WALL)
        self.T = 0
        height = self.info["height"]
        width = self.info["width"]
        for col in range(width):
            for row in range(height):
                cell = self.env.grid.get(col, row)
                if not isinstance(cell, Wall):
                    pose = np.array([col,row])
                    # possible directions
                    for dir in ORI:
                        # possible key and door
                        s1 = make_state(pose, dir, 0, 0)
                        s2 = make_state(pose, dir, 1, 0)
                        s3 = make_state(pose, dir, 0, 1)
                        s4 = make_state(pose, dir, 1, 1)
                        self.state.append(s1)
                        self.state.append(s2)
                        self.state.append(s3)
                        self.state.append(s4)
                        self.T += 4
        self.T -= 1


    def transition(self, node, u):
        assert u in [MF, TL, TR, PK, UD], 'Invalid transition'
        t = node[TIME]

        is_carrying = node[ISKEY]
        is_open = node[ISDOOR]

        agent_pos = np.asarray(node[POS]) if not isinstance(node[POS], np.ndarray) else node[POS]
        agent_dir = np.asarray(node[DIR]) if not isinstance(node[DIR], np.ndarray) else node[DIR]

        # check position
        agent_grid = self.env.grid.get(agent_pos[0], agent_pos[1])
        assert not isinstance(agent_grid, Wall), 'Invalid State, Agent is in WALL'

        # MF
        if u == MF:
            front_pos = agent_pos + agent_dir
            next_node = make_node(t+1, front_pos, agent_dir, is_carrying, is_open)
            front_grid = self.env.grid.get(front_pos[0], front_pos[1])
            # check wall
            if isinstance(front_grid, Wall):
                return next_node, np.inf
            # check key
            elif isinstance(front_grid, Key):
                return next_node, np.inf
            # check door
            elif isinstance(front_grid, Door):
                if not is_open:
                    return next_node, np.inf
            return next_node, MF_COST
        # TL
        elif u == TL:
            next_dir = TL_MAP[(agent_dir[0], agent_dir[1])]
            next_node = make_node(t+1, agent_pos, np.array([next_dir[0], next_dir[1]]), is_carrying, is_open)
            return next_node, TL_COST
        # TR
        elif u == TR:
            next_dir = TR_MAP[(agent_dir[0], agent_dir[1])]
            next_node = make_node(t + 1, agent_pos, np.array([next_dir[0], next_dir[1]]), is_carrying, is_open)
            return next_node, TR_COST
        # PK
        elif u == PK:
            front_pos = agent_pos + agent_dir
            front_grid = self.env.grid.get(front_pos[0], front_pos[1])
            next_node = make_node(t+1, agent_pos, agent_dir, 1, is_open)
            # check key
            if isinstance(front_grid, Key):
                return next_node, PK_COST
            return next_node, np.inf
        # UD
        elif u == UD:
            front_pos = agent_pos + agent_dir
            front_grid = self.env.grid.get(front_pos[0], front_pos[1])
            next_node = make_node(t + 1, agent_pos, agent_dir, is_carrying, 1)
            # check door
            if isinstance(front_grid, Door):
                if (not is_open) and (is_carrying):
                    return next_node, UD_COST
            return next_node, np.inf

    def reverse_transition(self, node, u):
        assert u in [MF, TL, TR, PK, UD]

        t = node[TIME]
        agent_pos = np.asarray(node[POS]) if not isinstance(node[POS], np.ndarray) else node[POS]
        agent_dir = np.asarray(node[DIR]) if not isinstance(node[DIR], np.ndarray) else node[DIR]
        is_carrying = node[ISKEY]
        is_open = node[ISDOOR]

        # check position
        agent_grid = self.env.grid.get(agent_pos[0], agent_pos[1])
        assert not isinstance(agent_grid, Wall), 'Invalid State, Agent is in WALL'
        prev_node = None
        # MF
        if u == MF:
            back_pos = agent_pos - agent_dir
            prev_node = make_node(t - 1, back_pos, agent_dir, is_carrying, is_open)
        # TL
        elif u == TL:
            prev_dir = TR_MAP[(agent_dir[0], agent_dir[1])]
            prev_node = make_node(t - 1, agent_pos, np.array([prev_dir[0], prev_dir[1]]), is_carrying, is_open)
        # TR
        elif u == TR:
            prev_dir = TL_MAP[(agent_dir[0], agent_dir[1])]
            prev_node = make_node(t - 1, agent_pos, np.array([prev_dir[0], prev_dir[1]]), is_carrying, is_open)
        # PK
        elif u == PK:
            prev_node = make_node(t - 1, agent_pos, agent_dir, 0, is_open)
        # UD
        elif u == UD:
            prev_node = make_node(t - 1, agent_pos, agent_dir, is_carrying, 0)
        return prev_node

    def fdp(self):
        """
        Forward Dynamic Programming

        Returns:
            opt_policy: n sequence of actions
        """
        # run fdp
        for t in range(2, self.T):
            for prev_state in self.state:
                prev_node = make_node(t-1, prev_state[0], prev_state[1], prev_state[2], prev_state[3])
                prev_value = self.value.get(prev_node, np.inf)
                for u in [MF, TL, TR, PK, UD]:
                    next_node, cost = self.transition(prev_node, u)
                    if cost == np.inf:
                        continue
                    total_cost = prev_value + cost
                    if total_cost < self.value.get(next_node, np.inf):
                        self.value[next_node] = total_cost
                        self.policy[next_node] = u
        # extract trajectory
        traj = self.extract_optimal_trajectory()
        return traj

    def is_goal(self, node):
        goal = self.info['goal_pos']
        node_pos = node[POS]
        return (node_pos[0] == goal[0] and node_pos[1] == goal[1])

    def extract_optimal_trajectory(self):
        # find reachable goal
        min_cost = np.inf
        best_goal_node = None
        for node in self.value:
            if self.is_goal(node) and self.value[node] < min_cost:
                min_cost = self.value[node]
                best_goal_node = node
        if best_goal_node is None:
            print("No reachable goal found.")
            return []
        # build action sequence
        traj = []
        node = best_goal_node
        while node in self.policy:
            a = self.policy[node]
            traj.insert(0, a)
            node = self.reverse_transition(node, a)
        return traj

