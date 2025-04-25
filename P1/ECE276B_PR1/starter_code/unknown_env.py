from utils import *
import numpy as np
from itertools import product


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
DOOR_1 = 4 # If door 1 is open
DOOR_2 = 5 # If door 2 is open
KEY_POS = 6 # Key Position
GOAL_POS = 7 # Goal Position

# cost
MF_COST = 1.0
TL_COST = 1.0
TR_COST = 1.0
PK_COST = 1.0
UD_COST = 1.0

# Possible Directions
RIGHT = (1,0)
LEFT = (-1,0)
UP = (0,-1)
DOWN = (0,1)
DIR_MAP = {
    (1, 0): 0,   # RIGHT
    (0, 1): 1,   # DOWN
    (0, -1): 2,  # UP
    (-1, 0): 3   # LEFT
}
DIR_LIST = [(1, 0), (0, 1), (0, -1), (-1, 0)]

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

# DOOR position
DOOR_POS = ((5,3), (5,7))

# WALL position
WALL_POS = ((5,0), (5,1), (5,2), (5,4), (5,5), (5,6), (5,8), (5,9))

# KEY position
KEY_POSITION = [(2,2), (2,3), (1,6)]

# GOAL position
GOAL_POSITION = [(6, 1), (7, 3), (6, 6)]

# Initial agent Pose
INIT_POS = (4,8)
INIT_DIR = UP

# T = number of states
NUM_STATE = 92 * 4 * 2 * 2 * 2 * 3 * 3

# Height and Width
HEIGHT = 10
WIDTH = 10

def make_node(t, pos, dir, key, door_1, door_2, key_pos, goal_pos):
    return (t, tuple(pos), tuple(dir), key, door_1, door_2, tuple(key_pos), tuple(goal_pos))

def make_state(pos, dir, key, door_1, door_2, key_pos, goal_pos):
    return (tuple(pos), tuple(dir), key, door_1, door_2, tuple(key_pos), tuple(goal_pos))

def make_node_from_state(state, t):
    pos, dir, key, door_1, door_2, key_pos, goal_pos = state
    return make_node(t, pos, dir, key, door_1, door_2, key_pos, goal_pos)


def encode_node(node):
    t, pos, dir, key, door1, door2, key_pos, goal_pos = node
    x, y = pos
    dx, dy = dir
    dir_idx = DIR_MAP[(dx, dy)]
    key_idx = int(key)
    door1_idx = int(door1)
    door2_idx = int(door2)
    key_pos_idx = KEY_POSITION.index(key_pos)
    goal_pos_idx = GOAL_POSITION .index(goal_pos)

    index = (
        t * (100 * 4 * 2 * 2 * 2 * 3 * 3) +
        (x * 10 + y) * (4 * 2 * 2 * 2 * 3 * 3) +
        dir_idx * (2 * 2 * 2 * 3 * 3) +
        key_idx * (2 * 2 * 3 * 3) +
        door1_idx * (2 * 3 * 3) +
        door2_idx * (3 * 3) +
        key_pos_idx * 3 +
        goal_pos_idx
    )
    return index


def decode_node(index):
    base = index

    goal_pos_idx = base % 3
    base //= 3
    key_pos_idx = base % 3
    base //= 3
    door2_idx = base % 2
    base //= 2
    door1_idx = base % 2
    base //= 2
    key_idx = base % 2
    base //= 2
    dir_idx = base % 4
    base //= 4
    pos_idx = base % 100
    base //= 100
    t = base

    x = pos_idx // 10
    y = pos_idx % 10
    dx, dy = DIR_LIST[dir_idx]
    key_pos = KEY_POSITION[key_pos_idx]
    goal_pos = GOAL_POSITION[goal_pos_idx]
    node = (
        t,
        (x, y),
        (dx, dy),
        key_idx,
        door1_idx,
        door2_idx,
        key_pos,
        goal_pos
    )
    return node


class UnknownPolicy(object):
    def __init__(self, t=NUM_STATE):
        self.t = 0
        self.T = t
        self.value = {}
        self.state = []
        self.policy = {}
        self.init_node = []
        self.init_state = []
        self.__init_agent()
        self.__init_state_space()
        self.__init_value()

    def __init_agent(self):
        # initalize initial state
        agent_pos = INIT_POS
        agent_dir = INIT_DIR
        is_key = False

        # possible key and goal position
        for key_pos in KEY_POSITION:
            for goal_pos in GOAL_POSITION:
                for door_1, door_2 in product([0, 1], repeat=2):
                    state = make_state(agent_pos, agent_dir, is_key, door_1, door_2, key_pos, goal_pos)
                    node = make_node(self.t, agent_pos, agent_dir, is_key, door_1, door_2, key_pos, goal_pos)
                    self.init_node.append(node)
                    self.init_state.append(state)

    def __init_state_space(self):
        # initialize state space (doesn't include WALL)
        height = HEIGHT
        width = WIDTH
        for col in range(width):
            for row in range(height):
                if (col, row) in WALL_POS:
                    continue
                # create states
                pos = (col, row)
                for dir in DIR_LIST:
                    for key_pos in KEY_POSITION:
                        for goal_pos in GOAL_POSITION:
                            for is_key, door_1, door_2 in product([0, 1], repeat=3):
                                state = make_state(pos, dir, is_key, door_1, door_2, key_pos, goal_pos)
                                self.state.append(state)

    def __init_value(self):
        # initialize value
        # initial node
        '''for i in range(self.t, self.T):
            for state in self.init_state:
                node = make_node_from_state(state, i)
                self.value[encode_node(node)] = 0.0'''
        for state in self.init_state:
            node0 = make_node_from_state(state, 0)
            self.value[encode_node(node0)] = 0.0
        # other node
        for state in self.state:
            if state in self.init_state:
                continue
            # at t=0 is inf
            node = make_node_from_state(state, self.t)
            self.value[encode_node(node)] = np.inf
            # at t=1 invalid transition is inf
            node = make_node_from_state(state, self.t+1)
            self.value[encode_node(node)] = np.inf
        # valid transition at t=1
        for node in self.init_node:
            for u in [MF,TL,TR,PK,UD]:
                    next_node, cost = self.transition(node, u)
                    if cost < np.inf:
                        if self.value.get(encode_node(next_node), np.inf) > cost:
                            self.value[encode_node(next_node)] = cost
                            self.policy[encode_node(next_node)] = u

    def transition(self, node, u):
        assert u in [MF, TL, TR, PK, UD], 'Invalid action'
        assert node[POS] not in WALL_POS, 'Invalid state'
        #assert 0 <= node[POS][0] < WIDTH and 0 <= node[POS][1] < HEIGHT, 'Invalid state'
        # state info
        t = node[TIME]
        agent_pos = np.asarray(node[POS]) if not isinstance(node[POS], np.ndarray) else node[POS]
        agent_dir = np.asarray(node[DIR]) if not isinstance(node[DIR], np.ndarray) else node[DIR]
        is_carrying = node[ISKEY]
        is_door1_open = node[DOOR_1]
        is_door2_open = node[DOOR_2]
        key_pos = node[KEY_POS]
        goal_pos = node[GOAL_POS]

        # MF
        if u == MF:
            front_pos = agent_pos + agent_dir
            next_node = make_node(t+1, front_pos, agent_dir, is_carrying, is_door1_open, is_door2_open, key_pos, goal_pos)
            # out boundary
            if not (0 <= front_pos[0] < WIDTH and 0 <= front_pos[1] < HEIGHT):
                return next_node, np.inf

            # check wall
            if tuple(front_pos) in WALL_POS:
                return next_node, np.inf
            # check key
            elif (tuple(front_pos) == key_pos) and (not is_carrying):
                return next_node, np.inf
            # check door 1
            elif tuple(front_pos) == DOOR_POS[0]:
                if not is_door1_open:
                    return next_node, np.inf
            # check door 2
            elif tuple(front_pos) == DOOR_POS[1]:
                if not is_door2_open:
                    return next_node, np.inf
            return next_node, MF_COST

        # TL
        elif u == TL:
            next_dir = TL_MAP[(agent_dir[0], agent_dir[1])]
            next_node = make_node(t + 1, agent_pos, next_dir, is_carrying, is_door1_open, is_door2_open, key_pos, goal_pos)
            return next_node, TL_COST
        # TR
        elif u == TR:
            next_dir = TR_MAP[(agent_dir[0], agent_dir[1])]
            next_node = make_node(t + 1, agent_pos, next_dir, is_carrying, is_door1_open, is_door2_open, key_pos, goal_pos)
            return next_node, TR_COST
        # PK
        elif u == PK:
            front_pos = agent_pos + agent_dir
            next_node = make_node(t+1, agent_pos, agent_dir, 1, is_door1_open, is_door2_open, key_pos, goal_pos)
            # cant pick key again
            if is_carrying:
                return next_node, np.inf
            # check key position
            if tuple(front_pos) == key_pos:
                return next_node, PK_COST
            return next_node, np.inf
        # UD
        elif u == UD:
            front_pos = agent_pos + agent_dir
            next_node_1 = make_node(t + 1, agent_pos, agent_dir, is_carrying, 1, is_door2_open, key_pos, goal_pos)
            next_node_2 = make_node(t + 1, agent_pos, agent_dir, is_carrying, is_door1_open, 1, key_pos, goal_pos)
            # check door 1
            if tuple(front_pos) == DOOR_POS[0]:
                if (not is_door1_open) and (is_carrying):
                    return next_node_1, UD_COST
                else:
                    return next_node_1, np.inf
            # check door 2
            elif tuple(front_pos) == DOOR_POS[1]:
                if (not is_door2_open) and (is_carrying):
                    return next_node_2, UD_COST
                else:
                    return next_node_2, np.inf
            else:
                next_node = make_node(t + 1, agent_pos, agent_dir, is_carrying, is_door1_open, is_door2_open, key_pos,
                                      goal_pos)
                return next_node, np.inf

    def fdp(self):
        """
        Forward Dynamic Programming

        Returns:
            opt_policy: n sequence of actions
        """
        # run fdp
        for t in range(2, self.T):
            updated = False
            for prev_state in self.state:
                prev_node = make_node_from_state(prev_state, t-1)
                prev_value = self.value.get(encode_node(prev_node), np.inf)
                for u in [MF, TL, TR, PK, UD]:
                    next_node, cost = self.transition(prev_node, u)
                    if cost == np.inf:
                        continue
                    total_cost = prev_value + cost
                    if total_cost < self.value.get(encode_node(next_node), np.inf):
                        self.value[encode_node(next_node)] = total_cost
                        self.policy[encode_node(next_node)] = u
                        updated = True
            if not updated:
                print(f"[Early Stop] FDP converged at t = {t}")
                break
        finite_value = {k: v for k, v in self.value.items() if np.isfinite(v)}
        np.savez_compressed("./output/unknown_sol.npz", value=finite_value, policy=self.policy)


class UnknownEnv(object):
    def __init__(self, env, info):
        self.env = env
        self.info = info
        self.goal_pose = self.info['goal_pos']
        self.key_pose = self.info['key_pos']
        self.is_door1_open = self.info['door_open'][0]
        self.is_door2_open = self.info['door_open'][1]
        self.init_agent_pos = INIT_POS
        self.init_agent_dir = INIT_DIR
        self.__load_policy()
        plot_env(self.env)


    def __load_policy(self):
        # load policy
        data = np.load("./output/unknown_sol.npz", allow_pickle=True)
        self.value = data["value"].item()
        self.policy = data["policy"].item()

    def reverse_transition(self, node, u):
        assert u in [MF, TL, TR, PK, UD], 'Invalid action'
        assert node[POS] not in WALL_POS, 'Invalid state'
        #assert 0 <= node[POS][0] < WIDTH and 0 <= node[POS][1] < HEIGHT, 'Invalid state'
        # state info
        t = node[TIME]
        agent_pos = np.asarray(node[POS]) if not isinstance(node[POS], np.ndarray) else node[POS]
        agent_dir = np.asarray(node[DIR]) if not isinstance(node[DIR], np.ndarray) else node[DIR]
        is_carrying = node[ISKEY]
        is_door1_open = node[DOOR_1]
        is_door2_open = node[DOOR_2]
        key_pos = node[KEY_POS]
        goal_pos = node[GOAL_POS]

        # MF
        if u == MF:
            back_pos = agent_pos - agent_dir
            prev_node = make_node(t - 1, back_pos, agent_dir, is_carrying, is_door1_open, is_door2_open, key_pos,
                                  goal_pos)
            return prev_node
        # TL
        elif u == TL:
            prev_dir = TR_MAP[(agent_dir[0], agent_dir[1])]
            prev_node = make_node(t - 1, agent_pos, prev_dir, is_carrying, is_door1_open, is_door2_open, key_pos,
                                  goal_pos)
            return prev_node
        # TR
        elif u == TR:
            prev_dir = TL_MAP[(agent_dir[0], agent_dir[1])]
            prev_node = make_node(t - 1, agent_pos, prev_dir, is_carrying, is_door1_open, is_door2_open, key_pos,
                                  goal_pos)
            return prev_node
        # PK
        elif u == PK:
            prev_node = make_node(t - 1, agent_pos, agent_dir, 0, is_door1_open, is_door2_open, key_pos, goal_pos)
            return prev_node
        # UD
        elif u == UD:
            front_pos = agent_pos + agent_dir
            if tuple(front_pos) == DOOR_POS[0] and is_door1_open == 1 and is_carrying:
                return make_node(t - 1, agent_pos, agent_dir, is_carrying, 0, is_door2_open, key_pos, goal_pos)
            elif tuple(front_pos) == DOOR_POS[1] and is_door2_open == 1 and is_carrying:
                return make_node(t - 1, agent_pos, agent_dir, is_carrying, is_door1_open, 0, key_pos, goal_pos)
            else:
                raise ValueError("Invalid reverse UD action")

    def extract_seq(self):
        key_pos = self.key_pose
        goal_pos = self.goal_pose
        d1_init = int(self.is_door1_open)
        d2_init = int(self.is_door2_open)

        # initial node
        valid_init = make_node(
            0, self.init_agent_pos, self.init_agent_dir,
            0, d1_init, d2_init, key_pos, goal_pos)

        best_cost, best_traj = np.inf, None

        for idx, v in self.value.items():
            if v == np.inf:
                continue
            node = decode_node(idx)
            t, pos, dir, is_key, d1, d2, k_pos, g_pos = node
            # check key pos and goal pose
            if tuple(pos) != tuple(goal_pos) or tuple(k_pos) != tuple(key_pos):
                continue

            # filter door state: if two doors are closed initially
            if d1_init == 0 and d2_init == 0:
                if (d1, d2) == (0, 0):
                    continue
            else:  # If at least 1 door at begining is open
                if d1 != d1_init or d2 != d2_init:
                    continue

            # backtrack to start state
            path, cur = [], node
            ok = True
            while cur[TIME] > 0:
                enc = encode_node(cur)
                if enc not in self.policy:
                    ok = False
                    break
                a = self.policy[enc]
                path.append(a)
                try:
                    cur = self.reverse_transition(cur, a)
                except Exception:
                    ok = False
                    break

            # any accept valid seq
            #print(list(reversed(path)))
            if ok and cur == valid_init and v < best_cost:
                best_cost = v
                best_traj = list(reversed(path))

        if best_traj is None:
            print("No reachable goal found for current env setup.")
            return []
        return best_traj
